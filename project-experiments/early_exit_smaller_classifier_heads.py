"""
This script implements and evaluates an Early-Exit BERT model for sequence classification.
It replaces the standard BERT structure with a multi-head architecture where each encoder layer 
is attached to a classifier head. The model dynamically exits the forward pass when a 
confidence threshold is met, optimizing for inference latency while maintaining accuracy.
The script includes routines for:
- Data loading and preprocessing for GLUE/MNLI tasks.
- Finetuning internal classifier heads.
- Bayesian optimization (via Optuna) to find optimal per-layer confidence thresholds.
- Profiling exit rates and pruning inactive heads.
"""
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import optuna
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator
)
from chop import MaseGraph

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print("Using CUDA device:", props.name)
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available — using CPU")

MODEL_CKPT = "./bert-base-glue-mnli-baseline"
TOKENIZER_CKPT = "bert-base-uncased"
NUM_LABELS = 3

raw = load_dataset("glue", "mnli")
raw = raw.filter(lambda x: x["label"] >= 0)
raw["test"] = raw["validation_matched"]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CKPT)
dataset = raw.map(
    lambda x: tokenizer(x["premise"], x["hypothesis"], truncation=True, padding="max_length", max_length=128),
    batched=True,
)

def make_loader(split):
    cols_to_remove = [c for c in dataset[split].column_names if c not in ["input_ids", "attention_mask", "label"]]
    ds = dataset[split].remove_columns(cols_to_remove).rename_column("label", "labels")
    return ds

train_loader = make_loader("train")
train_dataloader = DataLoader(train_loader, batch_size=32, shuffle=True, collate_fn=default_data_collator)

eval_loader = make_loader("test").train_test_split(test_size=0.5, seed=42)
val_dataloader = DataLoader(eval_loader["train"], batch_size=64, collate_fn=default_data_collator)
test_dataloader = DataLoader(eval_loader["test"], batch_size=64, collate_fn=default_data_collator)


@torch.no_grad()
def eval_accuracy(model, dataloader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(dataloader, desc="Eval Accuracy", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    acc = correct / total
    print(f"Accuracy: {acc * 100:.2f}% ({correct}/{total})")
    return acc

@torch.no_grad()
def eval_accuracy_head(model, dataloader, device="cuda", head_idx=0):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(dataloader, desc=f"Eval Head {head_idx}", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.forward_no_exit_criteria(**batch, head_idx=head_idx)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    acc = correct / total
    print(f"Head {head_idx} Accuracy: {acc * 100:.2f}% ({correct}/{total})")
    return acc

@torch.no_grad()
def eval_speed(model, dataloader, device="cuda", num_batches=100, warmup=10):
    model.eval()
    batches = list(dataloader)[:warmup + num_batches]
    for b in batches[:warmup]:
        model(**{k: v.to(device) for k, v in b.items()})
    if device == "cuda":
        torch.cuda.synchronize()

    times, samples = [], 0
    for b in batches[warmup:]:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(**{k: v.to(device) for k, v in b.items()})
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        samples += b["input_ids"].size(0)
    
    total_time = sum(times)
    avg_per_batch_ms = np.mean(times) * 1000
    avg_per_sample_ms = (total_time / samples) * 1000
    print(f"Throughput: {samples / total_time:.1f} samples/sec | Latency: {avg_per_sample_ms:.2f} ms/sample")
    return [avg_per_batch_ms, avg_per_sample_ms]


import copy
class EarlyExitBert(nn.Module):
    def __init__(self, original_model, threshold=0.9):
        super().__init__()
        self.bert = original_model.bert
        self.encoder_layers = nn.ModuleList(list(self.bert.encoder.layer.children()))
        self.num_layers = len(self.encoder_layers)
        self.classifier_heads = nn.ModuleList([copy.deepcopy(original_model.classifier) for _ in range(self.num_layers)])
        self.classifier_thresholds = [threshold for _ in range(self.num_layers)]

    def forward(self, input_ids, attention_mask, labels=None, return_exit_index=False):
        device = input_ids.device
        batch_size = input_ids.size(0)
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        hidden_states = self.bert.embeddings(input_ids=input_ids)
        
        final_logits = torch.zeros(batch_size, self.classifier_heads[-1].out_features, device=device)
        active_mask = extended_attention_mask
        active_indices = torch.arange(batch_size, device=device)
        exit_layer = torch.ones(batch_size, device=device) * -1

        for i, layer_module in enumerate(self.encoder_layers):
            layer_outputs = layer_module(hidden_states, attention_mask=active_mask)
            hidden_states = layer_outputs[0]
            pooled_output = self.bert.pooler(hidden_states)
            
            if i == self.num_layers - 1:
                logits = self.classifier_heads[i](pooled_output)
                exit_layer[active_indices] = i
                final_logits[active_indices] = logits
                break

            if self.classifier_thresholds[i] != 1.0:
                logits = self.classifier_heads[i](pooled_output)
                probs = F.softmax(logits, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1)
                confident = max_probs >= self.classifier_thresholds[i]
                
                confident_indices = active_indices[confident]
                final_logits[confident_indices] = logits[confident]
                exit_layer[confident_indices] = i
                
                not_confident = ~confident
                if not not_confident.any():
                    break
            
                active_indices = active_indices[not_confident]
                hidden_states = hidden_states[not_confident]
                active_mask = active_mask[not_confident]
        
        if return_exit_index:
            return {"logits": final_logits}, exit_layer
        return {"logits": final_logits}

    def forward_no_exit_criteria(self, input_ids, attention_mask, labels=None, head_idx=-1):
        if head_idx == -1:
            head_idx = self.num_layers - 1
        device = input_ids.device
        batch_size = input_ids.size(0)
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        hidden_states = self.bert.embeddings(input_ids=input_ids)
        active_mask = extended_attention_mask

        for i, layer_module in enumerate(self.encoder_layers):
            if i > head_idx:
                break
            layer_outputs = layer_module(hidden_states, attention_mask=active_mask)
            hidden_states = layer_outputs[0]
        
        pooled_output = self.bert.pooler(hidden_states)
        final_logits = self.classifier_heads[head_idx](pooled_output)
        return {"logits": final_logits}

    def finetune_classifier_head(self, head_idx, train_dataloader, device):
        for param in self.bert.parameters():
            param.requires_grad = False
        for i, classifier_head in enumerate(self.classifier_heads):
            for param in classifier_head.parameters():
                param.requires_grad = (i == head_idx)

        optimizer = torch.optim.Adam(self.classifier_heads[head_idx].parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for batch in tqdm(train_dataloader, desc=f"Finetuning head {head_idx}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.forward_no_exit_criteria(batch["input_ids"], batch["attention_mask"], head_idx=head_idx)
            logits = outputs["logits"]
            loss = loss_fn(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def set_threshold(self, threshold, head_idx=None):
        if head_idx is None:
            self.classifier_thresholds = [threshold for _ in range(self.num_layers)]
        else:
            self.classifier_thresholds[head_idx] = threshold
        
    @property
    def num_heads(self):
        return self.num_layers


if __name__ == "__main__":
    print("=== BASELINE ===")
    mg = MaseGraph.from_checkpoint(MODEL_CKPT)
    baseline_model = mg.model.to(DEVICE)
    eval_accuracy(baseline_model, test_dataloader)
    eval_speed(baseline_model, test_dataloader)

    hf_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
    hf_model.load_state_dict(baseline_model.state_dict(), strict=False)
    hf_model = hf_model.to(DEVICE)

    early_exit_bert = EarlyExitBert(hf_model, threshold=0.75).to(DEVICE)
    early_exit_bert_finetuned = copy.deepcopy(early_exit_bert)

    print("\n=== FINETUNING HEADS ===")
    for head_idx in range(early_exit_bert_finetuned.num_heads):
        early_exit_bert_finetuned.finetune_classifier_head(head_idx, train_dataloader, DEVICE)

    print("\n=== OPTUNA SEARCH ===")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    threshold_options = [0.3, 0.5, 0.65, 0.7, 0.73, 0.75, 0.77, 0.8, 0.9]
    latency_budget = 50 

    def objective(trial):
        for head_idx in range(early_exit_bert_finetuned.num_heads):
            t = trial.suggest_categorical(f"threshold_{head_idx}", threshold_options)
            early_exit_bert_finetuned.set_threshold(t, head_idx=head_idx)
        latency = eval_speed(early_exit_bert_finetuned, val_dataloader, device=DEVICE)[0]
        if latency > latency_budget:
            return 0.0
        return eval_accuracy(early_exit_bert_finetuned, val_dataloader, device=DEVICE)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print(f"Best validation accuracy: {study.best_value:.4f}")
    for head_idx in range(early_exit_bert_finetuned.num_heads):
        t = study.best_params[f"threshold_{head_idx}"]
        early_exit_bert_finetuned.set_threshold(t, head_idx=head_idx)

    print("\n=== EXIT RATE PROFILING ===")
    exit_counts = [0] * early_exit_bert_finetuned.num_heads
    total_samples = 0
    early_exit_bert_finetuned.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            batch_size = input_ids.size(0)
            total_samples += batch_size
            for i in range(batch_size):
                ids, mask = input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0)
                _, exit_head = early_exit_bert_finetuned(ids, mask, return_exit_index=True)
                for exit_idx in exit_head:
                    exit_counts[int(exit_idx.item())] += 1
            
    for i, count in enumerate(exit_counts):
        rate = count / total_samples
        print(f"Head {i:>2d} | Threshold {early_exit_bert_finetuned.classifier_thresholds[i]:.2f} | Exit Rate {rate:.3f}")

    MIN_EXIT_RATE = 0.01
    for i, count in enumerate(exit_counts):
        if (count / total_samples) < MIN_EXIT_RATE:
            early_exit_bert_finetuned.set_threshold(1.0, head_idx=i)

    print("\n=== FINAL EVALUATION ===")
    eval_accuracy(early_exit_bert_finetuned, test_dataloader, device=DEVICE)
    eval_speed(early_exit_bert_finetuned, test_dataloader, device=DEVICE)


