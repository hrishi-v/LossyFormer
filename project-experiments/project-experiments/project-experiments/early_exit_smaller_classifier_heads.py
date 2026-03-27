import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, DataCollatorWithPadding, AutoTokenizer
from torch.utils.data import DataLoader
from chop import MaseGraph
from chop.tools import get_tokenized_dataset
import math
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset


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
# === Data ===

raw = load_dataset("glue", "mnli")
raw = raw.filter(lambda x: x["label"] >= 0)
raw["test"] = raw["validation_matched"]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CKPT)
dataset = raw.map(
    lambda x: tokenizer(x["premise"], x["hypothesis"], truncation=True, padding="max_length", max_length=128),
    batched=True,
)

from transformers import default_data_collator

collator = default_data_collator

def make_loader(split):
    cols_to_remove = [c for c in dataset[split].column_names if c not in ["input_ids", "attention_mask", "label"]]
    ds = dataset[split].remove_columns(cols_to_remove).rename_column("label", "labels")
    return ds

train_loader = make_loader("train")
train_dataloader = DataLoader(train_loader, batch_size=32, shuffle=True, collate_fn=collator)

eval_loader = make_loader("test").train_test_split(test_size=0.5, seed=42)# Use only a subset of the eval set for faster experimentation
val_dataloader = DataLoader(eval_loader["train"], batch_size=64, collate_fn=collator)
test_dataloader = DataLoader(eval_loader["test"], batch_size=64, collate_fn=collator)


@torch.no_grad()
def eval_accuracy(model, dataloader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(dataloader, desc="Eval Accuracy"):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    acc = correct / total
    print(f"Accuracy: {acc * 100:.2f}% ({correct}/{total})")
    return acc
def eval_accuracy_head(model, dataloader, device="cuda", head_idx=0):
    model.eval()
    correct, total = 0, 0
    for batch in tqdm(dataloader, desc=f"Eval Accuracy Head {head_idx}"):
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
    # warmup
    for b in batches[:warmup]:
        model(**{k: v.to(device) for k, v in b.items()})
    if device == "cuda":
        torch.cuda.synchronize()

    # timed
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
    avg_per_batch_ms = np.mean(times) * 1000
    avg_per_sample_ms = (sum(times) / samples) * 1000
    total = sum(times)
    print(f"{samples} samples in {total:.4f}s")
    print(f"Throughput:      {samples / total:.1f} samples/sec")
    print(f"Avg batch:       {np.mean(times) * 1000:.2f} ms")
    print(f"Avg per-sample:  {total / samples * 1000:.2f} ms")
    return [avg_per_batch_ms, avg_per_sample_ms]


import copy
class EarlyExitBert(nn.Module):
    def __init__(self, original_model, threshold=0.9):
        super().__init__()
        # Extracted submodules 
        self.bert = original_model.bert
        print(self.bert)
        
        # In a generic module graph, children might not be inside a ModuleList with length
        self.encoder_layers = nn.ModuleList(list(self.bert.encoder.layer.children()))
        self.num_layers = len(self.encoder_layers)
        
        # Construct all classifier heads
        self.classifier_heads = nn.ModuleList([copy.deepcopy(original_model.classifier) for _ in range(self.num_layers)])
        # self.classifier_head = original_model.classifier
        self.classifier_thresholds = [threshold for _ in range(self.num_layers)]
    def forward(self, input_ids, attention_mask, labels=None, return_exit_index=False):
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Standard representation mask
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        
        # Step 1: Embeddings
        hidden_states = self.bert.embeddings(input_ids=input_ids)
        
        final_logits = torch.zeros(batch_size, self.classifier_heads[-1].out_features, device=device)
        
        # We index elements to dynamically shrink batch sizes across layers
        active_mask = extended_attention_mask
        active_indices = torch.arange(batch_size, device=device)
        exit_layer = torch.ones(batch_size, device=device) * -1  # initialize to -1 to indicate no exit yet

        for i, layer_module in enumerate(self.encoder_layers):
            
            # Step 2: Layer Forward
            layer_outputs = layer_module(hidden_states, attention_mask=active_mask)
            hidden_states = layer_outputs[0]
            
            # Step 3: Compute confidence through pooling and classification
            # cls_token = hidden_states[:, 0]
            # pooled_output = self.bert.pooler.dense(cls_token)
            # pooled_output = self.bert.pooler.activation(pooled_output)
            pooled_output = self.bert.pooler(hidden_states)
            
            if i == self.num_layers - 1:
                logits = self.classifier_heads[i](pooled_output)
                exit_layer[active_indices] = i
                final_logits[active_indices] = logits
                break

            if self.classifier_thresholds[i] != 1.0:  # if threshold is 1.0, we skip confidence check and exit nothing at this layer
                
                logits = self.classifier_heads[i](pooled_output)

                probs = F.softmax(logits, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1)
                
                # Step 4: Early Exit Criteria
                confident = max_probs >= self.classifier_thresholds[i]
                
                # Store completed outputs
                confident_indices = active_indices[confident]
                final_logits[confident_indices] = logits[confident]
                exit_layer[confident_indices] = i
                
                not_confident = ~confident
                if not not_confident.any():
                    break
            
                # Dynamically shrink tensor blocks to only compute unconfident targets in next layer
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

        # Standard representation mask
        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        
        # Step 1: Embeddings
        hidden_states = self.bert.embeddings(input_ids=input_ids)
        
        final_logits = torch.zeros(batch_size, self.classifier_heads[-1].out_features, device=device)
        
        # We index elements to dynamically shrink batch sizes across layers
        active_mask = extended_attention_mask
        active_indices = torch.arange(batch_size, device=device)

        for i, layer_module in enumerate(self.encoder_layers):
            if i > head_idx:
                break
            
            # Step 2: Layer Forward
            layer_outputs = layer_module(hidden_states, attention_mask=active_mask)
            hidden_states = layer_outputs[0]
            
            # Step 3: Compute confidence through pooling and classification
            # cls_token = hidden_states[:, 0]
            # pooled_output = self.bert.pooler.dense(cls_token)
            # pooled_output = self.bert.pooler.activation(pooled_output)
        
        pooled_output = self.bert.pooler(hidden_states)

        final_logits = self.classifier_heads[head_idx](pooled_output)

        return {"logits": final_logits}


    def finetune_classifier_head(self, head_idx, train_dataloader, device):
        # Freeze all layers except the specified classifier head
        for param in self.bert.parameters():
            param.requires_grad = False
        for i, classifier_head in enumerate(self.classifier_heads):
            if i == head_idx:
                for param in classifier_head.parameters():
                    param.requires_grad = True
            else:
                for param in classifier_head.parameters():
                    param.requires_grad = False
        # Just use loss.backward() and an optimizer step on the specified head
        optimizer = torch.optim.Adam(self.classifier_heads[head_idx].parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for batch in tqdm(train_dataloader, desc=f"Finetuning head {head_idx}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.forward_no_exit_criteria(batch["input_ids"], batch["attention_mask"], head_idx=head_idx)
            logits = outputs["logits"]
            loss = loss_fn(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Based on the final loss, set the threshold for the layer
    def set_threshold(self, threshold, head_idx=None):
        if head_idx is None:
            self.classifier_thresholds = [threshold for _ in range(self.num_layers)]
            return
        self.classifier_thresholds[head_idx] = threshold
        
    @property
    def num_heads(self):
        return self.num_layers


# Baseline
print("=== BASELINE ===")
mg = MaseGraph.from_checkpoint(MODEL_CKPT)
baseline_model = mg.model.to(DEVICE)
baseline_acc = eval_accuracy(baseline_model, test_dataloader)
print("GPU:"); eval_speed(baseline_model, test_dataloader)

THRESHOLD = 0.75
print("\n" + "="*50)
print(f"EVALUATING BERT BASE (Threshold = {THRESHOLD})")
print("="*50)

hf_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
hf_model.load_state_dict(baseline_model.state_dict(), strict=False)
hf_model = hf_model.to(DEVICE)


early_exit_bert = EarlyExitBert(hf_model, threshold=THRESHOLD).to(DEVICE)
early_exit_bert_finetuned = copy.deepcopy(early_exit_bert)


eval_accuracy(early_exit_bert, test_dataloader, device=DEVICE)
eval_speed(early_exit_bert, test_dataloader, device=DEVICE)

import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Finetune each head independently with exit criteria disabled, to get a better starting point for the search. This is optional but can lead to better results than starting from the original BERT head weights, which were not trained with early exits in mind.
for head_idx in range(early_exit_bert_finetuned.num_heads):
    print("\n" + "="*50)
    print(f"Finetuning head {head_idx} with exit criteria disabled")
    print("="*50)
    early_exit_bert_finetuned.finetune_classifier_head(head_idx, train_dataloader, DEVICE)


# ── 1. Reset all thresholds before search ────────────────────────────────────
for head_idx in range(early_exit_bert_finetuned.num_heads):
    early_exit_bert_finetuned.set_threshold(0.5, head_idx=head_idx)

# ── 2. Bayesian optimisation over joint threshold vector ─────────────────────
thresholds      = [0.3, 0.5, 0.65, 0.7, 0.73, 0.75, 0.77, 0.8, 0.9]
latency_budget  = 50   # ms per batch, TODO: Should be in terms of baseline latency, e.g., 50% of baseline latency, to be more generalizable across different hardware setups

def objective(trial):
    for head_idx in range(early_exit_bert_finetuned.num_heads):
        t = trial.suggest_categorical(f"threshold_{head_idx}", thresholds)
        early_exit_bert_finetuned.set_threshold(t, head_idx=head_idx)

    latency = eval_speed(early_exit_bert_finetuned, val_dataloader, device=DEVICE)[0]
    if latency > latency_budget:
        return 0.0                          # hard constraint

    return eval_accuracy(early_exit_bert_finetuned, val_dataloader, device=DEVICE)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, show_progress_bar=True)

print(f"\nBest validation accuracy : {study.best_value:.4f}")
print(f"Best thresholds          : {study.best_params}")

# best_params = {'threshold_0': 0.9, 'threshold_1': 0.77, 'threshold_2': 0.5, 'threshold_3': 0.73, 'threshold_4': 0.7, 'threshold_5': 0.75, 'threshold_6': 0.9, 'threshold_7': 0.8, 'threshold_8': 0.9, 'threshold_9': 0.65, 'threshold_10': 0.73, 'threshold_11': 0.65}

# thresholds = [best_params[f"threshold_{i}"] for i in range(early_exit_bert_finetuned.num_heads)]

# Apply best thresholds
for head_idx in range(early_exit_bert_finetuned.num_heads):
    t = study.best_params[f"threshold_{head_idx}"]
    early_exit_bert_finetuned.set_threshold(t, head_idx=head_idx)

# ── 3. Profile exit rates to find heads worth keeping ────────────────────────
early_exit_bert_finetuned.eval()

exit_counts = [0] * early_exit_bert_finetuned.num_heads
total        = 0

with torch.no_grad():
    for batch in val_dataloader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)
        batch_size     = input_ids.size(0)
        total         += batch_size

        # Run sample-by-sample so we can record which head each exits at.
        # Adjust this block if your model exposes exit indices in batch mode.
        for i in range(batch_size):
            ids  = input_ids[i].unsqueeze(0)
            mask = attention_mask[i].unsqueeze(0)

            _, exit_head = early_exit_bert_finetuned(
                ids, mask, return_exit_index=True   # <-- expose this in your model
            )
            for exit_idx in exit_head:
                exit_counts[int(exit_idx.item())] += 1
            
exit_rates = [c / total for c in exit_counts]

print("\nExit-rate profile after optimisation:")
for i, rate in enumerate(exit_rates):
    bar = "█" * int(rate * 40)
    print(f"  Head {i:>2d} | threshold {study.best_params[f'threshold_{i}']:.2f} "
          f"| exit rate {rate:.3f}  {bar}")

# ── 4. Prune heads that contribute nothing ───────────────────────────────────
MIN_EXIT_RATE = 0.01   # heads below 1 % exit rate are considered dead

dead_heads = [i for i, r in enumerate(exit_rates) if r < MIN_EXIT_RATE]
if dead_heads:
    print(f"\nPruning dead heads (exit rate < {MIN_EXIT_RATE}): {dead_heads}")
    # Setting threshold to 1.0 effectively disables the branch — nothing is
    # ever confident enough to exit, so every sample passes straight through.
    for i in dead_heads:
        early_exit_bert_finetuned.set_threshold(1.0, head_idx=i)
else:
    print("\nNo dead heads found — all branches are active.")

# ── 5. Final evaluation on test set ──────────────────────────────────────────
print("\n" + "="*50)
print("Final evaluation on test set")
print("="*50)
eval_accuracy(early_exit_bert_finetuned, test_dataloader, device=DEVICE)
eval_speed(early_exit_bert_finetuned,    test_dataloader, device=DEVICE)

# We have a search space: 5 thresholds for each of the 7 heads, which results in 5^7 = 78,125 possible combinations
# We want to optimize for the best accuracy while keeping latency under a certain budget (e.g., 20ms per batch)
# Use an optimisation algorithm like Bayesian optimization or a genetic algorithm to efficiently search through the threshold combinations and find the optimal set of thresholds that maximize accuracy while meeting the latency constraint.

early_exit_bert_finetuned.eval()

print("\n" + "="*50)
print(f"Evaluating Model after finetuning...")
print("="*50)
eval_accuracy(early_exit_bert_finetuned, test_dataloader, device=DEVICE)
eval_speed(early_exit_bert_finetuned, test_dataloader, device=DEVICE)


