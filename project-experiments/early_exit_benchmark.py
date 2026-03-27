"""
Early Exit Benchmark Suite: BERT and RoBERTa on MNLI

This script provides a standardized benchmarking environment to evaluate the 
accuracy-latency trade-offs of Early Exit models compared to their full-depth baselines.

Key features:
- Entropy-based exit criteria.
- Automated threshold sweeping for Pareto front characterization.
- Support for both BERT and RoBERTa architectures via a shared EarlyExitMixin.
- GPU-synchronized timing for accurate latency measurement.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, default_collate
from datasets import load_dataset
from chop import MaseGraph

# === Configuration ===
# Note: Ensure these paths are accessible or updated for the target environment
BERT_CKPT = "/vol/bitbucket/ug22/adls-data/models/bert-base-glue-mnli-baseline"
ROBERTA_CKPT = "/vol/bitbucket/ug22/adls-data/models/roberta-base-glue-mnli-baseline"
NUM_LABELS = 3
BATCH_SIZE = 32
THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
PLOT_PATH = "benchmark_results.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# === Data Loading ===
def get_dataloader(tokenizer_ckpt):
    print(f"Preparing data for {tokenizer_ckpt}...")
    raw_mnli = load_dataset("glue", "mnli").filter(lambda x: x["label"] >= 0)
    raw_mnli["test"] = raw_mnli["validation_matched"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    def tokenize_fn(x):
        return tokenizer(x["premise"], x["hypothesis"], truncation=True, padding="max_length", max_length=128)
    
    ds = raw_mnli.map(tokenize_fn, batched=True)
    keep = ["input_ids", "attention_mask", "label"]
    ds_test = ds["test"].remove_columns([c for c in ds["test"].column_names if c not in keep]).rename_column("label", "labels")
    
    # Using 50% split for evaluation speed
    ds_split = ds_test.train_test_split(test_size=0.5, seed=42)
    return DataLoader(ds_split["test"], batch_size=BATCH_SIZE, collate_fn=default_collate, shuffle=False)

# === Evaluation Functions ===
@torch.no_grad()
def eval_metrics(model, dataloader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    times = []
    
    # Warmup
    it = iter(dataloader)
    for _ in range(5):
        try: batch = next(it)
        except StopIteration: break
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch)
    
    if "cuda" in str(device): torch.cuda.synchronize()

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if "cuda" in str(device): torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**batch)
        if "cuda" in str(device): torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        
        logits = out["logits"] if isinstance(out, dict) else out.logits
        correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    
    return (correct / total), (np.mean(times) * 1000)

# === Model Implementation ===
class EarlyExitMixin:
    def init_early_exit(self, original_model, threshold):
        self.threshold = threshold
        self.num_labels = original_model.config.num_labels
        self.pooler = None 
        self.classifier = None 

    def compute_logits(self, hidden_states):
        if self.pooler is not None:
            return self.classifier(self.pooler(hidden_states))
        try:
            return self.classifier(hidden_states[:, 0])
        except Exception:
            return self.classifier(hidden_states)

    def evaluate_confidence(self, logits, active_indices, final_logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        confident = entropy <= self.threshold
        final_logits[active_indices[confident]] = logits[confident]
        return ~confident

    def set_threshold(self, threshold):
        self.threshold = threshold

class BertEarlyExit(nn.Module, EarlyExitMixin):
    def __init__(self, original_model, threshold=0.1):
        super().__init__()
        self.init_early_exit(original_model, threshold)
        self.embeddings = original_model.bert.embeddings
        self.layers = original_model.bert.encoder.layer
        self.pooler = original_model.bert.pooler
        self.classifier = original_model.classifier
        self.base_model = original_model.bert
        self.num_layers = len(self.layers)

    def forward(self, input_ids, attention_mask, **kwargs):
        device = input_ids.device
        batch_size = input_ids.size(0)
        hidden_states = self.embeddings(input_ids=input_ids)
        active_mask = self.base_model.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        final_logits = torch.zeros(batch_size, self.num_labels, device=device)
        active_indices = torch.arange(batch_size, device=device)

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, attention_mask=active_mask)
            hidden_states = layer_outputs[0]
            logits = self.compute_logits(hidden_states)
            if i == self.num_layers - 1:
                final_logits[active_indices] = logits
                break
            not_confident = self.evaluate_confidence(logits, active_indices, final_logits)
            if not not_confident.any(): break
            active_indices = active_indices[not_confident]
            hidden_states = hidden_states[not_confident]
            active_mask = active_mask[not_confident]
        return {"logits": final_logits}

class RobertaEarlyExit(nn.Module, EarlyExitMixin):
    def __init__(self, original_model, threshold=0.1):
        super().__init__()
        self.init_early_exit(original_model, threshold)
        self.embeddings = original_model.roberta.embeddings
        self.layers = nn.ModuleList(original_model.roberta.encoder.layer)
        self.classifier = original_model.classifier
        self.base_model = original_model.roberta
        self.num_layers = len(self.layers)

    def compute_logits(self, hidden_states):
        return self.classifier(hidden_states)

    def forward(self, input_ids, attention_mask, **kwargs):
        device = input_ids.device
        batch_size = input_ids.size(0)
        hidden_states = self.embeddings(input_ids=input_ids)
        active_mask = self.base_model.get_extended_attention_mask(attention_mask, input_ids.size(), device)
        final_logits = torch.zeros(batch_size, self.num_labels, device=device)
        active_indices = torch.arange(batch_size, device=device)

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, attention_mask=active_mask)
            hidden_states = layer_outputs[0]
            logits = self.compute_logits(hidden_states)
            if i == self.num_layers - 1:
                final_logits[active_indices] = logits
                break
            not_confident = self.evaluate_confidence(logits, active_indices, final_logits)
            if not not_confident.any(): break
            active_indices = active_indices[not_confident]
            hidden_states = hidden_states[not_confident]
            active_mask = active_mask[not_confident]
        return {"logits": final_logits}

# === Main Experiment ===
def main():
    bert_loader = get_dataloader("bert-base-uncased")
    roberta_loader = get_dataloader("roberta-base")

    print("Loading Models...")
    bert_mg = MaseGraph.from_checkpoint(BERT_CKPT)
    bert_baseline = bert_mg.model.to(DEVICE)
    bert_hf = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS).to(DEVICE)
    bert_hf.load_state_dict(bert_baseline.state_dict(), strict=False)
    bert_ee = BertEarlyExit(bert_hf)

    roberta_mg = MaseGraph.from_checkpoint(ROBERTA_CKPT)
    roberta_baseline = roberta_mg.model.to(DEVICE)
    roberta_hf = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=NUM_LABELS).to(DEVICE)
    roberta_hf.load_state_dict(roberta_baseline.state_dict(), strict=False)
    roberta_ee = RobertaEarlyExit(roberta_hf)

    def run_benchmark(model, baseline, loader, name):
        print(f"\n--- {name} Benchmark ---")
        base_acc, base_lat = eval_metrics(baseline, loader, DEVICE)
        print(f"Baseline: Accuracy={base_acc*100:.2f}%, Latency={base_lat:.2f}ms")

        ee_accs, ee_lats = [], []
        for t in THRESHOLDS:
            model.set_threshold(t)
            acc, lat = eval_metrics(model, loader, DEVICE)
            print(f"Threshold {t}: Accuracy={acc*100:.2f}%, Latency={lat:.2f}ms")
            ee_accs.append(acc * 100)
            ee_lats.append(lat)
        return ee_accs, ee_lats, base_acc * 100, base_lat

    bert_accs, bert_lats, bert_b_acc, bert_b_lat = run_benchmark(bert_ee, bert_baseline, bert_loader, "BERT")
    rob_accs, rob_lats, rob_b_acc, rob_b_lat = run_benchmark(roberta_ee, roberta_baseline, roberta_loader, "RoBERTa")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    def plot_res(ax, lats, accs, b_lat, b_acc, title, color):
        ax.plot(lats, accs, 'o-', label='Early Exit', color=color)
        ax.axhline(y=b_acc, color='red', linestyle='--', label=f'Baseline ({b_acc:.1f}%)')
        ax.axvline(x=b_lat, color='green', linestyle=':', label=f'Base Latency ({b_lat:.1f}ms)')
        for i, t in enumerate(THRESHOLDS):
            ax.annotate(f"{t}", (lats[i], accs[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        ax.set_xlabel('Latency (ms/batch)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_res(ax1, bert_lats, bert_accs, bert_b_lat, bert_b_acc, "BERT Early Exit", "blue")
    plot_res(ax2, rob_lats, rob_accs, rob_b_lat, rob_b_acc, "RoBERTa Early Exit", "purple")

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Benchmark results saved to {PLOT_PATH}")

if __name__ == "__main__":
    main()

