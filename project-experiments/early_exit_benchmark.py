import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset
from chop import MaseGraph
import matplotlib.pyplot as plt

# === Configuration ===
BERT_CKPT = "/vol/bitbucket/ug22/adls-data/models/bert-base-glue-mnli-baseline"
ROBERTA_CKPT = "/vol/bitbucket/ug22/adls-data/models/roberta-base-glue-mnli-baseline"
NUM_LABELS = 3
BATCH_SIZE = 32
THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
PLOT_PATH = "benchmark_results.png"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available — using CPU")

# === Data Loading ===
def get_dataloader(tokenizer_ckpt):
    print(f"Preparing data for {tokenizer_ckpt}...")
    raw_mnli = load_dataset("glue", "mnli")
    raw_mnli = raw_mnli.filter(lambda x: x["label"] >= 0)
    raw_mnli["test"] = raw_mnli["validation_matched"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
    def tokenize_fn(x):
        return tokenizer(x["premise"], x["hypothesis"], truncation=True, padding="max_length", max_length=128)
    
    ds = raw_mnli.map(tokenize_fn, batched=True)
    cols_to_remove = [c for c in ds["test"].column_names if c not in ["input_ids", "attention_mask", "label"]]
    ds_test = ds["test"].remove_columns(cols_to_remove).rename_column("label", "labels")
    
    # Using 50% of the validation matched set as requested by user
    ds_split = ds_test.train_test_split(test_size=0.5, seed=42)
    dataloader = DataLoader(ds_split["test"], batch_size=BATCH_SIZE, collate_fn=default_data_collator)
    return dataloader

# === Evaluation Functions ===
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
    return correct / total

@torch.no_grad()
def eval_speed(model, dataloader, device="cuda", num_batches=100, warmup=10):
    model.eval()
    batches = list(dataloader)[:warmup + num_batches]
    # warmup
    for b in batches[:warmup]:
        model(**{k: v.to(device) for k, v in b.items()})
    
    if "cuda" in str(device):
        torch.cuda.synchronize()

    # timed
    times, samples = [], 0
    for b in batches[warmup:]:
        batch_inputs = {k: v.to(device) for k, v in b.items()}
        if "cuda" in str(device):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(**batch_inputs)
        if "cuda" in str(device):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        samples += b["input_ids"].size(0)
    
    avg_per_batch_ms = np.mean(times) * 1000
    return avg_per_batch_ms

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
        print("confident", confident.sum().item())
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
    # 1. Prepare Data
    bert_loader = get_dataloader("bert-base-uncased")
    roberta_loader = get_dataloader("roberta-base")

    # 2. Setup Models
    print("Loading BERT base...")
    bert_mg = MaseGraph.from_checkpoint(BERT_CKPT)
    bert_baseline = bert_mg.model.to(DEVICE)
    bert_hf = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS).to(DEVICE)
    bert_hf.load_state_dict(bert_baseline.state_dict(), strict=False)
    bert_ee = BertEarlyExit(bert_hf, threshold=2)

    print("Loading RoBERTa base...")
    roberta_mg = MaseGraph.from_checkpoint(ROBERTA_CKPT)
    roberta_baseline = roberta_mg.model.to(DEVICE)
    roberta_hf = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=NUM_LABELS).to(DEVICE)
    roberta_hf.load_state_dict(roberta_baseline.state_dict(), strict=False)
    roberta_ee = RobertaEarlyExit(roberta_hf, threshold=2)

    # 3. Run Benchmarks
    def run_experiment(model, baseline, loader, name):
        print(f"\nBenchmarking {name}...")
        ee_accs, ee_lats = [], []
        for t in THRESHOLDS:
            model.set_threshold(t)
            acc = eval_accuracy(model, loader, DEVICE) * 100
            lat = eval_speed(model, loader, DEVICE)
            print(f"Threshold {t}: Accuracy={acc:.2f}%, Latency={lat:.2f}ms")
            ee_accs.append(acc)
            ee_lats.append(lat)
        
        print(f"Benchmarking {name} Baseline...")
        base_acc = eval_accuracy(baseline, loader, DEVICE) * 100
        base_lat = eval_speed(baseline, loader, DEVICE)
        print(f"Baseline: Accuracy={base_acc:.2f}%, Latency={base_lat:.2f}ms")
        return ee_accs, ee_lats, base_acc, base_lat

    bert_accs, bert_lats, bert_base_acc, bert_base_lat = run_experiment(bert_ee, bert_baseline, bert_loader, "BERT")
    roberta_accs, roberta_lats, roberta_base_acc, roberta_base_lat = run_experiment(roberta_ee, roberta_baseline, roberta_loader, "RoBERTa")

    # 4. Plotting
    print("\nGenerating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # BERT Plot
    ax1.plot(bert_lats, bert_accs, 'o-', label='Early Exit BERT', color='blue')
    ax1.axhline(y=bert_base_acc, color='red', linestyle='--', label=f'Baseline ({bert_base_acc:.2f}%)')
    ax1.axvline(x=bert_base_lat, color='green', linestyle=':', label=f'Baseline Latency ({bert_base_lat:.2f}ms)')
    for i, t in enumerate(THRESHOLDS):
        ax1.annotate(f"T={t}", (bert_lats[i], bert_accs[i]), textcoords="offset points", xytext=(0,10), ha='center')
    ax1.set_xlabel('Latency (ms/batch)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('BERT Early Exit: Accuracy vs Latency')
    ax1.legend()
    ax1.grid(True)

    # RoBERTa Plot
    ax2.plot(roberta_lats, roberta_accs, 's-', label='Early Exit RoBERTa', color='purple')
    ax2.axhline(y=roberta_base_acc, color='red', linestyle='--', label=f'Baseline ({roberta_base_acc:.2f}%)')
    ax2.axvline(x=roberta_base_lat, color='green', linestyle=':', label=f'Baseline Latency ({roberta_base_lat:.2f}ms)')
    for i, t in enumerate(THRESHOLDS):
        ax2.annotate(f"T={t}", (roberta_lats[i], roberta_accs[i]), textcoords="offset points", xytext=(0,10), ha='center')
    ax2.set_xlabel('Latency (ms/batch)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('RoBERTa Early Exit: Accuracy vs Latency')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    main()
