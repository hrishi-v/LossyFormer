import argparse
import torch
import csv
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset
from module import LossyFormer
from utils import eval_accuracy, eval_speed

# =========================================================
# CONFIGURATION
# =========================================================
parser = argparse.ArgumentParser(description="Run LossyFormer Trade-off Sweep")
parser.add_argument("--model", type=str, choices=["bert-base", "roberta"], required=True, help="Model to evaluate")
parser.add_argument("--max_ft_steps", type=int, default=500, help="Max fine-tuning steps per iteration")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BASE_PATH = "/rds/general/user/hv122/home/adls-data/utsav"

if args.model == "bert-base":
    checkpoint_name = "bert-base-uncased"
    baseline_checkpoint = f"{BASE_PATH}/bert-base-glue-mnli-baseline"
    output_csv = "bert_base_tradeoff_results.csv"
else:
    checkpoint_name = "roberta-base"
    baseline_checkpoint = f"{BASE_PATH}/roberta-base-glue-mnli-baseline"
    output_csv = "roberta_tradeoff_results.csv"

target_drops = [0.005, 0.01, 0.02, 0.04, 0.07, 0.10]

# =========================================================
# DATA SETUP
# =========================================================
print(f"Loading Dataset (GLUE MNLI) for {args.model}...")
raw = load_dataset("glue", "mnli")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)

tokenized_datasets = raw.map(tokenize_function, batched=True)

def make_loader(split_name, batch_size=64, smart_batch=False):
    ds = tokenized_datasets[split_name]
    if smart_batch:
        ds = ds.map(lambda x: {"length": [len(seq) for seq in x["input_ids"]]}, batched=True).sort("length", reverse=True)
    
    keep_cols = ["input_ids", "attention_mask", "token_type_ids", "label"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols).rename_column("label", "labels").with_format("torch")
    
    return DataLoader(ds, batch_size=batch_size, collate_fn=data_collator, shuffle=not smart_batch)

train_loader = make_loader("train", batch_size=32, smart_batch=False)
test_loader = make_loader("validation_matched", batch_size=64, smart_batch=True)

# =========================================================
# PHASE 1: BASELINE EVALUATION
# =========================================================
print("\n" + "="*50)
print(f"PHASE 1: BASE MODEL EVALUATION")
print("="*50)

# Load HF Model and inject custom baseline weights
hf_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name, num_labels=3)
state_dict = torch.load(f"{baseline_checkpoint}.pt", map_location=DEVICE, weights_only=False)
if hasattr(state_dict, "state_dict"): state_dict = state_dict.state_dict()
hf_model.load_state_dict(state_dict, strict=False)
hf_model.to(DEVICE)

base_acc = eval_accuracy(hf_model, test_loader, DEVICE)
base_tput, base_lat = eval_speed(hf_model, test_loader, DEVICE, n=50, warmup=5)
base_params = sum(p.numel() for p in hf_model.parameters()) / 1e6

print(f"Base Acc: {base_acc*100:.2f}% | Latency: {base_lat*1000:.2f}ms | Params: {base_params:.2f}M")

fieldnames = ["Target Drop (%)", "Accuracy (%)", "Latency (ms)", "Params (M)", "EE Threshold", "Total Reduction (%)"]
with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        "Target Drop (%)": 0.0, "Accuracy (%)": round(base_acc * 100, 2),
        "Latency (ms)": round(base_lat * 1000, 2), "Params (M)": round(base_params, 2),
        "EE Threshold": "N/A", "Total Reduction (%)": 0.0
    })

# =========================================================
# PHASE 2: LOSSYFORMER SWEEP
# =========================================================
for drop in target_drops:
    print("\n" + "="*50)
    print(f"TESTING ACCURACY DROP THRESHOLD: {drop*100:.1f}%")
    print("="*50)
    
    # Reload fresh baseline for each trial so drops don't compound
    current_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_name, num_labels=3)
    current_model.load_state_dict(state_dict, strict=False)
    
    lossy = LossyFormer(allowed_accuracy_loss=drop, device=DEVICE)
    opt_model = lossy.fit(current_model, train_loader, test_loader, max_ft_steps=args.max_ft_steps)
    
    pure_pruned_model = opt_model.base_model if hasattr(opt_model, "base_model") else opt_model

    print(f"\nEvaluating final optimized model for {drop*100:.1f}% drop...")
    acc = eval_accuracy(opt_model, test_loader, DEVICE)
    tput, lat = eval_speed(pure_pruned_model, test_loader, DEVICE, n=50, warmup=5)
    
    params = sum(p.numel() for p in pure_pruned_model.parameters()) / 1e6
    threshold = getattr(opt_model, "threshold", "N/A")
    
    final_reduction = lossy.iteration_history[-1]["percent_pruned"] if lossy.iteration_history else 0.0
    
    print(f"  Result Acc: {acc*100:.2f}% | Latency: {lat*1000:.2f}ms | Params: {params:.2f}M")
    
    with open(output_csv, mode="a", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow({
            "Target Drop (%)": drop * 100, "Accuracy (%)": round(acc * 100, 2),
            "Latency (ms)": round(lat * 1000, 2), "Params (M)": round(params, 2),
            "EE Threshold": round(threshold, 2) if isinstance(threshold, float) else threshold,
            "Total Reduction (%)": round(final_reduction, 2)
        })
        
    del lossy, opt_model, current_model
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nSweep complete! Final data is in {output_csv}")