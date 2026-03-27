import torch
import csv
import gc
from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset
from module import LossyFormer, eval_accuracy, MaseGraph
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

checkpoint_name = "bert-base-uncased"
# FIXED: Pointing directly to your absolute utsav directory
baseline_checkpoint = "/vol/bitbucket/hv122/adls-data/bert-base-glue-mnli-baseline"
output_csv = "ablation_study_results.csv"

print("Loading Dataset (GLUE MNLI)...")
raw = load_dataset("glue", "mnli")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

# FIXED: Added split_name and shuffle logic to ensure we don't train on the test set
def get_loader(mode, split_name="validation_matched", is_train=False):
    pad_strat = "max_length" if mode == "static" else False
    
    def tokenize_fn(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], 
                         truncation=True, max_length=128, padding=pad_strat)
    
    ds = raw.map(tokenize_fn, batched=True)
    
    if mode == "smart" and not is_train:
        ds = ds.map(lambda x: {"length": [len(seq) for seq in x["input_ids"]]}, batched=True)
        ds[split_name] = ds[split_name].sort("length", reverse=True)
        
    collator = DataCollatorWithPadding(tokenizer=tokenizer) if mode in ["dynamic", "smart"] else default_data_collator
    
    target_ds = ds[split_name]
    keep_cols = ["input_ids", "attention_mask", "token_type_ids", "label"]
    target_ds = target_ds.remove_columns([c for c in target_ds.column_names if c not in keep_cols]).rename_column("label", "labels").with_format("torch")
    
    do_shuffle = is_train and mode != "smart"
    batch_size = 32 if is_train else 64
    return DataLoader(target_ds, batch_size=batch_size, collate_fn=collator, shuffle=do_shuffle)

# Get our three distinct test loaders
loaders = {
    "Static": get_loader("static"),
    "Dynamic": get_loader("dynamic"),
    "Smart": get_loader("smart")
}

results = []

def profile_model(model, model_name):
    print(f"\n--- Profiling {model_name} ---")
    model.eval()
    
    for mode_name, loader in loaders.items():
        print(f"  Testing with {mode_name} Batching...")
        acc = eval_accuracy(model, loader, DEVICE)
        
        # =======================================================
        # PURE GPU COMPUTE PROFILER (Bypasses CPU DataLoader Time)
        # =======================================================
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        latencies = []
        iterator = iter(loader)
        
        # WARMUP
        with torch.no_grad():
            for _ in range(5):
                batch = next(iterator)
                batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "token_type_ids"}
                _ = model(**batch)
        torch.cuda.synchronize()
        
        # PROFILING
        total_samples = 0
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            for _ in range(50):
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                
                # Move to GPU *before* starting the timer
                batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "token_type_ids"}
                total_samples += batch["input_ids"].size(0)
                
                # Start recording exact GPU active time
                start_event.record()
                _ = model(**batch)
                end_event.record()
                
                # Wait for GPU math to finish
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
        
        # Calculate pure compute metrics
        avg_gpu_lat_ms = np.mean(latencies)
        gpu_throughput = total_samples / (sum(latencies) / 1000)
        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        print(f"    -> GPU Throughput: {gpu_throughput:.2f} seq/s")
        print(f"    -> Peak VRAM: {vram_mb:.2f} MB")
        
        results.append({
            "Architecture": model_name,
            "Data Pipeline": mode_name,
            "Accuracy (%)": round(acc * 100, 2),
            "GPU Throughput (seq/s)": round(gpu_throughput, 2),
            "GPU Latency (ms/batch)": round(avg_gpu_lat_ms, 2),
            "Peak VRAM (MB)": round(vram_mb, 2)
        })
        torch.cuda.empty_cache()

# =========================================================
# PHASE 1: Profile Base Model
# =========================================================
print("\n" + "="*50)
print("PHASE 1: BASE MODEL ABLATION")
print("="*50)
base_mg = MaseGraph.from_checkpoint(baseline_checkpoint)
profile_model(base_mg.model, "Base BERT")

# =========================================================
# PHASE 2: Train LossyFormer (Using Dynamic for training speed)
# =========================================================
print("\n" + "="*50)
print("PHASE 2: COMPRESSING MODEL WITH LOSSYFORMER")
print("="*50)
train_loader = get_loader("dynamic", split_name="train", is_train=True)
lossy = LossyFormer(allowed_accuracy_loss=0.03, device=DEVICE)
opt_model = lossy.fit(baseline_checkpoint, train_loader, loaders["Dynamic"], pruning_trials=2)

del base_mg
gc.collect()
torch.cuda.empty_cache()

# =========================================================
# PHASE 3: Profile Optimized Model
# =========================================================
print("\n" + "="*50)
print("PHASE 3: LOSSYFORMER ABLATION")
print("="*50)

# FIXED: Attempt to fuse LoRA weights to solve the VRAM spike
try:
    if hasattr(opt_model, "merge_and_unload"):
        print("Merging LoRA weights to reduce VRAM footprint...")
        opt_model = opt_model.merge_and_unload()
    elif hasattr(opt_model, "base_model") and hasattr(opt_model.base_model, "merge_and_unload"):
        print("Merging LoRA weights to reduce VRAM footprint...")
        opt_model.base_model = opt_model.base_model.merge_and_unload()
except Exception as e:
    print(f"Skipping LoRA merge: {e}")

profile_model(opt_model, "LossyFormer")

# =========================================================
# SAVE RESULTS
# =========================================================
print("\nSaving Ablation Study to CSV...")
with open(output_csv, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Success! Ablation data written to {output_csv}")