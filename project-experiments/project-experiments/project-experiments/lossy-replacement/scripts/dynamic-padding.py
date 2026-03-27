import torch
import time
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, default_data_collator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from chop import MaseGraph

def measure_throughput(model, loader, device, use_fp16=False, warmup=10, measure_steps=100):
    model.eval().to(device)
    times = []
    samples = 0
    
    loader_iter = iter(loader)
    
    # Warmup
    for _ in range(warmup):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
            
        b = {k: v.to(device) for k, v in batch.items()}
        if isinstance(model, torch.fx.GraphModule):
            b = {k: v for k, v in b.items() if k != "token_type_ids"}
            
        with torch.autocast("cuda" if device == "cuda" else "cpu", enabled=use_fp16):
            with torch.no_grad():
                model(**b)

    if device == "cuda": torch.cuda.synchronize()

    for _ in tqdm(range(measure_steps), desc="Measuring", leave=False):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
            
        b = {k: v.to(device) for k, v in batch.items()}
        if isinstance(model, torch.fx.GraphModule):
            b = {k: v for k, v in b.items() if k != "token_type_ids"}
            
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.autocast("cuda" if device == "cuda" else "cpu", enabled=use_fp16):
            with torch.no_grad():
                model(**b)
                
        if device == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        samples += b["input_ids"].size(0)

    avg_time_ms = np.mean(times) * 1000 if times else 0.0
    throughput = samples / sum(times) if sum(times) > 0 else 0.0
    return throughput, avg_time_ms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/vol/bitbucket/hv122/adls-data/utsav/bert-base-glue-mnli-baseline"
    
    print(f"Loading MASE Graph from {checkpoint_path}...")
    mg = MaseGraph.from_checkpoint(checkpoint_path)
    model = mg.model
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("glue", "mnli", split="validation_matched")

    print("\nPreparing Standard Pipeline (Max Length 128)...")
    def tokenize_static(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=128)
    
    ds_static = dataset.map(tokenize_static, batched=True).remove_columns(["premise", "hypothesis", "idx"])
    ds_static = ds_static.rename_column("label", "labels").with_format("torch")
    loader_static = DataLoader(ds_static, batch_size=32, collate_fn=default_data_collator)

    print("Preparing Optimized Pipeline (Dynamic Padding)...")
    def tokenize_dynamic(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)
    
    ds_dynamic = dataset.map(tokenize_dynamic, batched=True).remove_columns(["premise", "hypothesis", "idx"])
    ds_dynamic = ds_dynamic.rename_column("label", "labels").with_format("torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader_dynamic = DataLoader(ds_dynamic, batch_size=32, collate_fn=collator)

    print("\n" + "="*50)
    print("TEST 1: The Baseline (Static Padding, FP32)")
    tput_base, lat_base = measure_throughput(model, loader_static, device, use_fp16=False)
    print(f"Latency: {lat_base:.2f} ms/batch | Throughput: {tput_base:.2f} samples/sec")

    print("\nTEST 2: The Software Optimization (Dynamic Padding + FP16)")
    tput_opt, lat_opt = measure_throughput(model, loader_dynamic, device, use_fp16=True)
    print(f"Latency: {lat_opt:.2f} ms/batch | Throughput: {tput_opt:.2f} samples/sec")
    
    speedup = (tput_opt / tput_base) if tput_base > 0 else 0
    print("="*50)
    print(f"TOTAL SOFTWARE SPEEDUP: {speedup:.2f}x Faster!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()