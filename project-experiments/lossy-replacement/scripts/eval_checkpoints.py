import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.activations import GELUActivation
from safetensors.torch import load_file

from chop import MaseGraph


def swap_gelu_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.GELU, GELUActivation)):
            parent_name, _, attr_name = name.rpartition(".")
            parent = dict(model.named_modules()).get(parent_name, model)
            setattr(parent, attr_name, nn.ReLU())
    return model


def get_trainer_with_split(model, tokenized_dataset, tokenizer, output_dir="eval-trainer", eval_split="validation_matched"):
    metric = evaluate.load("accuracy")

    def compute_accuracy(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir,
        report_to="none",
        num_train_epochs=1,
        save_strategy="no",
        warmup_steps=500,
        learning_rate=1e-5,
        disable_tqdm=True,
    )

    return Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[eval_split],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
    )


def measure_performance(model, device, batch_size=32, seq_len=128):
    model.eval()
    dummy_input = {
        "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=device),
        "labels": torch.zeros(batch_size, dtype=torch.long, device=device),
    }
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(10):
            model(**dummy_input)

    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        t0 = time.time()

    with torch.no_grad():
        for _ in range(50):
            model(**dummy_input)

    if device == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        total_time_sec = start_event.elapsed_time(end_event) / 1000.0
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        total_time_sec = time.time() - t0
        vram_mb = 0.0

    latency_ms = (total_time_sec / 50.0) * 1000.0
    throughput = (batch_size * 50) / total_time_sec
    return vram_mb, latency_ms, throughput


def make_eval_fn(dataset, tokenizer, device, eval_split="validation_matched"):
    def eval_fn(model):
        trainer = get_trainer_with_split(
            model=model.to(device),
            tokenized_dataset=dataset,
            tokenizer=tokenizer,
            eval_split=eval_split,
        )
        metrics = trainer.evaluate()
        return metrics["eval_accuracy"]
    return eval_fn


def load_model(model_or_path, mode="hf", base_model_name=None):
    if mode == "checkpoint":
        mg = MaseGraph.from_checkpoint(model_or_path)
        model = mg.model.cpu()
        del mg
        return model
    elif mode == "retrained":
        model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
        weights_path = os.path.join(model_or_path, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path, device="cpu")
        else:
            bin_path = os.path.join(model_or_path, "pytorch_model.bin")
            state_dict = torch.load(bin_path, map_location="cpu")
        result = model.load_state_dict(state_dict, strict=False)
        del state_dict
        print(f"    Loaded weights — missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}")
        if result.unexpected_keys:
            print(f"    Unexpected keys: {result.unexpected_keys}")
        model = swap_gelu_modules(model)
        return model
    else:
        return AutoModelForSequenceClassification.from_pretrained(model_or_path)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer and dataset for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("glue", "mnli")

    def tokenize_fn(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    eval_fn = make_eval_fn(tokenized_dataset, tokenizer, device)

    model_shortname = args.model_name.split("/")[-1]

    stages = [
        ("Pre-Swap", args.model_name, "hf"),
        ("Post-Swap", args.swap_checkpoint, "checkpoint"),
        ("Retrained", args.retrained_path, "retrained"),
    ]

    results = []
    for stage, path, mode in stages:
        if path is None:
            print(f"\n  [{stage}] No path provided, skipping.")
            continue

        if mode != "hf" and not (Path(path).exists() or Path(str(path) + ".mz").exists()):
            print(f"\n  [{stage}] Path not found: {path}, skipping.")
            continue

        print(f"\n  [{stage}] Loading from {path}...")

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        model = load_model(path, mode=mode, base_model_name=args.model_name)
        model.to(device)
        model.eval()

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        vram, latency, throughput = measure_performance(model, device)
        print(f"    VRAM (MB):    {vram:.2f}")
        print(f"    Latency (ms): {latency:.2f}")
        print(f"    Throughput:   {throughput:.1f} seq/s")

        accuracy = eval_fn(model)
        print(f"    Accuracy:     {accuracy:.4f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        results.append({
            "Stage": stage,
            "Accuracy": round(accuracy, 4),
            "VRAM_MB": round(vram, 2),
            "Latency_ms": round(latency, 2),
            "Throughput_seq_per_sec": round(throughput, 1),
        })

    csv_file = os.path.join(args.output_dir, f"{model_shortname}_clean_metrics.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Stage", "Accuracy", "VRAM_MB", "Latency_ms", "Throughput_seq_per_sec"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--swap_checkpoint", type=str, required=True)
    parser.add_argument("--retrained_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    main(args)