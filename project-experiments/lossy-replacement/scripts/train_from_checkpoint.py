import os
import sys
import csv
import time
import argparse

import torch
import torch.nn as nn
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_logger


def get_trainer_with_split(model, tokenized_dataset, tokenizer, output_dir="mase-trainer", eval_split="validation_matched"):
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


def swap_gelu_for_relu(mg):
    logger = get_logger("mase_logger")
    logger.setLevel("INFO")
    count = 0
    for node in mg.fx_graph.nodes:
        if node.op == "call_module":
            module = mg.modules[node.target]
            if "gelu" in module.__class__.__name__.lower():
                logger.info(f"Replacing module GeLU: {node.target}")
                parent_name, _, attr_name = node.target.rpartition(".")
                parent_module = mg.modules[parent_name] if parent_name else mg.model
                setattr(parent_module, attr_name, nn.ReLU())
                count += 1
        elif node.op == "call_function" and "gelu" in str(node.target).lower():
            logger.info(f"Replacing functional gelu: {node.name}")
            with mg.fx_graph.inserting_after(node):
                new_node = mg.fx_graph.call_function(torch.nn.functional.relu, args=(node.args[0],))
                node.replace_all_uses_with(new_node)
                mg.fx_graph.erase_node(node)
            count += 1

    mg.fx_graph.lint()
    logger.info(f"Replaced {count} GeLU activation(s) with ReLU")
    return mg, {}


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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer from {args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print("Loading and tokenizing MNLI dataset...")
    dataset = load_dataset("glue", "mnli")

    def tokenize_fn(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    def eval_fn(model):
        trainer = get_trainer_with_split(model.to(device), tokenized_dataset, tokenizer)
        metrics = trainer.evaluate()
        return metrics["eval_accuracy"]

    # --- Stage 1: Load and evaluate original checkpoint ---
    print(f"\nLoading MASE checkpoint from {args.checkpoint}...")
    mg = MaseGraph.from_checkpoint(args.checkpoint)

    print("\n[Pre-Swap] Evaluating original model...")
    pre_acc = eval_fn(mg.model)
    pre_vram, pre_lat, pre_thr = measure_performance(mg.model.to(device), device)
    print(f"  Accuracy:  {pre_acc:.4f}")
    print(f"  VRAM (MB): {pre_vram:.2f}")
    print(f"  Latency:   {pre_lat:.2f} ms")
    print(f"  Throughput: {pre_thr:.1f} seq/s")

    del mg
    torch.cuda.empty_cache()

    # --- Stage 2: Swap GeLU -> ReLU and evaluate ---
    print("\n[Post-Swap] Swapping GeLU -> ReLU...")
    mg = MaseGraph.from_checkpoint(args.checkpoint)
    mg.model.cpu()
    mg, _ = swap_gelu_for_relu(mg)

    dummy_input = {
        "input_ids": torch.ones(1, 128, dtype=torch.long),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
        "labels": torch.zeros(1, dtype=torch.long),
    }
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg, pass_args={"dummy_in": dummy_input, "add_value": False},
    )

    if args.save_path:
        print(f"  Exporting swapped model to {args.save_path}...")
        mg.export(args.save_path)

    post_acc = eval_fn(mg.model)
    post_vram, post_lat, post_thr = measure_performance(mg.model.to(device), device)
    print(f"  Accuracy:  {post_acc:.4f}")
    print(f"  VRAM (MB): {post_vram:.2f}")
    print(f"  Latency:   {post_lat:.2f} ms")
    print(f"  Throughput: {post_thr:.1f} seq/s")

    # --- Stage 3: Retrain and evaluate ---
    print("\n[Retrained] Retraining for 1 epoch...")
    trainer = get_trainer_with_split(
        model=mg.model.to(device),
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    if args.save_path:
        retrained_dir = os.path.join(args.save_path, "retrained")
        trainer.save_model(retrained_dir)
        print(f"  Saved retrained model to {retrained_dir}")

    final_metrics = trainer.evaluate()
    retrained_acc = final_metrics["eval_accuracy"]

    mg.model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    retrained_vram, retrained_lat, retrained_thr = measure_performance(mg.model.to(device), device)
    print(f"  Accuracy:  {retrained_acc:.4f}")
    print(f"  VRAM (MB): {retrained_vram:.2f}")
    print(f"  Latency:   {retrained_lat:.2f} ms")
    print(f"  Throughput: {retrained_thr:.1f} seq/s")

    # --- Save results ---
    results = [
        {"Stage": "Pre-Swap", "Accuracy": pre_acc, "VRAM_MB": pre_vram, "Latency_ms": pre_lat, "Throughput": pre_thr},
        {"Stage": "Post-Swap", "Accuracy": post_acc, "VRAM_MB": post_vram, "Latency_ms": post_lat, "Throughput": post_thr},
        {"Stage": "Retrained", "Accuracy": retrained_acc, "VRAM_MB": retrained_vram, "Latency_ms": retrained_lat, "Throughput": retrained_thr},
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_name = os.path.basename(args.checkpoint)
    csv_file = os.path.join(args.output_dir, f"{checkpoint_name}_results.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Stage", "Accuracy", "VRAM_MB", "Latency_ms", "Throughput"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MASE checkpoint (without .mz/.pt)")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="HF tokenizer name")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save swapped/retrained models")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)