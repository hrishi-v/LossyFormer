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
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from chop import MaseGraph
import chop.passes as passes
from chop.passes.module import report_trainable_parameters_analysis_pass
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


def _to_masegraph(model):
    device = next(model.parameters()).device
    model.config.use_cache = False

    dummy_input = {
        "input_ids": torch.ones(1, 128, dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, 128, dtype=torch.long, device=device),
        "labels": torch.zeros(1, dtype=torch.long, device=device),
    }

    mg = MaseGraph(model, hf_input_names=["input_ids", "attention_mask", "labels"])
    mg, _ = passes.init_metadata_analysis_pass(mg)

    with open(os.devnull, "w") as devnull:
        try:
            sys.stdout = devnull
            mg, _ = passes.add_common_metadata_analysis_pass(
                mg,
                pass_args={"dummy_in": dummy_input, "add_value": False},
            )
        finally:
            sys.stdout = sys.__stdout__

    return mg


def main(args):
    print(f"Loading model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)
    model.config.problem_type = "single_label_classification"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading and tokenizing MNLI dataset...")
    dataset = load_dataset("glue", "mnli")

    def tokenize_fn(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    model_shortname = args.model_name.split("/")[-1]
    trainer_output_dir = f"{model_shortname}-trainer"

    def eval_fn(model):
        trainer = get_trainer_with_split(
            model=model.to(device),
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            output_dir=trainer_output_dir,
        )
        metrics = trainer.evaluate()
        return metrics["eval_accuracy"]

    if args.finetune_first:
        print("\nFinetuning on MNLI first...")
        if hasattr(model, "roberta"):
            print("  Freezing RoBERTa embeddings...")
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
        trainer = get_trainer_with_split(
            model=model.to(device),
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            output_dir=trainer_output_dir,
        )
        trainer.train()
        model = trainer.model

    print("\nEvaluating pre-swap model...")
    pre_swap_accuracy = eval_fn(model)
    pre_vram, pre_lat, pre_thr = measure_performance(model.to(device), device)
    print(f"\n{args.model_name} accuracy: {pre_swap_accuracy:.4f}")

    print("\nSwapping GELU for ReLU...")
    model.cpu()
    mg = _to_masegraph(model)
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

    print(f"Exporting MaseGraph to {args.save_path}...")
    mg.export(args.save_path)

    print("Reloading swapped model and evaluating...")
    mg_loaded = MaseGraph.from_checkpoint(args.save_path)
    post_swap_accuracy = eval_fn(mg_loaded.model)
    post_vram, post_lat, post_thr = measure_performance(mg_loaded.model.to(device), device)
    print(f"ReLU-swapped (pre-retrain) accuracy: {post_swap_accuracy:.4f}")

    print("\nStarting retraining pipeline...")
    print("\nTrainable parameters before freezing:")
    _, _ = report_trainable_parameters_analysis_pass(mg_loaded.model)

    if hasattr(mg_loaded.model, "roberta"):
        print("  Freezing RoBERTa embeddings...")
        for param in mg_loaded.model.roberta.embeddings.parameters():
            param.requires_grad = False
    elif hasattr(mg_loaded.model, "model") and hasattr(mg_loaded.model.model, "roberta"):
        print("  Freezing RoBERTa embeddings (nested)...")
        for param in mg_loaded.model.model.roberta.embeddings.parameters():
            param.requires_grad = False

    print("\nTrainable parameters after freezing:")
    _, _ = report_trainable_parameters_analysis_pass(mg_loaded.model)

    trainer = get_trainer_with_split(
        model=mg_loaded.model.to(device),
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        output_dir=trainer_output_dir,
        eval_split="validation_matched",
    )

    trainer.train()
    mg_loaded.model.config.hidden_act = "relu"
    trainer.save_model(os.path.join(args.save_path, f"{model_shortname}_final_retrained"))

    print("\nEvaluating final retrained model...")
    final_metrics = trainer.evaluate()
    retrained_accuracy = final_metrics["eval_accuracy"]

    mg_loaded.model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    retrained_vram, retrained_lat, retrained_thr = measure_performance(mg_loaded.model.to(device), device)
    print(f"ReLU-swapped retrained accuracy: {retrained_accuracy:.4f}")

    csv_filename = f"{model_shortname}_performance_metrics.csv"
    csv_file = os.path.join(args.save_path, csv_filename)
    os.makedirs(args.save_path, exist_ok=True)

    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "Accuracy", "VRAM_MB", "Latency_ms", "Throughput_seq_per_sec"])
        writer.writerow(["Pre-Swap", pre_swap_accuracy, pre_vram, pre_lat, pre_thr])
        writer.writerow(["Post-Swap", post_swap_accuracy, post_vram, post_lat, post_thr])
        writer.writerow(["Retrained", retrained_accuracy, retrained_vram, retrained_lat, retrained_thr])

    print(f"All experiment metrics successfully saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HF model name (e.g. FacebookAI/roberta-base)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save swapped/retrained models")
    parser.add_argument("--finetune_first", action="store_true", help="Finetune on MNLI before swapping (for base models not already finetuned)")
    args = parser.parse_args()
    main(args)