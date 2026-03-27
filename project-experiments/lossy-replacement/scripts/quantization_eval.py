import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.quantization as tq
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


def swap_gelu_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.GELU, GELUActivation)):
            parent_name, _, attr_name = name.rpartition(".")
            parent = dict(model.named_modules()).get(parent_name, model)
            setattr(parent, attr_name, nn.ReLU())
    return model


def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def save_and_measure_size(model, path):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size_mb


def get_trainer_with_split(model, tokenized_dataset, tokenizer, output_dir="quant-eval", eval_split="validation_matched"):
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
        no_cuda=True,
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


def evaluate_model(model, tokenized_dataset, tokenizer, eval_split="validation_matched"):
    trainer = get_trainer_with_split(model, tokenized_dataset, tokenizer, eval_split=eval_split)
    metrics = trainer.evaluate()
    return metrics["eval_accuracy"]


def measure_latency(model, batch_size=32, seq_len=128, num_warmup=10, num_runs=50):
    model.eval()
    dummy_input = {
        "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }

    with torch.no_grad():
        for _ in range(num_warmup):
            model(**dummy_input)

    t0 = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(**dummy_input)
    total = time.time() - t0

    latency_ms = (total / num_runs) * 1000.0
    throughput = (batch_size * num_runs) / total
    return latency_ms, throughput


def dynamic_quantize(model):
    model.cpu()
    model.eval()
    quantized = tq.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def load_retrained_relu(model_name, retrained_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    weights_path = os.path.join(retrained_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path, device="cpu")
    else:
        bin_path = os.path.join(retrained_path, "pytorch_model.bin")
        state_dict = torch.load(bin_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    model = swap_gelu_modules(model)
    return model


def run_comparison(model_name, retrained_path, tokenized_dataset, tokenizer):
    model_shortname = model_name.split("/")[-1]
    results = []

    configs = [
        ("GeLU (original)", lambda: AutoModelForSequenceClassification.from_pretrained(model_name)),
        ("ReLU (retrained)", lambda: load_retrained_relu(model_name, retrained_path)),
    ]

    for label, load_fn in configs:
        print(f"\n  === {model_shortname} — {label} ===")

        model = load_fn()
        model.cpu()
        model.eval()

        fp32_size = get_model_size_mb(model)
        fp32_disk = save_and_measure_size(model, "/tmp/_quant_tmp.pt")
        print(f"    FP32 in-memory: {fp32_size:.2f} MB | on-disk: {fp32_disk:.2f} MB")

        print(f"    Evaluating FP32...")
        fp32_acc = evaluate_model(model, tokenized_dataset, tokenizer)
        fp32_lat, fp32_thr = measure_latency(model)
        print(f"    FP32 accuracy: {fp32_acc:.4f} | latency: {fp32_lat:.2f} ms | throughput: {fp32_thr:.1f} seq/s")

        print(f"    Applying dynamic INT8 quantization...")
        q_model = dynamic_quantize(model)
        del model

        int8_size = get_model_size_mb(q_model)
        int8_disk = save_and_measure_size(q_model, "/tmp/_quant_tmp.pt")
        print(f"    INT8 in-memory: {int8_size:.2f} MB | on-disk: {int8_disk:.2f} MB")

        print(f"    Evaluating INT8...")
        int8_acc = evaluate_model(q_model, tokenized_dataset, tokenizer)
        int8_lat, int8_thr = measure_latency(q_model)
        print(f"    INT8 accuracy: {int8_acc:.4f} | latency: {int8_lat:.2f} ms | throughput: {int8_thr:.1f} seq/s")

        compression = fp32_disk / int8_disk if int8_disk > 0 else 0
        acc_drop = fp32_acc - int8_acc

        print(f"    Compression ratio: {compression:.2f}x")
        print(f"    Accuracy drop from quantization: {acc_drop:.4f}")

        results.append({
            "Model": model_shortname,
            "Variant": label,
            "FP32_Size_MB": round(fp32_disk, 2),
            "INT8_Size_MB": round(int8_disk, 2),
            "Compression": round(compression, 2),
            "FP32_Accuracy": round(fp32_acc, 4),
            "INT8_Accuracy": round(int8_acc, 4),
            "Accuracy_Drop": round(acc_drop, 4),
            "FP32_Latency_ms": round(fp32_lat, 2),
            "INT8_Latency_ms": round(int8_lat, 2),
            "FP32_Throughput": round(fp32_thr, 1),
            "INT8_Throughput": round(int8_thr, 1),
        })

        del q_model

    return results


def main(args):
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("glue", "mnli")

    def tokenize_fn(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    results = run_comparison(args.model_name, args.retrained_path, tokenized_dataset, tokenizer)

    model_shortname = args.model_name.split("/")[-1]
    csv_file = os.path.join(args.output_dir, f"{model_shortname}_quantization_results.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    fieldnames = ["Model", "Variant", "FP32_Size_MB", "INT8_Size_MB", "Compression",
                  "FP32_Accuracy", "INT8_Accuracy", "Accuracy_Drop",
                  "FP32_Latency_ms", "INT8_Latency_ms", "FP32_Throughput", "INT8_Throughput"]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HF model name (GeLU original)")
    parser.add_argument("--retrained_path", type=str, required=True, help="Path to retrained ReLU model")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
