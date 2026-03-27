import argparse
import csv
import os
import sys
import time

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
import chop.passes as passes


def swap_gelu_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.GELU, GELUActivation)):
            parent_name, _, attr_name = name.rpartition(".")
            parent = dict(model.named_modules()).get(parent_name, model)
            setattr(parent, attr_name, nn.ReLU())
    return model


def to_masegraph(model):
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


def get_trainer_with_split(model, tokenized_dataset, tokenizer, output_dir="mase-quant-eval", eval_split="validation_matched"):
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


def evaluate_model(model, tokenized_dataset, tokenizer, device):
    trainer = get_trainer_with_split(model.to(device), tokenized_dataset, tokenizer)
    metrics = trainer.evaluate()
    return metrics["eval_accuracy"]


def apply_mase_quantization(model, width, frac_width):
    model.cpu()
    mg = to_masegraph(model)

    quantization_config = {
        "by": "type",
        "default": {
            "config": {
                "name": None,
            }
        },
        "linear": {
            "config": {
                "name": "integer",
                "data_in_width": width,
                "data_in_frac_width": frac_width,
                "weight_width": width,
                "weight_frac_width": frac_width,
                "bias_width": width,
                "bias_frac_width": frac_width,
            }
        },
    }

    mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)
    return mg.model


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


def run_single_model(model_name, retrained_path, tokenized_dataset, tokenizer, device, bit_widths):
    model_shortname = model_name.split("/")[-1]
    results = []

    configs = [
        ("GeLU (original)", lambda: AutoModelForSequenceClassification.from_pretrained(model_name)),
        ("ReLU (retrained)", lambda: load_retrained_relu(model_name, retrained_path)),
    ]

    for label, load_fn in configs:
        print(f"\n  === {model_shortname} — {label} ===")

        model = load_fn()
        model.eval()

        print(f"    Evaluating FP32...")
        fp32_acc = evaluate_model(model, tokenized_dataset, tokenizer, device)
        print(f"    FP32 accuracy: {fp32_acc:.4f}")
        del model
        torch.cuda.empty_cache()

        for width, frac_width in bit_widths:
            print(f"    Quantizing to {width}-bit (frac={frac_width})...")
            model_fresh = load_fn()
            model_fresh.eval()
            q_model = apply_mase_quantization(model_fresh, width, frac_width)
            del model_fresh
            torch.cuda.empty_cache()

            print(f"    Evaluating {width}-bit...")
            q_acc = evaluate_model(q_model, tokenized_dataset, tokenizer, device)
            acc_drop = fp32_acc - q_acc
            print(f"    {width}-bit accuracy: {q_acc:.4f} | drop: {acc_drop:.4f}")

            results.append({
                "Model": model_shortname,
                "Variant": label,
                "Bit_Width": width,
                "Frac_Width": frac_width,
                "FP32_Accuracy": round(fp32_acc, 4),
                "Quantized_Accuracy": round(q_acc, 4),
                "Accuracy_Drop": round(acc_drop, 4),
            })

            del q_model
            torch.cuda.empty_cache()

    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    bit_widths = [(16, 8), (8, 4)]

    print(f"Loading tokenizer and dataset for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset("glue", "mnli")

    def tokenize_fn(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    model_shortname = args.model_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Processing {model_shortname}")
    print(f"{'='*60}")

    results = run_single_model(
        args.model_name, args.retrained_path,
        tokenized_dataset, tokenizer, device, bit_widths,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = os.path.join(args.output_dir, f"{model_shortname}_mase_quant_results.csv")

    fieldnames = ["Model", "Variant", "Bit_Width", "Frac_Width", "FP32_Accuracy", "Quantized_Accuracy", "Accuracy_Drop"]

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"Results saved to {csv_file}")
    print(f"{'='*60}")

    print(f"\n{'Model':<30} {'Variant':<20} {'Bits':>5} {'FP32':>8} {'Quant':>8} {'Drop':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['Model']:<30} {r['Variant']:<20} {r['Bit_Width']:>5} {r['FP32_Accuracy']:>8.4f} {r['Quantized_Accuracy']:>8.4f} {r['Accuracy_Drop']:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--retrained_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)