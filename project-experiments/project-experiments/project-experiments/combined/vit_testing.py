import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from module import LossyFormer, eval_accuracy, eval_speed
import numpy as np
import os

# Setup Device
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# Data Setup
model_name = "nateraw/vit-base-patch16-224-cifar10"
print(f"Loading Dataset (CIFAR-10)...")

# Load CIFAR-10
raw = load_dataset("cifar10")
print("Dataset columns:", raw["train"].column_names)

processor = AutoImageProcessor.from_pretrained(model_name)


def preprocess_function(examples):
    """Preprocess images for ViT"""
    inputs = processor(examples["img"], return_tensors="pt")
    inputs["labels"] = examples["label"]
    return inputs


print("Preprocessing dataset...")
# Using batched mapping for speed
preprocessed_datasets = raw.map(
    preprocess_function,
    batched=True,
    remove_columns=["img"],
)


def make_loader(split_name, batch_size=32, shuffle=False):
    """Create DataLoader for ViT"""
    ds = preprocessed_datasets[split_name]
    # Keep only model input columns
    keep_cols = ["pixel_values", "labels"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols)
    ds.set_format("torch")

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


train_loader = make_loader("train", batch_size=32, shuffle=True)
test_loader = make_loader("test", batch_size=32)

print("Dataset loaded successfully.")

# Load baseline model from HuggingFace
print(f"Loading model...")

try:
    baseline_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=10,  # CIFAR-10 has 10 classes
        attn_implementation="sdpa",
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    print(f"✓ Successfully loaded {model_name}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    raise

print(f"Model loaded: {type(baseline_model)._name_}")

# Initialize LossyFormer
# We set allowed_accuracy_loss=0.03 (3%) tolerance
# Note: LossyFormer.fit() is currently optimized for BERT/text models
# For ViT, you may need to adapt the pruning strategy
lossy = LossyFormer(allowed_accuracy_loss=0.03, device=DEVICE)

# Run Fit (if LossyFormer supports ViT)
# Otherwise, just evaluate baseline
print(f"Starting LossyFormer fit process...")
try:
    model = lossy.fit(
        baseline_model,
        train_loader,
        test_loader,
        # pruning_trials=2,
        max_ft_steps=30,
    )
except Exception as e:
    print(f"Note: LossyFormer.fit() may not fully support ViT yet: {e}")
    print("Evaluating baseline model instead...")
    model = baseline_model

# Final Evaluation
print("\nFinal Evaluation:")
acc = eval_accuracy(model, test_loader, DEVICE)
print(f"Final Accuracy: {acc * 100:.2f}%")

throughput, latency = eval_speed(model, test_loader, DEVICE)
print(f"Final Throughput: {throughput:.2f} samples/sec")
print(f"Final Latency: {latency * 1000:.2f} ms/batch")