import math
import time
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from tqdm.auto import tqdm
from chop import MaseGraph
from collections.abc import Mapping
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer, BertPooler
from peft import LoraConfig, get_peft_model, TaskType
import peft


@torch.no_grad()
def eval_accuracy(model, loader, device="cuda"):
    model.eval().to(device)
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Evaluating Accuracy", leave=False):
        if isinstance(batch, Mapping):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = (
                outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
            )
            labels = batch["labels"]
            
            if isinstance(model, torch.fx.GraphModule):
                forward_kwargs = {k: v for k, v in batch.items() if k != "token_type_ids"}
                outputs = model(**forward_kwargs)
            else:
                outputs = model(**batch)

            logits = (
                outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
            )
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)

        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0
    acc = correct / total
    return acc


@torch.no_grad()
def eval_speed(model, loader, device="cuda", n=50, warmup=5):
    model.eval().to(device)
    compiled = model

    batches = list(loader)
    if not batches:
        return 0.0, 0.0

    if len(batches) > n + warmup:
        batches = batches[: warmup + n]

    # Warmup
    for b in batches[:warmup]:
        if isinstance(b, Mapping):
            b = {k: v.to(device) for k, v in b.items()}
            if isinstance(compiled, torch.fx.GraphModule):
                forward_kwargs = {k: v for k, v in b.items() if k != "token_type_ids"}
                compiled(**forward_kwargs)
            else:
                compiled(**b)
        else:
            inputs, _ = b
            compiled(inputs.to(device))

    if device == "cuda":
        torch.cuda.synchronize()

    samples, times = 0, []
    desc_pbar = tqdm(batches[warmup:], desc="Evaluating Speed", leave=False)
    for b in desc_pbar:
        if isinstance(b, Mapping):
            b = {k: v.to(device) for k, v in b.items()}
        else:
            inputs, _ = b
            b = inputs.to(device)  # simplify for timing

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if isinstance(b, Mapping):
            if isinstance(compiled, torch.fx.GraphModule):
                forward_kwargs = {k: v for k, v in b.items() if k != "token_type_ids"}
                compiled(**forward_kwargs)
            else:
                compiled(**b)
        else:
            compiled(b)

        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        if isinstance(b, Mapping) and "input_ids" in b:
            samples += b["input_ids"].size(0)
        else:
            try:
                samples += b.size(0)
            except:
                samples += 1

    avg_time = np.mean(times) if times else 0.0
    throughput = samples / sum(times) if sum(times) > 0 else 0.0
    return throughput, avg_time

