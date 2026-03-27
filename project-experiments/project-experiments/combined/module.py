import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections.abc import Mapping
from transformers import AutoModelForSequenceClassification

from pruning.finetune import fine_tune_lora
from utils import eval_accuracy, eval_speed
from early_exit.bert_base import BertEarlyExit
from early_exit.roberta import RobertaEarlyExit
from pruning.pruning import (
    instrument_model, 
    remove_instrumentation,
    decide_heads_to_prune, 
    prune_heads_pass, 
    calibrate_with_survival
)

logger = logging.getLogger(__name__)

def get_early_exit_model(model, threshold=0.3):
    if hasattr(model, "roberta") or "roberta" in str(type(model)).lower():
        return RobertaEarlyExit(model, threshold=threshold)
    return BertEarlyExit(model, threshold=threshold)

def get_vram_usage():
    if torch.cuda.is_available():
        current_vram = torch.cuda.memory_allocated() / 1024**2
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        return current_vram, peak_vram
    return 0, 0

# ==============================================================================
# LossyFormer
# ==============================================================================

def calibrate(model, modules, loader, device="cuda", n_batches=100):
    model.eval().to(device)
    
    # Enable gradients for Taylor Pruning
    for param in model.parameters():
        param.requires_grad = True

    for m in modules.values():
        m.collecting = True
        m.imp_scores = []

    model.zero_grad()
    iter_loader = iter(loader)
    total_steps = min(len(loader), n_batches)

    for _ in tqdm(range(total_steps), desc="Calibrating Importance (Taylor)", leave=False):
        try:
            batch = next(iter_loader)
        except StopIteration:
            break

        if isinstance(batch, Mapping):
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.get("labels")
        else:
            inputs = {"input_ids": batch[0].to(device), "attention_mask": batch[1].to(device)}
            labels = batch[2].to(device) if len(batch) > 2 else None

        outputs = model(**inputs)
        
        loss = outputs.get("loss", None) if isinstance(outputs, Mapping) else getattr(outputs, "loss", None)
        
        if loss is None and labels is not None:
            logits = outputs.get("logits") if isinstance(outputs, Mapping) else getattr(outputs, "logits", outputs)
            loss = F.cross_entropy(logits, labels)

        if loss is not None:
            loss.backward()
            model.zero_grad()
            del outputs, loss

    for m in modules.values():
        m.collecting = False
    torch.cuda.empty_cache()


class LossyFormer:
    def __init__(self, allowed_accuracy_loss=0.01, device="cuda"):
        self.allowed_loss = allowed_accuracy_loss
        self.device = device
        self.best_pruned_model = None
        self.best_ee_model = None
        self.baseline_accuracy = None
        self.iteration_history = []

    def fit(self, model_or_path, train_loader, eval_loader, max_ft_steps=500):
        print(f"Loading baseline...", flush=True)
        if isinstance(model_or_path, str):
            current_model = AutoModelForSequenceClassification.from_pretrained(model_or_path)
        else:
            current_model = model_or_path
            
        current_model.to(self.device)

        print("Evaluating Baseline...")
        self.baseline_accuracy = eval_accuracy(current_model, eval_loader, self.device)
        print(f"Baseline Accuracy: {self.baseline_accuracy * 100:.2f}%")
        
        target_acc = self.baseline_accuracy - self.allowed_loss
        tput, _ = eval_speed(current_model, eval_loader, self.device)
        print(f"Baseline throughput: {tput:.2f} samples/sec")

        print("Training Early Exit Classifiers for Traffic Estimation...")
        ee_model = get_early_exit_model(current_model, threshold=0.3).to(self.device)

        # Freeze Backbone, Train Classifiers
        # for param in ee_model.base_model.parameters(): param.requires_grad = False
        # if ee_model.pooler:
        #     for param in ee_model.pooler.parameters(): param.requires_grad = False
        # if ee_model.classifier:
        #     for param in ee_model.classifier.parameters(): param.requires_grad = True

        ee_model.freeze_backbone_unfreeze_classifier()

        ee_model.train()
        optimizer = torch.optim.AdamW([p for p in ee_model.parameters() if p.requires_grad], lr=1e-3)
        init_epochs = 100

        pbar = tqdm(train_loader, total=min(len(train_loader), init_epochs), desc="Training Classifiers", leave=False)
        for i, batch in enumerate(pbar):
            if i >= init_epochs: break
            optimizer.zero_grad()
            
            if isinstance(batch, Mapping):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = ee_model(**batch, output_all_logits=True)
                labels = batch["labels"]
            else:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                out = ee_model(inputs, None, output_all_logits=True)

            loss = sum(F.cross_entropy(logits, labels) for logits in out["logits"])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # Save classifier states to re-inject after pruning/finetuning
        classifier_state = {k: v.cpu() for k, v in ee_model.classifier.state_dict().items()} if ee_model.classifier else None
        pooler_state = {k: v.cpu() for k, v in ee_model.pooler.state_dict().items()} if ee_model.pooler else None

        # --- Step 2: Iterative Pruning Loop ---
        print("Starting Iterative Pruning Loop...", flush=True)
        self.best_pruned_model = copy.deepcopy(current_model)

        iteration = 0
        max_iterations = 25
        step_keep_ratio = 0.90

        while iteration < max_iterations:
            iteration += 1
            total_reduction = 1.0 - (1.0 * (0.90**iteration))
            print(f"\n--- Iteration {iteration}: Pruning 10% of current heads (Total Reduction ~{total_reduction * 100:.1f}%) ---")

            # A. Recalibrate Traffic
            current_ee_model = get_early_exit_model(current_model, threshold=0.3).to(self.device)
            if classifier_state: current_ee_model.classifier.load_state_dict(classifier_state)
            if pooler_state and current_ee_model.pooler: current_ee_model.pooler.load_state_dict(pooler_state)

            print("  Recalibrating Survival Probabilities...")
            survival_probs = calibrate_with_survival(
                current_ee_model, {}, eval_loader, thresholds=[0.15, 0.3, 0.45, 0.6], device=self.device, n_batches=int(max_ft_steps / 2)
            )
            del current_ee_model
            torch.cuda.empty_cache()

            # B. Profile Importance
            print("  Profiling Head Importance...")
            current_model.cpu()
            prof_model = copy.deepcopy(current_model)
            
            # Setup Native Hooks
            imp_modules, handles = instrument_model(prof_model)
            prof_model.to(self.device)

            calibrate(prof_model, imp_modules, eval_loader, device=self.device, n_batches=200)

            # Decide what to drop
            heads_to_prune = decide_heads_to_prune(imp_modules, survival_probs, keep_ratio=step_keep_ratio)
            # --- DEBUG SNIPPET ---
            print(f"\n[DEBUG] Iteration {iteration} | heads_to_prune: {heads_to_prune}")
            if heads_to_prune:
                total_pruned = sum(len(heads) for heads in heads_to_prune.values())
                print(f"[DEBUG] Total heads targeted for pruning this step: {total_pruned}")
            # ---------------------
            remove_instrumentation(handles)
            del prof_model, imp_modules
            torch.cuda.empty_cache()

            # --- EARLY STOPPING: Parameter Floor Check ---
            if not heads_to_prune or all(len(heads) == 0 for heads in heads_to_prune.values()):
                print("  No more heads can be safely pruned (reached parameter floor of 1 head per layer). Stopping early.")
                break

            # Prune Native Hugging Face Model
            print(f"  Pruning...")
            current_model = prune_heads_pass(current_model, heads_to_prune)

            # C. Fine-tune (Recovery)
            print("  Fine-tuning with LoRA to recover accuracy...")
            temp_ee_for_ft = get_early_exit_model(current_model, threshold=0.3).to(self.device)
            if classifier_state: temp_ee_for_ft.classifier.load_state_dict(classifier_state)
            if pooler_state and temp_ee_for_ft.pooler: temp_ee_for_ft.pooler.load_state_dict(pooler_state)

            temp_ee_for_ft = fine_tune_lora(
                temp_ee_for_ft, train_loader, eval_loader, max_steps=max_ft_steps, lr=3e-4, device=self.device
            )

            # Extract states and base model back out
            def clean_state_dict(sd):
                return {k.replace("modules_to_save.default.", "").replace("default.", ""): v.cpu() for k, v in sd.items()}

            if temp_ee_for_ft.classifier: classifier_state = clean_state_dict(temp_ee_for_ft.classifier.state_dict())
            if temp_ee_for_ft.pooler: pooler_state = clean_state_dict(temp_ee_for_ft.pooler.state_dict())
            
            current_model = temp_ee_for_ft.original_model
            del temp_ee_for_ft
            torch.cuda.empty_cache()

            # D. Evaluate Thresholds
            temp_ee = get_early_exit_model(current_model, threshold=0.3).to(self.device)
            if classifier_state: temp_ee.classifier.load_state_dict(classifier_state)
            if pooler_state and temp_ee.pooler: temp_ee.pooler.load_state_dict(pooler_state)

            iter_best_tput, iter_best_acc, iter_best_th = 0.0, 0.0, 1.1
            
            for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            # for th in [0.5]:
                temp_ee.threshold = th
                current_acc = eval_accuracy(temp_ee, eval_loader, device=self.device)
                current_tput, _ = eval_speed(temp_ee, eval_loader, device=self.device, n=100, warmup=10)
                
                if current_acc >= target_acc and current_tput > iter_best_tput:
                    iter_best_tput, iter_best_acc, iter_best_th = current_tput, current_acc, th

            if iter_best_tput == 0.0:
                temp_ee.threshold = 1.1
                iter_best_acc = eval_accuracy(temp_ee, eval_loader, device=self.device)
                iter_best_tput, _ = eval_speed(temp_ee, eval_loader, device=self.device, n=100, warmup=10)
                iter_best_th = 1.1

            param_count = sum(p.numel() for p in current_model.parameters())
            print(f"  Iteration {iteration} Results: Acc: {iter_best_acc * 100:.2f}% | throughput: {iter_best_tput:.2f} | Params: {param_count:,}, percentage pruned: {total_reduction * 100:.1f}%, best threshold: {iter_best_th}")

            self.iteration_history.append({
                "iteration": iteration, "accuracy": iter_best_acc, "throughput": iter_best_tput,
                "params": param_count, "percent_pruned": total_reduction * 100, "threshold": iter_best_th
            })

            # --- EARLY STOPPING: Accuracy Drop Check ---
            if iter_best_acc < target_acc:
                print("  Accuracy dropped below target! Stopping.")
                del temp_ee
                break
            else:
                self.best_pruned_model = copy.deepcopy(current_model)
                self.best_ee_model = copy.deepcopy(temp_ee)
                self.best_ee_model.threshold = iter_best_th

            del temp_ee
            torch.cuda.empty_cache()

        print("Iterative Pruning Complete.")
        if self.best_ee_model is None and hasattr(self, 'iteration_history') and len(self.iteration_history) == 0:
            print("Warning: No pruning iterations succeeded.")
        
        print(self.iteration_history)
        return self.best_ee_model