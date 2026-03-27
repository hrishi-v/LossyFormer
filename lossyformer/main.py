import copy
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from collections.abc import Mapping
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from datasets import load_dataset

from .pruning.pruning import (
    instrument_model,
    remove_instrumentation,
    decide_heads_to_prune,
    prune_heads_pass,
    calibrate_with_survival,
)
from .pruning.finetune import fine_tune_lora
from .utils import eval_accuracy, eval_speed
from .early_exit import get_early_exit_model, EARLY_EXIT_CONSTRUCTION_MAP

# Get logger (configuration handled by application)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LossyFormer:
    """Iteratively prunes transformer heads while maintaining accuracy via early exit classifiers."""

    def __init__(
        self,
        model_name,
        allowed_accuracy_loss=0.01,
        device="cuda",
        max_iterations=25,
        step_keep_ratio=0.90,
        dataset_kwargs=None,
    ):
        """Initialize LossyFormer.

        Args:
            model_name: Model identifier for early exit construction (e.g., 'bert-tiny')
            allowed_accuracy_loss: Max accuracy drop tolerated (e.g., 0.01 for 1%)
            device: Compute device ('cuda' or 'cpu')
            max_iterations: Maximum pruning rounds
            step_keep_ratio: Fraction of heads to retain per iteration (0.90 = remove 10%)
            dataset_kwargs: Dict containing dataset_name, tokenizer_name, text_columns, etc.
        """
        self.model_name = model_name
        self.allowed_loss = allowed_accuracy_loss
        self.device = device
        self.max_iterations = max_iterations
        self.step_keep_ratio = step_keep_ratio
        self.best_pruned_model = None
        self.best_ee_model = None
        self._eval_loader = None
        self._train_loader = None
        self.iteration_history = []
        self.entropy_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        # Extract dataset kwargs
        if dataset_kwargs is None:
            raise ValueError("dataset_kwargs must be provided")

        self.dataset_name = dataset_kwargs.get("dataset_name")
        self.dataset_config = dataset_kwargs.get("dataset_config", None)
        self.tokenizer_name = dataset_kwargs.get("tokenizer_name")
        self.text_columns = dataset_kwargs.get("text_columns")
        self.max_length = dataset_kwargs.get("max_length", 128)
        self.train_batch_size = dataset_kwargs.get("train_batch_size", 32)
        self.eval_batch_size = dataset_kwargs.get("eval_batch_size", 64)
        self.num_labels = dataset_kwargs.get("num_labels", 3)

    def __add_to_history(
        self,
        iteration,
        accuracy,
        throughput,
        latency,
        params,
        percent_pruned,
        threshold,
    ):
        """Track metrics for each pruning iteration."""
        self.iteration_history.append(
            {
                "iteration": iteration,
                "accuracy": accuracy,
                "throughput": throughput,
                "latency": latency,
                "params": params,
                "percent_pruned": percent_pruned,
                "threshold": threshold,
            }
        )

    def __save_iteration_history(self, model_name, allowed_loss):
        """Save iteration history to a JSON file with timestamp."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = (
            log_dir
            / f"iteration_history_{model_name}_allowedloss{int(allowed_loss * 100)}_{timestamp}.json"
        )

        safe_name = model_name.replace("/", "_")
        with open(log_file, "w") as f:
            json.dump(self.iteration_history, f, indent=2)

        logger.info(f"Iteration history saved to {log_file}")

    def _clean_state_dict(self, sd):
        """Remove LoRA-specific keys from state dict."""
        return {
            k.replace("modules_to_save.default.", "").replace("default.", ""): v.cpu()
            for k, v in sd.items()
        }

    def _evaluate_baseline(self, model, eval_loader):
        """Evaluate baseline accuracy, throughput, latency, and parameters."""
        logger.info("Evaluating Baseline...")
        baseline_accuracy = eval_accuracy(model, eval_loader, self.device)
        logger.info(f"Baseline Accuracy: {baseline_accuracy * 100:.2f}%")

        tput, lat = eval_speed(model, eval_loader, self.device)
        logger.info(f"Baseline throughput: {tput:.2f} samples/sec")
        logger.info(f"Baseline latency: {lat:.2f} sec")

        baseline_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Baseline parameters: {baseline_params}")

        return baseline_accuracy, tput, lat, baseline_params

    def _train_early_exit_classifiers(self, model, train_loader):
        """Train early exit classifiers and return saved states."""
        logger.info("Training Early Exit Classifiers for Traffic Estimation...")
        constructor = EARLY_EXIT_CONSTRUCTION_MAP.get(self.model_name, None)
        if constructor:
            ee_model = constructor(model, threshold=0.3).to(self.device)
        else:
            ee_model = get_early_exit_model(model, threshold=0.3).to(self.device)
        ee_model.freeze_backbone_unfreeze_classifier()
        ee_model.train_classifiers(train_loader, device=self.device)

        classifier_state = (
            {k: v.cpu() for k, v in ee_model.classifier.state_dict().items()}
            if ee_model.classifier
            else None
        )
        pooler_state = (
            {k: v.cpu() for k, v in ee_model.pooler.state_dict().items()}
            if ee_model.pooler
            else None
        )

        return classifier_state, pooler_state

    def _calibrate_survival_probs(
        self, model, classifier_state, pooler_state, eval_loader, max_ft_steps
    ):
        """Calibrate survival probabilities for current model state."""
        logger.info("Recalibrating Survival Probabilities...")
        current_ee_model = get_early_exit_model(model, threshold=0.3).to(self.device)
        if classifier_state:
            current_ee_model.classifier.load_state_dict(classifier_state)
        if pooler_state and current_ee_model.pooler:
            current_ee_model.pooler.load_state_dict(pooler_state)

        survival_probs = calibrate_with_survival(
            current_ee_model,
            {},
            eval_loader,
            thresholds=[0.15, 0.3, 0.45, 0.6],
            device=self.device,
            n_batches=int(max_ft_steps / 2),
        )
        del current_ee_model
        torch.cuda.empty_cache()
        return survival_probs

    def _profile_and_prune_heads(self, model, survival_probs, eval_loader):
        """Profile head importance and decide which heads to prune."""
        logger.info("Profiling Head Importance...")
        model.cpu()
        prof_model = copy.deepcopy(model)

        imp_modules, handles = instrument_model(prof_model)
        prof_model.to(self.device)

        calibrate(
            prof_model, imp_modules, eval_loader, device=self.device, n_batches=200
        )

        heads_to_prune = decide_heads_to_prune(
            imp_modules, survival_probs, keep_ratio=self.step_keep_ratio
        )
        remove_instrumentation(handles)
        del prof_model, imp_modules
        torch.cuda.empty_cache()

        return heads_to_prune

    def _finetune_and_search_threshold(
        self,
        model,
        classifier_state,
        pooler_state,
        train_loader,
        eval_loader,
        target_acc,
        max_ft_steps,
        max_entropy_threshold,
    ):
        """Fine-tune with LoRA and search for best early exit threshold."""
        logger.info("Fine-tuning with LoRA to recover accuracy...")
        temp_ee_for_ft = get_early_exit_model(model, threshold=0.3).to(self.device)
        if classifier_state:
            temp_ee_for_ft.classifier.load_state_dict(classifier_state)
        if pooler_state and temp_ee_for_ft.pooler:
            temp_ee_for_ft.pooler.load_state_dict(pooler_state)

        temp_ee_for_ft = fine_tune_lora(
            temp_ee_for_ft,
            train_loader,
            eval_loader,
            max_steps=max_ft_steps,
            lr=3e-4,
            device=self.device,
        )

        # Update classifier and pooler states
        classifier_state = (
            self._clean_state_dict(temp_ee_for_ft.classifier.state_dict())
            if temp_ee_for_ft.classifier
            else classifier_state
        )
        pooler_state = (
            self._clean_state_dict(temp_ee_for_ft.pooler.state_dict())
            if temp_ee_for_ft.pooler
            else pooler_state
        )

        current_model = temp_ee_for_ft.original_model
        del temp_ee_for_ft
        torch.cuda.empty_cache()

        # Search for best threshold
        temp_ee = get_early_exit_model(current_model, threshold=0.3).to(self.device)
        if classifier_state:
            temp_ee.classifier.load_state_dict(classifier_state)
        if pooler_state and temp_ee.pooler:
            temp_ee.pooler.load_state_dict(pooler_state)

        logger.info("Searching for best early exit threshold...")
        iter_best_tput, iter_best_acc, iter_best_lat, iter_best_th = (
            None,
            None,
            None,
            max_entropy_threshold,
        )

        for th in self.entropy_thresholds:
            temp_ee.threshold = th
            current_acc = eval_accuracy(temp_ee, eval_loader, device=self.device)
            current_tput, curr_batch_lat = eval_speed(
                temp_ee, eval_loader, device=self.device, n=100, warmup=10
            )

            if current_acc >= target_acc and (
                iter_best_tput is None or current_tput > iter_best_tput
            ):
                iter_best_tput, iter_best_acc, iter_best_lat, iter_best_th = (
                    current_tput,
                    current_acc,
                    curr_batch_lat,
                    th,
                )

        if iter_best_tput is None or iter_best_acc is None:
            temp_ee.threshold = max_entropy_threshold
            iter_best_acc = eval_accuracy(temp_ee, eval_loader, device=self.device)
            iter_best_tput, iter_best_lat = eval_speed(
                temp_ee, eval_loader, device=self.device, n=100, warmup=10
            )
            iter_best_th = max_entropy_threshold

        return (
            current_model,
            temp_ee,
            classifier_state,
            pooler_state,
            iter_best_tput,
            iter_best_acc,
            iter_best_lat,
            iter_best_th,
        )

    def _build_loaders(
        self,
        dataset_name,
        dataset_config,
        tokenizer_name,
        text_columns,
        max_length=128,
        train_batch_size=32,
        eval_batch_size=64,
    ):
        """Load and tokenize dataset, return train/eval data loaders."""
        if dataset_config is not None:
            raw = load_dataset(dataset_name, dataset_config)
        else:
            raw = load_dataset(dataset_name)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def tokenize_fn(examples):
            return tokenizer(
                *[examples[col] for col in text_columns],
                truncation=True,
                max_length=max_length,
            )

        tokenized = raw.map(tokenize_fn, batched=True)

        if dataset_name == "imdb":
            split_ds = tokenized["test"].train_test_split(test_size=0.5, seed=42)
            tokenized["validation"] = split_ds["train"]
            tokenized["test_split"] = split_ds["test"]
            eval_split_name = "validation"
        elif dataset_name == "glue" and dataset_config == "mnli":
            eval_split_name = "validation_matched"
        else:
            eval_split_name = "validation"

        def make_loader(split, batch_size, shuffle=False, smart_batch=False):
            ds = tokenized[split]
            if smart_batch:
                ds = ds.map(
                    lambda x: {"length": [len(seq) for seq in x["input_ids"]]},
                    batched=True,
                ).sort("length", reverse=True)

            keep_cols = ["input_ids", "attention_mask", "label", "token_type_ids"]
            remove_cols = [c for c in ds.column_names if c not in keep_cols]

            ds = ds.remove_columns(remove_cols).rename_column("label", "labels")
            ds.set_format("torch")

            return DataLoader(
                ds, batch_size=batch_size, collate_fn=collator, shuffle=shuffle
            )

        train_loader = make_loader("train", train_batch_size, shuffle=True)
        eval_loader = make_loader(eval_split_name, eval_batch_size, smart_batch=True)

        return train_loader, eval_loader

    def fit(
        self,
        model,
        train_loader=None,
        eval_loader=None,
        max_ft_steps=500,
    ):
        """Run iterative pruning pipeline. Returns optimized early-exit model."""
        if train_loader is None or eval_loader is None:
            train_loader, eval_loader = self._build_loaders(
                self.dataset_name,
                self.dataset_config,
                self.tokenizer_name,
                self.text_columns,
                self.max_length,
                self.train_batch_size,
                self.eval_batch_size,
            )

        self._train_loader = train_loader
        self._eval_loader = eval_loader
        MAX_ENTROPY_THRESHOLD = 1.4  # Early exit disabled above this threshold

        current_model = model.to(self.device)
        self.iteration_history = []

        # Establish baseline
        baseline_acc, baseline_tput, baseline_lat, baseline_params = (
            self._evaluate_baseline(current_model, eval_loader)
        )
        target_acc = baseline_acc - self.allowed_loss
        self.__add_to_history(
            iteration=0,
            accuracy=baseline_acc,
            throughput=baseline_tput,
            latency=baseline_lat,
            params=baseline_params,
            percent_pruned=0.0,
            threshold=MAX_ENTROPY_THRESHOLD,
        )

        # Train early exit classifiers
        classifier_state, pooler_state = self._train_early_exit_classifiers(
            current_model, train_loader
        )

        # Iterative pruning loop
        logger.info("Starting Iterative Pruning Loop...")
        self.best_pruned_model = copy.deepcopy(current_model)

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                f"Iteration {iteration}: Attempting to prune {(1 - self.step_keep_ratio) * 100:.1f}% of current heads"
            )

            survival_probs = self._calibrate_survival_probs(
                current_model, classifier_state, pooler_state, eval_loader, max_ft_steps
            )

            heads_to_prune = self._profile_and_prune_heads(
                current_model, survival_probs, eval_loader
            )

            if not heads_to_prune or all(
                len(heads) == 0 for heads in heads_to_prune.values()
            ):
                logger.info("No more heads can be safely pruned. Stopping early.")
                break

            logger.info("Pruning...")
            current_model = prune_heads_pass(current_model, heads_to_prune)

            (
                current_model,
                best_ee,
                classifier_state,
                pooler_state,
                iter_tput,
                iter_acc,
                iter_lat,
                iter_th,
            ) = self._finetune_and_search_threshold(
                current_model,
                classifier_state,
                pooler_state,
                train_loader,
                eval_loader,
                target_acc,
                max_ft_steps,
                MAX_ENTROPY_THRESHOLD,
            )

            param_count = sum(p.numel() for p in current_model.parameters())
            total_reduction = 1.0 - (param_count / baseline_params)
            logger.info(
                f"Iteration {iteration} Results: Acc: {iter_acc * 100:.2f}% | throughput: {iter_tput:.2f} | Params: {param_count:,}, percentage pruned: {total_reduction * 100:.1f}%, best threshold: {iter_th}"
            )
            self.__add_to_history(
                iteration=iteration,
                accuracy=iter_acc,
                throughput=iter_tput,
                latency=iter_lat,
                params=param_count,
                percent_pruned=total_reduction * 100,
                threshold=iter_th,
            )

            if iter_acc < target_acc:
                logger.warning("Accuracy dropped below target! Stopping.")
                del best_ee
                break
            else:
                self.best_pruned_model = copy.deepcopy(current_model)
                self.best_ee_model = copy.deepcopy(best_ee)
                self.best_ee_model.threshold = iter_th

            del best_ee
            torch.cuda.empty_cache()

        logger.info("Iterative Pruning Complete.")
        if self.best_ee_model is None:
            logger.warning("No pruning iterations succeeded.")
            fallback_ee = get_early_exit_model(
                self.best_pruned_model, threshold=MAX_ENTROPY_THRESHOLD
            ).to(self.device)
            return fallback_ee

        self.__save_iteration_history(self.model_name, self.allowed_loss)
        return self.best_ee_model


def calibrate(model, modules, loader, device="cuda", n_batches=100):
    model.eval().to(device)

    for param in model.parameters():
        param.requires_grad = True

    for m in modules.values():
        m.collecting = True
        m.imp_scores = []

    model.zero_grad()
    iter_loader = iter(loader)
    total_steps = min(len(loader), n_batches)

    for _ in tqdm(
        range(total_steps), desc="Calibrating Importance (Taylor)", leave=False
    ):
        try:
            batch = next(iter_loader)
        except StopIteration:
            break

        if isinstance(batch, Mapping):
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.get("labels")
        else:
            inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
            }
            labels = batch[2].to(device) if len(batch) > 2 else None

        outputs = model(**inputs)

        loss = (
            outputs.get("loss", None)
            if isinstance(outputs, Mapping)
            else getattr(outputs, "loss", None)
        )

        if loss is None and labels is not None:
            logits = (
                outputs.get("logits")
                if isinstance(outputs, Mapping)
                else getattr(outputs, "logits", outputs)
            )
            loss = F.cross_entropy(logits, labels)

        if loss is not None:
            loss.backward()
            model.zero_grad()
            del outputs, loss

    for m in modules.values():
        m.collecting = False
    torch.cuda.empty_cache()
