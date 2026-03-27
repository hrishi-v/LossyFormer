import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock, patch
from collections.abc import Mapping
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
    default_data_collator,
)
from torch.utils.data import DataLoader, TensorDataset
from lossyformer.main import LossyFormer, calibrate
from lossyformer.utils import get_vram_usage
from lossyformer.early_exit import BertEarlyExit, RobertaEarlyExit, get_early_exit_model

BERT_CFG = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=128,
    vocab_size=100,
    max_position_embeddings=32,
    num_labels=3,
)

ROBERTA_CFG = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=128,
    vocab_size=100,
    max_position_embeddings=32,
    num_labels=3,
)

IMDB_DATASET_KWARGS = {
    "dataset_name": "imdb",
    "dataset_config": None,
    "tokenizer_name": "prajjwal1/bert-tiny",
    "text_columns": ["text"],
    "max_length": 128,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "num_labels": 2,
}


def _tiny_bert():
    cfg = BertConfig(**BERT_CFG)
    return BertForSequenceClassification(cfg)


def _tiny_roberta():
    cfg = RobertaConfig(**ROBERTA_CFG)
    return RobertaForSequenceClassification(cfg)


def _dummy_inputs(batch=4, seq_len=8, vocab_size=100):
    ids = torch.randint(0, vocab_size, (batch, seq_len))
    mask = torch.ones_like(ids)
    labels = torch.randint(0, 3, (batch,))
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _make_loader(n_batches=3, batch_size=4, seq_len=8, vocab_size=100):
    all_inputs = []
    for _ in range(n_batches):
        all_inputs.append(_dummy_inputs(batch_size, seq_len, vocab_size))
    return all_inputs


# ==============================================================================
# calibrate
# ==============================================================================


class TestCalibrate:
    def test_calibrate_collects_scores(self):
        from lossyformer.pruning import instrument_model, remove_instrumentation

        model = _tiny_bert()
        modules, handles = instrument_model(model)
        loader = _make_loader(n_batches=3)
        calibrate(model, modules, loader, device="cpu", n_batches=3)
        remove_instrumentation(handles)
        for mod in modules.values():
            scores = mod.get_scores()
            assert scores is not None
            assert len(scores) > 0

    def test_calibrate_stops_collecting_after_completion(self):
        from lossyformer.pruning import instrument_model, remove_instrumentation

        model = _tiny_bert()
        modules, handles = instrument_model(model)
        loader = _make_loader(n_batches=2)
        calibrate(model, modules, loader, device="cpu", n_batches=2)
        remove_instrumentation(handles)
        for mod in modules.values():
            assert mod.collecting is False

    def test_calibrate_respects_n_batches_limit(self):
        from lossyformer.pruning import instrument_model, remove_instrumentation

        model = _tiny_bert()
        modules, handles = instrument_model(model)
        loader = _make_loader(n_batches=10)
        calibrate(model, modules, loader, device="cpu", n_batches=2)
        remove_instrumentation(handles)
        for mod in modules.values():
            assert len(mod.imp_scores) <= 2


# ==============================================================================
# LossyFormer
# ==============================================================================


class TestLossyFormer:
    def test_init_sets_allowed_loss(self):
        lf = LossyFormer(
            model_name="bert-tiny",
            allowed_accuracy_loss=0.05,
            device="cpu",
            dataset_kwargs=IMDB_DATASET_KWARGS,
        )
        assert lf.allowed_loss == 0.05

    def test_init_defaults(self):
        lf = LossyFormer(model_name="bert-tiny", dataset_kwargs=IMDB_DATASET_KWARGS)
        assert lf.allowed_loss == 0.01
        assert lf.best_pruned_model is None
        assert lf.best_ee_model is None
        assert lf.iteration_history == []

    def test_init_device(self):
        lf = LossyFormer(
            model_name="bert-tiny",
            device="cpu",
            dataset_kwargs=IMDB_DATASET_KWARGS,
        )
        assert lf.device == "cpu"

    def test_fit_returns_model_or_none(self):
        model = _tiny_bert()
        lf = LossyFormer(
            model_name="bert-tiny",
            allowed_accuracy_loss=0.5,
            device="cpu",
            dataset_kwargs=IMDB_DATASET_KWARGS,
        )
        loader = _make_loader(n_batches=5)
        result = lf.fit(model, loader, loader, max_ft_steps=2)
        assert result is None or hasattr(result, "forward")

    def test_fit_populates_iteration_history(self):
        model = _tiny_bert()
        lf = LossyFormer(
            model_name="bert-tiny",
            allowed_accuracy_loss=0.99,
            device="cpu",
            dataset_kwargs=IMDB_DATASET_KWARGS,
        )
        loader = _make_loader(n_batches=5)
        lf.fit(model, loader, loader, max_ft_steps=2)
        if lf.iteration_history:
            entry = lf.iteration_history[0]
            assert "iteration" in entry
            assert "accuracy" in entry
            assert "throughput" in entry
            assert "params" in entry
            assert "percent_pruned" in entry
            assert "threshold" in entry

    def test_fit_with_string_path_raises_on_invalid_path(self):
        lf = LossyFormer(
            model_name="bert-tiny",
            device="cpu",
            dataset_kwargs=IMDB_DATASET_KWARGS,
        )
        loader = _make_loader(n_batches=2)
        with pytest.raises(Exception):
            lf.fit("nonexistent-model-path", loader, loader, max_ft_steps=2)


# ==============================================================================
# get_vram_usage
# ==============================================================================


class TestGetVramUsage:
    def test_returns_tuple_of_two_numbers(self):
        current, peak = get_vram_usage()
        assert isinstance(current, (int, float))
        assert isinstance(peak, (int, float))

    def test_values_are_non_negative(self):
        current, peak = get_vram_usage()
        assert current >= 0
        assert peak >= 0


# ==============================================================================
# Integration: Full calibrate + prune cycle through module.py
# ==============================================================================


class TestCalibrateAndPruneCycle:
    def test_full_cycle_bert(self):
        from lossyformer.pruning import (
            instrument_model,
            remove_instrumentation,
            decide_heads_to_prune,
            prune_heads_pass,
        )

        model = _tiny_bert()
        before_params = sum(p.numel() for p in model.parameters())

        modules, handles = instrument_model(model)
        loader = _make_loader(n_batches=3)
        calibrate(model, modules, loader, device="cpu", n_batches=3)

        survival_probs = [1.0] * len(modules)
        heads_to_prune = decide_heads_to_prune(modules, survival_probs, keep_ratio=0.5)
        remove_instrumentation(handles)

        prune_heads_pass(model, heads_to_prune)
        after_params = sum(p.numel() for p in model.parameters())

        assert after_params < before_params

        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_full_cycle_roberta(self):
        from lossyformer.pruning import (
            instrument_model,
            remove_instrumentation,
            decide_heads_to_prune,
            prune_heads_pass,
        )

        model = _tiny_roberta()
        before_params = sum(p.numel() for p in model.parameters())

        modules, handles = instrument_model(model)
        loader = _make_loader(n_batches=3)
        calibrate(model, modules, loader, device="cpu", n_batches=3)

        survival_probs = [1.0] * len(modules)
        heads_to_prune = decide_heads_to_prune(modules, survival_probs, keep_ratio=0.5)
        remove_instrumentation(handles)

        prune_heads_pass(model, heads_to_prune)
        after_params = sum(p.numel() for p in model.parameters())

        assert after_params < before_params

        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)
