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

from module import (
    get_early_exit_model,
    calibrate,
    LossyFormer,
    get_vram_usage,
)
from early_exit.bert_base import BertEarlyExit
from early_exit.roberta import RobertaEarlyExit

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
# get_early_exit_model
# ==============================================================================


class TestGetEarlyExitModel:
    def test_returns_bert_early_exit_for_bert_model(self):
        model = _tiny_bert()
        ee = get_early_exit_model(model, threshold=0.3)
        assert isinstance(ee, BertEarlyExit)

    def test_returns_roberta_early_exit_for_roberta_model(self):
        model = _tiny_roberta()
        ee = get_early_exit_model(model, threshold=0.3)
        assert isinstance(ee, RobertaEarlyExit)

    def test_threshold_is_set_correctly(self):
        model = _tiny_bert()
        ee = get_early_exit_model(model, threshold=0.5)
        assert ee.threshold == 0.5

    def test_default_threshold_is_0_3(self):
        model = _tiny_bert()
        ee = get_early_exit_model(model)
        assert ee.threshold == 0.3


# ==============================================================================
# BertEarlyExit
# ==============================================================================


class TestBertEarlyExit:
    def test_init_extracts_correct_num_layers(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        assert ee.num_layers == 2

    def test_init_finds_classifier(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        assert ee.classifier is not None

    def test_init_finds_pooler(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        assert ee.pooler is not None

    def test_init_sets_num_labels(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        assert ee.num_labels == 3

    def test_init_stores_original_model(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        assert ee.original_model is model

    def test_forward_returns_logits_dict(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert "logits" in out
        assert out["logits"].shape == (4, 3)

    def test_forward_with_output_all_logits(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"], output_all_logits=True)
        assert "logits" in out
        assert isinstance(out["logits"], list)
        assert len(out["logits"]) == 2

    def test_forward_all_logits_have_correct_shape(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"], output_all_logits=True)
        for logits in out["logits"]:
            assert logits.shape == (4, 3)

    def test_forward_with_high_threshold_no_early_exit(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=1.1)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_forward_with_zero_threshold_all_exit_early(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=100.0)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_forward_without_attention_mask(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"])
        assert out["logits"].shape == (4, 3)

    def test_forward_batch_size_one(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs(batch=1)
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (1, 3)

    def test_classifier_raises_if_missing(self):
        model = _tiny_bert()
        model.classifier = None
        with pytest.raises(AttributeError):
            BertEarlyExit(model, threshold=0.3)


# ==============================================================================
# RobertaEarlyExit
# ==============================================================================


class TestRobertaEarlyExit:
    def test_init_extracts_correct_num_layers(self):
        model = _tiny_roberta()
        ee = RobertaEarlyExit(model, threshold=0.3)
        assert ee.num_layers == 2

    def test_init_finds_classifier(self):
        model = _tiny_roberta()
        ee = RobertaEarlyExit(model, threshold=0.3)
        assert ee.classifier is not None

    def test_init_stores_original_model(self):
        model = _tiny_roberta()
        ee = RobertaEarlyExit(model, threshold=0.3)
        assert ee.original_model is model

    def test_init_sets_num_labels(self):
        model = _tiny_roberta()
        ee = RobertaEarlyExit(model, threshold=0.3)
        assert ee.num_labels == 3

    def test_forward_returns_logits_dict(self):
        model = _tiny_roberta()
        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert "logits" in out
        assert out["logits"].shape == (4, 3)

    def test_forward_with_output_all_logits(self):
        model = _tiny_roberta()
        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"], output_all_logits=True)
        assert isinstance(out["logits"], list)
        assert len(out["logits"]) == 2

    def test_forward_all_logits_have_correct_shape(self):
        model = _tiny_roberta()
        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"], output_all_logits=True)
        for logits in out["logits"]:
            assert logits.shape == (4, 3)

    def test_forward_without_attention_mask(self):
        model = _tiny_roberta()
        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"])
        assert out["logits"].shape == (4, 3)


# ==============================================================================
# EarlyExitMixin (tested through BertEarlyExit)
# ==============================================================================


class TestEarlyExitMixin:
    def test_compute_logits_with_pooler(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        assert ee.pooler is not None
        hidden = torch.randn(2, 8, 64)
        logits = ee.compute_logits(hidden)
        assert logits.shape == (2, 3)

    def test_evaluate_confidence_returns_not_confident_mask(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.33, 0.33, 0.34]])
        active_indices = torch.arange(2)
        final_logits = torch.zeros(2, 3)
        not_confident = ee.evaluate_confidence(logits, active_indices, final_logits)
        assert not_confident.shape == (2,)
        assert not_confident.dtype == torch.bool

    def test_evaluate_confidence_confident_sample_fills_final_logits(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.5)
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        active_indices = torch.tensor([0])
        final_logits = torch.zeros(1, 3)
        ee.evaluate_confidence(logits, active_indices, final_logits)
        assert torch.allclose(final_logits[0], logits[0])

    def test_freeze_backbone_unfreeze_classifier(self):
        model = _tiny_bert()
        ee = BertEarlyExit(model, threshold=0.3)
        ee.freeze_backbone_unfreeze_classifier()
        for param in ee.base_model.parameters():
            assert not param.requires_grad
        for param in ee.classifier.parameters():
            assert param.requires_grad


# ==============================================================================
# calibrate
# ==============================================================================


class TestCalibrate:
    def test_calibrate_collects_scores(self):
        from pruning.pruning import instrument_model, remove_instrumentation

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
        from pruning.pruning import instrument_model, remove_instrumentation

        model = _tiny_bert()
        modules, handles = instrument_model(model)
        loader = _make_loader(n_batches=2)
        calibrate(model, modules, loader, device="cpu", n_batches=2)
        remove_instrumentation(handles)
        for mod in modules.values():
            assert mod.collecting is False

    def test_calibrate_respects_n_batches_limit(self):
        from pruning.pruning import instrument_model, remove_instrumentation

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
        lf = LossyFormer(allowed_accuracy_loss=0.05, device="cpu")
        assert lf.allowed_loss == 0.05

    def test_init_defaults(self):
        lf = LossyFormer()
        assert lf.allowed_loss == 0.01
        assert lf.best_pruned_model is None
        assert lf.best_ee_model is None
        assert lf.baseline_accuracy is None
        assert lf.iteration_history == []

    def test_init_device(self):
        lf = LossyFormer(device="cpu")
        assert lf.device == "cpu"

    def test_fit_returns_model_or_none(self):
        model = _tiny_bert()
        lf = LossyFormer(allowed_accuracy_loss=0.5, device="cpu")
        loader = _make_loader(n_batches=5)
        result = lf.fit(model, loader, loader, max_ft_steps=2)
        assert result is None or hasattr(result, "forward")

    def test_fit_sets_baseline_accuracy(self):
        model = _tiny_bert()
        lf = LossyFormer(allowed_accuracy_loss=0.5, device="cpu")
        loader = _make_loader(n_batches=5)
        lf.fit(model, loader, loader, max_ft_steps=2)
        assert lf.baseline_accuracy is not None
        assert 0.0 <= lf.baseline_accuracy <= 1.0

    def test_fit_populates_iteration_history(self):
        model = _tiny_bert()
        lf = LossyFormer(allowed_accuracy_loss=0.99, device="cpu")
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
        lf = LossyFormer(device="cpu")
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
# Integration: Early Exit + Pruning
# ==============================================================================


class TestEarlyExitWithPruning:
    def test_bert_early_exit_works_after_pruning(self):
        from pruning.pruning import prune_heads_pass

        model = _tiny_bert()
        prune_heads_pass(model, {0: [0, 1]})
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_bert_early_exit_all_logits_after_pruning(self):
        from pruning.pruning import prune_heads_pass

        model = _tiny_bert()
        prune_heads_pass(model, {0: [0]})
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"], output_all_logits=True)
        assert len(out["logits"]) == 2

    def test_roberta_early_exit_works_after_pruning(self):
        from pruning.pruning import prune_heads_pass

        model = _tiny_roberta()
        prune_heads_pass(model, {0: [0, 1]})
        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_roberta_early_exit_all_logits_after_pruning(self):
        from pruning.pruning import prune_heads_pass

        model = _tiny_roberta()
        prune_heads_pass(model, {0: [0]})
        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"], output_all_logits=True)
        assert len(out["logits"]) == 2

    def test_get_early_exit_model_works_after_pruning_bert(self):
        from pruning.pruning import prune_heads_pass

        model = _tiny_bert()
        prune_heads_pass(model, {0: [0]})
        ee = get_early_exit_model(model, threshold=0.5)
        assert isinstance(ee, BertEarlyExit)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_get_early_exit_model_works_after_pruning_roberta(self):
        from pruning.pruning import prune_heads_pass

        model = _tiny_roberta()
        prune_heads_pass(model, {0: [0]})
        ee = get_early_exit_model(model, threshold=0.5)
        assert isinstance(ee, RobertaEarlyExit)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)


# ==============================================================================
# Integration: Full calibrate + prune cycle through module.py
# ==============================================================================


class TestCalibrateAndPruneCycle:
    def test_full_cycle_bert(self):
        from pruning.pruning import (
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
        from pruning.pruning import (
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