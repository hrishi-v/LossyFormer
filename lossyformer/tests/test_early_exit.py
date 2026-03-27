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
# Integration: Early Exit + Pruning
# ==============================================================================


class TestEarlyExitWithPruning:
    def test_bert_early_exit_works_after_pruning(self):
        from lossyformer.pruning import prune_heads_pass

        model = _tiny_bert()
        prune_heads_pass(model, {0: [0, 1]})
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_bert_early_exit_all_logits_after_pruning(self):
        from lossyformer.pruning import prune_heads_pass

        model = _tiny_bert()
        prune_heads_pass(model, {0: [0]})
        ee = BertEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"], output_all_logits=True)
        assert len(out["logits"]) == 2

    def test_roberta_early_exit_works_after_pruning(self):
        from lossyformer.pruning import prune_heads_pass

        model = _tiny_roberta()
        prune_heads_pass(model, {0: [0, 1]})
        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_roberta_early_exit_all_logits_after_pruning(self):
        from lossyformer.pruning import prune_heads_pass

        model = _tiny_roberta()
        prune_heads_pass(model, {0: [0]})
        ee = RobertaEarlyExit(model, threshold=0.3)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"], output_all_logits=True)
        assert len(out["logits"]) == 2

    def test_get_early_exit_model_works_after_pruning_bert(self):
        from lossyformer.pruning import prune_heads_pass

        model = _tiny_bert()
        prune_heads_pass(model, {0: [0]})
        ee = get_early_exit_model(model, threshold=0.5)
        assert isinstance(ee, BertEarlyExit)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)

    def test_get_early_exit_model_works_after_pruning_roberta(self):
        from lossyformer.pruning import prune_heads_pass

        model = _tiny_roberta()
        prune_heads_pass(model, {0: [0]})
        ee = get_early_exit_model(model, threshold=0.5)
        assert isinstance(ee, RobertaEarlyExit)
        inputs = _dummy_inputs()
        out = ee(inputs["input_ids"], inputs["attention_mask"])
        assert out["logits"].shape == (4, 3)
