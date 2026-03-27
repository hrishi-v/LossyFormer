import pytest
import torch
import torch.nn as nn
from .base import EarlyExitMixin

class DummyConfig:
    def __init__(self, num_labels):
        self.num_labels = num_labels

class DummyModel:
    def __init__(self, num_labels):
        self.config = DummyConfig(num_labels)

class DummyModelNoConfig:
    config = type('DummyConfigEmpty', (), {})()

class TestEarlyExitMixin:
    def test_init_early_exit(self):
        mixin = EarlyExitMixin()
        model = DummyModel(num_labels=4)
        mixin.init_early_exit(model, threshold=0.5)
        
        assert mixin.threshold == 0.5
        assert mixin.num_labels == 4
        assert mixin.pooler is None
        assert mixin.classifier is None

    def test_init_early_exit_no_num_labels(self):
        mixin = EarlyExitMixin()
        model = DummyModelNoConfig()
        mixin.init_early_exit(model, threshold=0.2)
        
        assert mixin.threshold == 0.2
        assert mixin.num_labels is None

    def test_compute_logits_with_pooler(self):
        mixin = EarlyExitMixin()
        mixin.init_early_exit(DummyModelNoConfig(), 0.5)
        mixin.pooler = nn.Linear(10, 10)
        mixin.classifier = nn.Linear(10, 2)
        
        hidden_states = torch.randn(4, 5, 10)
        logits = mixin.compute_logits(hidden_states)
        assert logits.shape == (4, 5, 2) or logits.shape == (4, 2)

    def test_compute_logits_without_pooler_cls_token(self):
        mixin = EarlyExitMixin()
        mixin.init_early_exit(DummyModelNoConfig(), 0.5)
        mixin.classifier = nn.Linear(10, 2)
        
        # Batch size 4, sequence length 5, hidden size 10
        hidden_states = torch.randn(4, 5, 10)
        logits = mixin.compute_logits(hidden_states)
        
        # Should slice to hidden_states[:, 0], which is (4, 10), so logits should be (4, 2)
        assert logits.shape == (4, 2)

    def test_compute_logits_without_pooler_exception_fallback(self):
        mixin = EarlyExitMixin()
        mixin.init_early_exit(DummyModelNoConfig(), 0.5)
        mixin.classifier = nn.Linear(10, 2)
        
        # 1D tensor to trigger IndexError on [:, 0]
        hidden_states = torch.randn(10)
        logits = mixin.compute_logits(hidden_states)
        
        assert logits.shape == (2,)

    def test_evaluate_confidence(self):
        mixin = EarlyExitMixin()
        mixin.threshold = 0.5
        
        # 1. Very confident (low entropy, ~0.0) -> [10.0, -10.0]
        # 2. Pretty confident (entropy < 0.5) -> [2.0, -2.0]
        # 3. Not confident (high entropy, ~0.69) -> [0.0, 0.0]
        logits = torch.tensor([
            [10.0, -10.0],
            [2.0, -2.0],
            [0.0, 0.0]
        ])
        
        active_indices = torch.tensor([0, 1, 2])
        final_logits = torch.zeros(3, 2)
        
        leftover_mask = mixin.evaluate_confidence(logits, active_indices, final_logits)
        
        # Expect samples 0 and 1 to be confident (False in leftover mask)
        # Expect sample 2 to be not confident (True in leftover mask)
        assert leftover_mask.tolist() == [False, False, True]
        
        # Final logits should be updated for index 0 and 1
        assert torch.allclose(final_logits[0], logits[0])
        assert torch.allclose(final_logits[1], logits[1])
        # Index 2 should remain zeroes
        assert torch.allclose(final_logits[2], torch.zeros(2))
