import pytest
import torch
import torch.nn as nn
import sys
import os
from .bert_base import BertEarlyExit

class DummyConfig:
    def __init__(self, num_labels):
        self.num_labels = num_labels

class DummyLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return (hidden_states + 0.1,)

class DummyEncoder(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        self.layer = nn.ModuleList([DummyLayer() for _ in range(num_layers)])

class DummyEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 10)
        
    def forward(self, input_ids):
        return self.emb(input_ids)

class DummyPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(10, 10)
        
    def forward(self, hidden_states):
        # We simulate pooling by just taking the first token's representation
        first_token_tensor = hidden_states[:, 0]
        return self.dense(first_token_tensor)

class DummyBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = DummyEmbeddings()
        self.encoder = DummyEncoder()
        self.pooler = DummyPooler()

class DummyModelForBert(nn.Module):
    def __init__(self, has_classifier=True, has_bert_attr=True):
        super().__init__()
        self.config = DummyConfig(num_labels=2)
        
        if has_bert_attr:
            self.bert = DummyBert()
        else:
            # Fallback to model itself acting as base_model
            self.embeddings = DummyEmbeddings()
            self.encoder = DummyEncoder()
            self.pooler = DummyPooler()
            
        if has_classifier:
            self.classifier = nn.Linear(10, 2)
            
    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        return attention_mask

class TestBertEarlyExit:
    def test_init_success(self):
        model = DummyModelForBert()
        early_exit_model = BertEarlyExit(model, threshold=0.5)
        
        assert early_exit_model.threshold == 0.5
        assert early_exit_model.num_labels == 2
        assert early_exit_model.num_layers == 3
        
    def test_init_no_bert_attr(self):
        model = DummyModelForBert(has_bert_attr=False)
        early_exit_model = BertEarlyExit(model, threshold=0.5)
        
        assert early_exit_model.num_layers == 3
        
    def test_init_missing_classifier(self):
        model = DummyModelForBert(has_classifier=False)
        with pytest.raises(AttributeError, match="Could not find classifier in the provided model."):
            BertEarlyExit(model, threshold=0.5)
            
    def test_forward_all_logits(self):
        model = DummyModelForBert()
        early_exit_model = BertEarlyExit(model, threshold=0.5)
        
        input_ids = torch.randint(0, 100, (4, 5))
        outputs = early_exit_model(input_ids, output_all_logits=True)
        
        assert "logits" in outputs
        assert len(outputs["logits"]) == 3 # 3 layers
        for logits in outputs["logits"]:
            assert logits.shape == (4, 2)
            
    def test_forward_early_exit_low_threshold(self):
        # Force a very low threshold so no samples exit early
        model = DummyModelForBert()
        early_exit_model = BertEarlyExit(model, threshold=-1.0) 
        early_exit_model.eval()
        
        input_ids = torch.randint(0, 100, (4, 5))
        outputs = early_exit_model(input_ids, output_all_logits=False)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 2)
        
    def test_forward_early_exit_high_threshold(self):
        # Force a very high threshold so all samples exit at the first layer
        model = DummyModelForBert()
        early_exit_model = BertEarlyExit(model, threshold=100.0) 
        early_exit_model.eval()
        
        input_ids = torch.randint(0, 100, (4, 5))
        # No attention mask provided initially to test the default `torch.ones` creation
        outputs = early_exit_model(input_ids, output_all_logits=False)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 2)
