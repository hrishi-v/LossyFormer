import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from early_exit.roberta import RobertaEarlyExit

class DummyConfig:
    def __init__(self, num_labels=None):
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
        first_token_tensor = hidden_states[:, 0]
        return self.dense(first_token_tensor)

class DummyRoberta(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = DummyEmbeddings()
        self.encoder = DummyEncoder()
        self.pooler = DummyPooler()

class DummyOutProj(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.proj = nn.Linear(in_features, out_features)
        
    def forward(self, hidden_states):
        return self.proj(hidden_states)

class DummyRobertaClassifier(nn.Module):
    def __init__(self, has_out_proj=True):
        super().__init__()
        if has_out_proj:
            self.out_proj = DummyOutProj(10, 3)
        else:
            self.out_features = 3
            self.classifier = nn.Linear(10, 3)
            
    def forward(self, hidden_states):
        if hasattr(self, "out_proj"):
            return self.out_proj(hidden_states)
        return self.classifier(hidden_states)

class DummyModelForRoberta(nn.Module):
    def __init__(self, has_classifier=True, has_roberta_attr=True, has_out_proj=True, has_config_num_labels=False):
        super().__init__()
        
        num_labels = 3 if has_config_num_labels else None
        self.config = DummyConfig(num_labels=num_labels)
        
        if has_roberta_attr:
            self.roberta = DummyRoberta()
        else:
            self.embeddings = DummyEmbeddings()
            self.encoder = DummyEncoder()
            self.pooler = DummyPooler()
            
        if has_classifier:
            self.classifier = DummyRobertaClassifier(has_out_proj=has_out_proj)
            
    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        return attention_mask

class TestRobertaEarlyExit:
    def test_init_success_with_out_proj(self):
        model = DummyModelForRoberta(has_out_proj=True)
        early_exit_model = RobertaEarlyExit(model, threshold=0.5)
        
        assert early_exit_model.threshold == 0.5
        assert early_exit_model.num_labels == 3
        assert early_exit_model.num_layers == 3
        
    def test_init_success_with_out_features(self):
        model = DummyModelForRoberta(has_out_proj=False)
        early_exit_model = RobertaEarlyExit(model, threshold=0.5)
        
        assert early_exit_model.num_labels == 3
        
    def test_init_success_with_config_labels(self):
        model = DummyModelForRoberta(has_config_num_labels=True)
        model.classifier.out_proj.out_features = 100
        
        early_exit_model = RobertaEarlyExit(model, threshold=0.5)
        
        assert early_exit_model.num_labels == 3 # should match config, not out_proj

    def test_init_no_roberta_attr(self):
        model = DummyModelForRoberta(has_roberta_attr=False)
        early_exit_model = RobertaEarlyExit(model, threshold=0.5)
        
        assert early_exit_model.num_layers == 3
        
    def test_init_missing_classifier(self):
        model = DummyModelForRoberta(has_classifier=False)
        with pytest.raises(AttributeError, match="Could not find classifier in the provided model."):
            RobertaEarlyExit(model, threshold=0.5)
            
    def test_forward_all_logits(self):
        model = DummyModelForRoberta()
        early_exit_model = RobertaEarlyExit(model, threshold=0.5)
        
        input_ids = torch.randint(0, 100, (4, 5))
        outputs = early_exit_model(input_ids, output_all_logits=True)
        
        assert "logits" in outputs
        assert len(outputs["logits"]) == 3
        for logits in outputs["logits"]:
            assert logits.shape == (4, 3)
            
    def test_forward_early_exit_low_threshold(self):
        model = DummyModelForRoberta()
        early_exit_model = RobertaEarlyExit(model, threshold=-1.0) 
        early_exit_model.eval()
        
        input_ids = torch.randint(0, 100, (4, 5))
        outputs = early_exit_model(input_ids, output_all_logits=False)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 3)
        
    def test_forward_early_exit_high_threshold(self):
        model = DummyModelForRoberta()
        early_exit_model = RobertaEarlyExit(model, threshold=100.0) 
        early_exit_model.eval()
        
        input_ids = torch.randint(0, 100, (4, 5))
        outputs = early_exit_model(input_ids, output_all_logits=False)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 3)