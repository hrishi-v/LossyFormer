import torch
import torch.nn.functional as F

class EarlyExitMixin:
    """Universal early exit tracking, confidence evaluation, and classifier routing."""

    def init_early_exit(self, original_model, threshold):
        self.threshold = threshold
        self.num_labels = getattr(original_model.config, "num_labels", None)
        self.pooler = None
        self.classifier = None

    def compute_logits(self, hidden_states):
        """Universal routing for classifiers (Handles [CLS] vs Pooler)"""
        if self.pooler is not None:
            return self.classifier(self.pooler(hidden_states))
        
        try:
            return self.classifier(hidden_states[:, 0])
        except Exception:
            return self.classifier(hidden_states)

    def evaluate_confidence(self, logits, active_indices, final_logits):
        """Checks threshold, saves confident predictions, returns the leftover mask."""
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        confident = entropy <= self.threshold

        if confident.any():
            final_logits[active_indices[confident]] = logits[confident]

        return ~confident

    def freeze_backbone_unfreeze_classifier(self):
      for param in self.base_model.parameters():
          param.requires_grad = False
      if self.pooler:
          for param in self.pooler.parameters():
              param.requires_grad = False
      if self.classifier:
          for param in self.classifier.parameters():
              param.requires_grad = True