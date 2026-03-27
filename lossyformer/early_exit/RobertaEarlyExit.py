import torch
import torch.nn as nn
from .EarlyExitBase import EarlyExitBase


class RobertaEarlyExit(nn.Module, EarlyExitBase):
    """Early exit wrapper for RoBERTa with entropy-based layer-wise exit."""

    def __init__(self, original_model, threshold=0.3):
        """Initialize RobertaEarlyExit.

        Args:
            original_model: RoBERTa model for sequence classification
            threshold: Entropy threshold for confidence (lower = higher bar)
        """
        super().__init__()

        original_model = original_model
        base_model = (
            original_model.roberta
            if hasattr(original_model, "roberta")
            else original_model
        )

        self.embeddings = base_model.embeddings
        self.layers = base_model.encoder.layer
        pooler = getattr(base_model, "pooler", None)
        classifier = getattr(original_model, "classifier", None)

        # Initialize EarlyExitBase explicitly due to multiple inheritance
        EarlyExitBase.__init__(
            self, original_model, base_model, threshold, classifier, pooler
        )

        if self.num_labels is None:
            # RoBERTa classifiers usually use out_proj instead of a direct weight matrix
            if hasattr(self.classifier, "out_proj"):
                self.num_labels = self.classifier.out_proj.out_features
            else:
                self.num_labels = getattr(self.classifier, "out_features", 3)

        self.num_layers = len(self.layers)

    def forward(
        self, input_ids, attention_mask=None, output_all_logits=False, **kwargs
    ):
        """Forward pass with early exit at each layer based on entropy threshold."""
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Embed input
        hidden_states = self.embeddings(input_ids=input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=device)

        # Use native HF method for attention masks
        extended_attention_mask = self.original_model.get_extended_attention_mask(
            attention_mask, input_ids.size(), device
        )

        final_logits = torch.zeros(
            batch_size, self.num_labels, device=device, dtype=hidden_states.dtype
        )
        active_indices = torch.arange(batch_size, device=device)
        active_mask = extended_attention_mask

        all_logits_list = []

        # Process through layers, removing confident samples from active batch
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, attention_mask=active_mask)
            hidden_states = layer_outputs[0]

            # The mixin handles routing the hidden states to the classifier safely
            logits = self.compute_logits(hidden_states)
            final_logits = final_logits.to(dtype=logits.dtype)

            if output_all_logits:
                all_logits_list.append(logits)

            if i == self.num_layers - 1:
                final_logits[active_indices] = logits
                break

            if output_all_logits:
                continue

            not_confident = self.evaluate_confidence(
                logits, active_indices, final_logits
            )

            if not not_confident.any():
                break  # All samples exited

            # Keep only uncertain samples for next layer
            active_indices = active_indices[not_confident]
            hidden_states = hidden_states[not_confident]
            active_mask = active_mask[not_confident]

        if output_all_logits:
            return {"logits": all_logits_list}

        return {"logits": final_logits}
