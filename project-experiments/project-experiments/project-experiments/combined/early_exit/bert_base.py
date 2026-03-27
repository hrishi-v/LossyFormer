import torch
import torch.nn as nn
from .base import EarlyExitMixin

class BertEarlyExit(nn.Module, EarlyExitMixin):
    def __init__(self, original_model, threshold=0.3):
        super().__init__()
        self.init_early_exit(original_model, threshold)

        # 1. Directly extract native Hugging Face submodules
        self.original_model = original_model
        self.base_model = getattr(original_model, "bert", original_model)
        
        self.embeddings = self.base_model.embeddings
        self.layers = self.base_model.encoder.layer
        self.pooler = getattr(self.base_model, "pooler", None)
        self.classifier = getattr(original_model, "classifier", None)

        if self.classifier is None:
            raise AttributeError("Could not find classifier in the provided model.")

        if self.num_labels is None:
            self.num_labels = getattr(self.classifier, "out_features", 2)

        self.num_layers = len(self.layers)

    def forward(self, input_ids, attention_mask=None, output_all_logits=False, **kwargs):
        device = input_ids.device
        batch_size = input_ids.size(0)

        # 1. Setup
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

        # 2. Execution Loop
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, attention_mask=active_mask)
            hidden_states = layer_outputs[0]

            logits = self.compute_logits(hidden_states)
            final_logits = final_logits.to(dtype=logits.dtype)

            if output_all_logits:
                all_logits_list.append(logits)

            if i == self.num_layers - 1:
                final_logits[active_indices] = logits
                break

            if output_all_logits:
                continue

            not_confident = self.evaluate_confidence(logits, active_indices, final_logits)

            if not not_confident.any():
                break  # Everyone exited!

            # 3. Shrink batched tensors for the next layer
            active_indices = active_indices[not_confident]
            hidden_states = hidden_states[not_confident]
            active_mask = active_mask[not_confident]

        if output_all_logits:
            return {"logits": all_logits_list}

        return {"logits": final_logits}