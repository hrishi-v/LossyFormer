import torch
import torch.nn.functional as F
from collections.abc import Mapping
from tqdm.auto import tqdm


class EarlyExitBase:
    """Universal early exit tracking, confidence evaluation, and classifier routing."""

    def __init__(self, original_model, base_model, threshold, classifier, pooler=None):
        """Initialize early exit infrastructure.

        Args:
            original_model: Full model for state management
            base_model: Base transformer encoder
            threshold: Entropy threshold for confidence decision
            classifier: Classification head
            pooler: Optional pooling layer
        """
        super().__init__()
        self.threshold = threshold
        self.base_model = base_model
        self.original_model = original_model  # For safe-keeping and weight loading
        self.num_labels = getattr(original_model.config, "num_labels", None)
        self.pooler = pooler
        if classifier is None:
            raise AttributeError("Could not find classifier in the provided model.")
        self.classifier = classifier

    def compute_logits(self, hidden_states):
        """Route hidden states to classifier, handling both pooler and direct access."""
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
        """Freeze backbone, unfreeze only classifier for training."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        if self.pooler:
            for param in self.pooler.parameters():
                param.requires_grad = False
        if self.classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    def train_classifiers(self, train_loader, device="cuda"):
        """Train exit classifiers at each layer using multi-task loss."""
        self.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.original_model.parameters() if p.requires_grad], lr=1e-3
        )
        init_epochs = 100

        pbar = tqdm(
            train_loader,
            total=min(len(train_loader), init_epochs),
            desc="Training Classifiers",
            leave=False,
        )
        for i, batch in enumerate(pbar):
            if i >= init_epochs:
                break
            optimizer.zero_grad()

            if isinstance(batch, Mapping):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.forward(**batch, output_all_logits=True)
                labels = batch["labels"]
            else:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                out = self.forward(inputs, None, output_all_logits=True)

            loss = sum(F.cross_entropy(logits, labels) for logits in out["logits"])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
