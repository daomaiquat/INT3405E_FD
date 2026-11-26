"""Compact TabNet with a shared encoder for masked pretraining and classification."""

import torch
from pytorch_tabnet.tab_network import TabNetPretraining
from torch import nn


class MaskedTabNet(nn.Module):
    """Wrap the TabNet attention, group masking and reconstruction blocks."""

    def __init__(self, n_numeric: int, category_sizes: tuple[int, ...]):
        super().__init__()
        input_dim = n_numeric + len(category_sizes)
        decision_width = 12
        self.tabnet = TabNetPretraining(
            input_dim=input_dim,
            pretraining_ratio=0.20,
            n_d=decision_width,
            n_a=decision_width,
            n_steps=3,
            n_independent=1,
            n_shared=1,
            cat_idxs=list(range(n_numeric, input_dim)),
            cat_dims=list(category_sizes),
            cat_emb_dim=[min(4, size) for size in category_sizes],
            virtual_batch_size=4096,
            group_attention_matrix=torch.eye(input_dim),
        )
        self.classifier = nn.Linear(decision_width, 1)
        # The library stores these matrices as plain attributes. Keep device-aware
        # copies so model.to("cuda") and saved checkpoints have the expected behavior.
        self.register_buffer(
            "attention_groups", self.tabnet.encoder.group_attention_matrix, persistent=False
        )
        self.register_buffer(
            "mask_groups", self.tabnet.masker.group_matrix, persistent=False
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.tabnet.encoder.group_attention_matrix = self.attention_groups
        # Sparsemax and the attention loss inside TabNet use small constants.
        with torch.autocast(device_type=x.device.type, enabled=False):
            embedded = self.tabnet.embedder(x.float())
            decisions, _ = self.tabnet.encoder(embedded)
            return torch.stack(decisions, dim=0).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x)).squeeze(-1)

    def pretrain_loss(self, x: torch.Tensor, objective: str) -> torch.Tensor:
        if objective != "masked":
            raise ValueError(f"TabNet supports masked pretraining, got {objective!r}")

        self.tabnet.encoder.group_attention_matrix = self.attention_groups
        self.tabnet.masker.group_matrix = self.mask_groups
        with torch.autocast(device_type=x.device.type, enabled=False):
            embedded = self.tabnet.embedder(x.float())
            obscured, masked_groups, masked_features = self.tabnet.masker(embedded)
            decisions, _ = self.tabnet.encoder(obscured, prior=1.0 - masked_groups)
            reconstructed = self.tabnet.decoder(decisions)

            # Reconstruct masked numerical values and categorical embeddings only.
            # Cap the influence of almost constant embeddings of rare categories.
            target = embedded.detach()
            variance = target.var(dim=0, unbiased=False)
            scale = variance.clamp_min(0.1)
            errors = (reconstructed - target).square() * masked_features / scale
            masked_count = masked_features.sum(dim=1).clamp_min(1.0)
            return (errors.sum(dim=1) / masked_count).mean()


def build_tabnet(n_numeric: int, category_sizes: tuple[int, ...]) -> MaskedTabNet:
    return MaskedTabNet(n_numeric, category_sizes)
