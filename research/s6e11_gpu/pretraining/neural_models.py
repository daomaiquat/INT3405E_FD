"""Small PyTorch encoders for self-supervised learning on tabular features."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _MLPEncoder(nn.Module):
    output_width = 64
    token_width = 8

    def __init__(self, n_numeric: int, n_categorical: int):
        super().__init__()
        input_width = 2 * n_numeric + n_categorical * self.token_width
        self.network = nn.Sequential(
            nn.Linear(input_width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, self.output_width),
            nn.LayerNorm(self.output_width),
            nn.GELU(),
            nn.Dropout(0.10),
        )

    def forward(self, numeric: Tensor, mask: Tensor, category_tokens: Tensor) -> Tensor:
        numeric = numeric.masked_fill(mask, 0)
        features = torch.cat((numeric, mask.to(numeric.dtype), category_tokens.flatten(1)), dim=1)
        return self.network(features)


class _FTEncoder(nn.Module):
    output_width = 32
    token_width = output_width

    def __init__(self, n_numeric: int):
        super().__init__()
        width = self.token_width
        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, width))
        self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, width))
        self.numeric_missing = nn.Parameter(torch.empty(n_numeric, width))
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        for parameter in (self.numeric_weight, self.numeric_missing, self.cls_token):
            nn.init.normal_(parameter, std=0.02)
        self.attention = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=64,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, numeric: Tensor, mask: Tensor, category_tokens: Tensor) -> Tensor:
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        numeric_tokens = torch.where(mask.unsqueeze(-1), self.numeric_missing, numeric_tokens)
        cls_tokens = self.cls_token.expand(numeric.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, numeric_tokens, category_tokens), dim=1)
        return self.output_norm(self.attention(tokens)[:, 0])


def _column_mask(x: Tensor, fraction: float) -> Tensor:
    width = x.shape[1]
    count = max(1, round(width * fraction))
    chosen = torch.rand_like(x).topk(count, dim=1, largest=False).indices
    return torch.zeros_like(x, dtype=torch.bool).scatter_(1, chosen, True)


class _TabularModel(nn.Module):
    def __init__(self, encoder: _MLPEncoder | _FTEncoder, n_numeric: int, category_sizes: tuple[int, ...]):
        super().__init__()
        self.encoder = encoder
        self.n_numeric = n_numeric
        self.category_sizes = category_sizes
        sizes = torch.tensor(category_sizes, dtype=torch.long)
        sizes_with_mask = sizes + 1
        self.register_buffer("category_offsets", sizes_with_mask.cumsum(0) - sizes_with_mask)
        self.register_buffer("category_mask_ids", sizes)
        self.category_embedding = nn.Embedding(max(1, sum(category_sizes) + len(category_sizes)), encoder.token_width)
        nn.init.normal_(self.category_embedding.weight, std=0.02)

        width = encoder.output_width
        self.classifier = nn.Linear(width, 1)
        self.reconstruction = nn.Linear(width, n_numeric + sum(category_sizes))
        self.scarf_projection = (
            nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
            if isinstance(encoder, _MLPEncoder)
            else None
        )

    def _encode(self, x: Tensor, mask: Tensor) -> Tensor:
        split = self.n_numeric
        category_ids = torch.where(mask[:, split:], self.category_mask_ids, x[:, split:].long())
        tokens = self.category_embedding(category_ids + self.category_offsets)
        return self.encoder(x[:, :split], mask[:, :split], tokens)

    def encode(self, x: Tensor) -> Tensor:
        """Return one embedding per row of unmasked features."""
        return self._encode(x, torch.zeros_like(x, dtype=torch.bool))

    def forward(self, x: Tensor) -> Tensor:
        """Return one binary classification logit per row."""
        return self.classifier(self.encode(x)).squeeze(-1)

    def _masked_loss(self, x: Tensor) -> Tensor:
        mask = _column_mask(x, 0.30)
        prediction = self.reconstruction(self._encode(x, mask))
        split = self.n_numeric
        numeric_mask = mask[:, :split]
        numeric_errors = F.smooth_l1_loss(prediction[:, :split], x[:, :split], reduction="none")
        numeric_loss = (numeric_errors * numeric_mask).sum() / numeric_mask.sum().clamp_min(1)

        categorical_mask = mask[:, split:]
        categorical_logits = prediction[:, split:].split(self.category_sizes, dim=1)
        categorical_error = prediction.new_zeros(())
        for index, logits in enumerate(categorical_logits):
            target = x[:, split + index].long()
            errors = F.cross_entropy(logits, target, reduction="none")
            categorical_error = categorical_error + (errors * categorical_mask[:, index]).sum()
        categorical_loss = categorical_error / categorical_mask.sum().clamp_min(1)
        return numeric_loss + categorical_loss

    def _scarf_loss(self, x: Tensor, projection: nn.Module) -> Tensor:
        # A shuffled minibatch provides samples from each column's empirical marginal.
        donors = torch.randint(x.shape[0], x.shape, device=x.device)
        replacements = x[donors, torch.arange(x.shape[1], device=x.device)]
        corrupted = torch.where(_column_mask(x, 0.60), replacements, x)
        embeddings = self.encode(torch.cat((x, corrupted), dim=0))
        anchor, positive = F.normalize(projection(embeddings).float(), dim=1).chunk(2, dim=0)
        similarities = (anchor @ positive.T).float() / 0.10
        matches = torch.arange(x.shape[0], device=x.device)
        return F.cross_entropy(similarities, matches)

    def pretrain_loss(self, x: Tensor, objective: str) -> Tensor:
        if objective == "masked":
            return self._masked_loss(x)
        if objective == "scarf" and self.scarf_projection is not None:
            return self._scarf_loss(x, self.scarf_projection)
        raise ValueError(f"Unsupported pretraining objective for this encoder: {objective!r}")


def build_model(architecture: str, n_numeric: int, category_sizes: tuple[int, ...]) -> nn.Module:
    """Build an MLP or FT-Transformer expecting floats followed by zero-based category IDs."""
    category_sizes = tuple(category_sizes)
    if n_numeric < 0 or any(size < 1 for size in category_sizes) or n_numeric + len(category_sizes) == 0:
        raise ValueError("Provide at least one usable feature and positive category sizes.")
    if architecture == "mlp":
        encoder = _MLPEncoder(n_numeric, len(category_sizes))
    elif architecture == "ft":
        encoder = _FTEncoder(n_numeric)
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")
    return _TabularModel(encoder, n_numeric, category_sizes)
