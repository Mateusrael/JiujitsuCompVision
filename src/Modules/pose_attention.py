"""Pre-normalized self-attention over the two athletes' 34 joint slots."""

import math
from numbers import Real

import torch
from torch import nn


def resolve_dropout_rates(dropout=0.0, *, attention_dropout=None, mlp_dropout=None):
    """Resolve branch overrides independently, including explicit zero values."""
    values = {"dropout": dropout,
              "attention_dropout": dropout if attention_dropout is None else attention_dropout,
              "mlp_dropout": dropout if mlp_dropout is None else mlp_dropout}
    for name, value in values.items():
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or not 0 <= value < 1):
            raise ValueError(f"{name} must be a finite number in [0, 1)")
    return float(values["attention_dropout"]), float(values["mlp_dropout"])


def validate_attention_parameters(*, num_classes, embedding_dim, num_heads,
                                  num_layers, mlp_dim, dropout, pooling,
                                  attention_dropout=None, mlp_dropout=None):
    for name, value in (("num_classes", num_classes),
                        ("embedding_dim", embedding_dim),
                        ("num_heads", num_heads), ("num_layers", num_layers),
                        ("mlp_dim", mlp_dim)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if embedding_dim % num_heads:
        raise ValueError("embedding_dim must be divisible by num_heads")
    resolve_dropout_rates(dropout, attention_dropout=attention_dropout,
                          mlp_dropout=mlp_dropout)
    if pooling not in ("mean", "cls"):
        raise ValueError("pooling must be 'mean' or 'cls'")


class PreNormAttentionBlock(nn.Module):
    """Residual attention followed by a residual per-joint MLP."""

    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.0, *,
                 attention_dropout=None, mlp_dropout=None):
        super().__init__()
        attention_rate, mlp_rate = resolve_dropout_rates(
            dropout, attention_dropout=attention_dropout, mlp_dropout=mlp_dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=attention_rate, batch_first=True)
        self.attention_dropout = nn.Dropout(attention_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim), nn.GELU(), nn.Dropout(mlp_rate),
            nn.Linear(mlp_dim, embedding_dim), nn.Dropout(mlp_rate))

    def forward(self, tokens, key_padding_mask):
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=key_padding_mask, need_weights=False)
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.mlp(self.norm2(tokens))


class PoseAttentionClassifier(nn.Module):
    """Classify one frame, with one token per joint rather than per video frame.

    Slots 0-16 are athlete 1's COCO joints; slots 17-33 are athlete 2's.
    Each slot has its own learned positional embedding, so it identifies both
    the joint and the athlete slot. The input is the existing normalized
    `(batch, 102)` pose feature vector, reshaped without changing its order.
    """

    def __init__(self, num_classes=10, *, embedding_dim=128, num_heads=4,
                 num_layers=4, mlp_dim=512, dropout=0.0, pooling="mean",
                 attention_dropout=None, mlp_dropout=None):
        super().__init__()
        validate_attention_parameters(
            num_classes=num_classes, embedding_dim=embedding_dim,
            num_heads=num_heads, num_layers=num_layers, mlp_dim=mlp_dim,
            dropout=dropout, pooling=pooling,
            attention_dropout=attention_dropout, mlp_dropout=mlp_dropout)
        attention_rate, mlp_rate = resolve_dropout_rates(
            dropout, attention_dropout=attention_dropout, mlp_dropout=mlp_dropout)
        self.pooling = pooling
        self.input_projection = nn.Linear(3, embedding_dim)
        self.position_embedding = nn.Parameter(torch.empty(1, 34, embedding_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)
        self.blocks = nn.ModuleList([
            PreNormAttentionBlock(embedding_dim, num_heads, mlp_dim,
                                  attention_dropout=attention_rate, mlp_dropout=mlp_rate)
            for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != 102:
            raise ValueError("Pose attention expects input shape (batch, 102)")
        joints = features.reshape(features.shape[0], 34, 3)
        observed = joints[..., 2] > 0
        # Mask before projection: even nonzero coordinates at missing joints
        # cannot affect the logits. Preprocessing normally supplies three zeros.
        joints = joints.masked_fill(~observed.unsqueeze(-1), 0)
        tokens = self.input_projection(joints) + self.position_embedding
        key_padding_mask = ~observed
        if self.pooling == "cls":
            tokens = torch.cat((self.cls_token.expand(features.shape[0], -1, -1),
                                tokens), dim=1)
            cls_mask = torch.zeros((features.shape[0], 1), dtype=torch.bool,
                                   device=features.device)
            key_padding_mask = torch.cat((cls_mask, key_padding_mask), dim=1)
        else:
            # Attention cannot receive an entirely masked key sequence. Expose
            # a zero-input placeholder only for empty poses; mean pooling below
            # still discards every token in those samples.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[:, 0] &= observed.any(dim=1)
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask)
        tokens = self.final_norm(tokens)
        if self.pooling == "cls":
            pooled = tokens[:, 0]
        else:
            tokens = tokens.masked_fill(~observed.unsqueeze(-1), 0)
            denominator = observed.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = tokens.sum(dim=1) / denominator
        return self.classifier(pooled)
