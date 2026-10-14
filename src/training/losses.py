"""Loss-function factories for the training pipeline."""

from __future__ import annotations

import torch.nn as nn


def create_loss(name: str) -> nn.Module:
    """Create a supported classification loss by name."""
    normalized = name.strip().lower()
    if normalized in {"cross_entropy", "crossentropyloss", "ce"}:
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: {name}")


__all__ = ["create_loss"]
