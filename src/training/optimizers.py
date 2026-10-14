"""Optimizer factories for the training pipeline."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def create_optimizer(
    name: str,
    parameters: Iterable[nn.Parameter],
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Create Adam or SGD with validated hyperparameters."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    normalized = name.strip().lower()
    if normalized == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if normalized == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


__all__ = ["create_optimizer"]
