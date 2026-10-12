"""
DeepMLP Architecture
====================
A deeper fully-connected network with Batch Normalization.

This file is the production implementation corresponding to the design
specified in `docs/PHASE2_PLAN.md` (Người 1).

Architecture (default):
    Linear(784 -> 512) -> BatchNorm1d -> ReLU -> Dropout
    Linear(512 -> 256) -> BatchNorm1d -> ReLU -> Dropout
    Linear(256 -> 128) -> BatchNorm1d -> ReLU -> Dropout
    Linear(128 -> 10)

Input shape:  (batch, 1, 28, 28)  (auto-flattened to (batch, 784))
Output shape: (batch, 10)         (raw logits; use CrossEntropyLoss)
"""

import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    """Deeper MLP with 3 hidden layers and Batch Normalization.

    Why a deeper MLP at all when CNNs clearly beat MLPs on images?
    ------------------------------------------------------
    * It's a **baseline** that often surprises people: a simple MLP can
      reach ~88% accuracy on FashionMNIST after just a few epochs.
    * It is also a great pedagogical stepping-stone before ResNet — the
      idea of stacking ``Linear -> BN -> Dropout`` blocks generalises to
      the conv blocks used later.
    * BatchNorm + Dropout together keep the network stable even though we
      blow up the parameter count (~560k weights).

    Args:
        input_size:   Flattened input size (default 28*28 = 784).
        hidden_sizes: List of hidden layer widths.
        num_classes:  Number of output classes (10 for FashionMNIST).
        dropout:      Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes=(80, 256, 128),
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()

        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one entry")

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_size = h

        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)

        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, 1, 28, 28) or already-flat (B, 784).
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_sizes={self.hidden_sizes}, "
            f"num_classes={self.num_classes}, dropout={self.dropout}"
        )