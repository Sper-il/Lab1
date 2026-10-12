"""
SimpleCNN Architecture
======================
A small, classic CNN for FashionMNIST. Two conv blocks + two FC layers.

Architecture (default):
    Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool(2)
    Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool(2)
    Flatten -> Linear(64*7*7=3136, 128) -> ReLU -> Dropout
    Linear(128, 10)

Spatial dimension trace:
    Input  : (B, 1, 28, 28)
    After conv1+pool : (B, 32, 14, 14)
    After conv2+pool : (B, 64, 7, 7)
    After fc1     : (B, 128)
    Output        : (B, 10)
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A small, vanilla CNN.

    This is intentionally simple so that the training loop can converge
    in seconds — yet it normally beats every MLP variant on FashionMNIST
    (~91–92% after 10 epochs). It is the architecture we hand off to
    Người 2 / Người 3 unless the search picks the ResNet.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        dropout: float = 0.5,
        hidden_fc: int = 128,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # 28 -> 14 -> 7 after two 2x2 max pools
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, num_classes)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(self.relu(self.conv1(x)))  # (B, 32, 14, 14)
        # Block 2
        x = self.pool(self.relu(self.conv2(x)))  # (B, 64, 7, 7)
        # Flatten
        x = x.view(x.size(0), -1)                # (B, 3136)
        # Classifier head
        x = self.relu(self.fc1(x))               # (B, 128)
        x = self.dropout(x)
        x = self.fc2(x)                          # (B, 10)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, num_classes={self.num_classes}, "
            f"dropout={self.dropout_rate}"
        )