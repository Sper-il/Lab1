"""
Baseline MLP Model
==================
The simplest possible neural network for FashionMNIST, exactly as the
design document in ``docs/PHASE2_PLAN.md`` specifies.

Architecture:
    Flatten -> Linear(784 -> hidden_size) -> ReLU -> Dropout -> Linear(hidden_size -> 10)

Input shape:  (batch, 1, 28, 28) → flattened to (batch, 784)
Output shape: (batch, 10) — raw logits (use ``nn.CrossEntropyLoss``)

Why start from such a small MLP? Two reasons:
1. It is the smallest model that can still beat 10% random accuracy, so
   it works as a *sanity check* for the whole pipeline.
2. It gives a lower-bound accuracy that we use to compare against the
   deeper architectures in :mod:`src.models.architectures`.
"""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """One-hidden-layer MLP baseline.

    Args:
        input_size:   Flattened input size (28*28 = 784 by default).
        hidden_size:  Width of the single hidden layer.
        num_classes:  Number of output classes (10 for FashionMNIST).
        dropout:      Dropout probability in the hidden layer.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, 1, 28, 28)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        x = self.flatten(x)            # (B, 1, 28, 28) -> (B, 784)
        x = self.fc1(x)                # -> (B, hidden_size)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                # -> (B, num_classes)
        return x

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_classes={self.num_classes}, dropout={self.dropout_p}"
        )


if __name__ == "__main__":
    # Tiny self-test: verify shape and that loss decreases over one step.
    import torch.nn.functional as F

    model = BaselineMLP()
    print(model)
    print(f"#parameters: {model.count_parameters():,}")

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    logits = model(x)
    print(f"logits shape: {tuple(logits.shape)}")  # expect (8, 10)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss1 = F.cross_entropy(logits, y).item()
    optimizer.zero_grad()
    F.cross_entropy(logits, y).backward()
    optimizer.step()
    loss2 = F.cross_entropy(model(x), y).item()
    print(f"loss before step: {loss1:.4f}, after step: {loss2:.4f}")
    assert loss2 <= loss1 + 1e-6, "Loss did not decrease!"
    print("BaselineMLP self-test passed.")