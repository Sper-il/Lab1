"""
Architectures Package
=====================
Network architectures for FashionMNIST classification.

Available architectures
-----------------------
* :class:`DeepMLP`         - 3 hidden FC layers + BatchNorm + Dropout
* :class:`SimpleCNN`       - 2 conv blocks + 2 FC layers
* :class:`ResNetMini`      - 3 stages of residual blocks + global avg pool

The :class:`BaselineMLP` lives in ``src.models.baseline_model`` because
that is the entry point defined in ``docs/PHASE2_PLAN.md``.
"""

from .advanced_cnn import BasicBlock, ResNetMini
from .mlp import DeepMLP
from .simple_cnn import SimpleCNN

__all__ = [
    "DeepMLP",
    "SimpleCNN",
    "BasicBlock",
    "ResNetMini",
]