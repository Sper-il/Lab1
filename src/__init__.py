"""
FashionMNIST PyTorch Classification Project — Người 1 entry point
================================================================

Modules:
- data:      Dataset loading, EDA, preprocessing, splitting (Phase 1)
- models:    Architectures + selection + factory (Người 1)
- training:  Hyperparameter tuning + scheduler (Người 1) | trainer (Người 2)
- evaluation: Evaluator / I / model IO (Người 3)
"""

__version__ = "0.2.0"
__author__ = "FashionMNIST Team"

from . import data, models, training

__all__ = ["data", "models", "training"]