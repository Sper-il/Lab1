"""
FashionMNIST PyTorch Classification Project
==========================================

Modules:
- data: Dataset loading, EDA, preprocessing, splitting
- models: Neural network architectures
- training: Training loops and optimizers
- evaluation: Metrics and evaluation utilities
"""

__version__ = "0.1.0"
__author__ = "FashionMNIST Team"

from . import dataset
from . import eda
from . import preprocessing
from . import split

__all__ = [
    "dataset",
    "eda",
    "preprocessing",
    "split",
]
