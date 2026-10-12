"""
Models Module
=============
Neural network architectures for FashionMNIST classification.

This package exposes four architectures through :func:`ModelFactory.create`:

* ``baseline_mlp`` – :class:`BaselineMLP`       (1 hidden layer, dropout)
* ``deep_mlp``     – :class:`DeepMLP`           (3 hidden layers + BN)
* ``simple_cnn``   – :class:`SimpleCNN`         (2 conv blocks + 2 FC)
* ``resnet_mini``  – :class:`ResNetMini`        (3 residual stages)

Người 2 (training) interacts with this package exclusively through
:func:`ModelFactory.create(name, **kwargs)` so that the architecture
choice is decoupled from the training loop.
"""

__version__ = "0.2.0"

from .baseline_model import BaselineMLP
from .model_factory import ModelFactory
from .model_selection import SelectionEntry, compare_architectures, results_to_markdown

__all__ = [
    "BaselineMLP",
    "ModelFactory",
    "SelectionEntry",
    "compare_architectures",
    "results_to_markdown",
]