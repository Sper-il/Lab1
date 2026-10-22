"""
Model Factory
=============
Factory pattern for creating model instances from a string name. This is
the *single point of contact* between Người 1 (Network) and Người 2
(Training) — Người 2 never has to know which architecture a string refers
to.

Usage
-----
>>> from src.models.model_factory import ModelFactory
>>> model = ModelFactory.create("simple_cnn")
>>> model = ModelFactory.create("baseline_mlp", hidden_size=512, dropout=0.3)

Adding a new architecture is one line: register the class in the
``_models`` dict.
"""

from typing import Type

import torch.nn as nn

from .architectures import DeepMLP, ResNetMini, SimpleCNN
from .baseline_model import BaselineMLP


class ModelFactory:
    """Registry-based factory for FashionMNIST architectures."""

    # Registry mapping name -> nn.Module class. Keep this ordered so that
    # iteration order in ``list_models`` matches the search-space in
    # ``configs/hyperopt_config.yaml``.
    _models: dict[str, Type[nn.Module]] = {
        "baseline_mlp": BaselineMLP,
        "deep_mlp": DeepMLP,
        "simple_cnn": SimpleCNN,
        "resnet_mini": ResNetMini,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """Instantiate a model by name.

        Args:
            name: Architecture name (see :attr:`_models`).
            **kwargs: Forwarded to the model constructor.

        Returns:
            The constructed ``nn.Module`` (untrained).

        Raises:
            ValueError: If ``name`` is not registered.
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available architectures: {available}"
            )
        return cls._models[name](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return a list of registered architecture names."""
        return list(cls._models.keys())

    @classmethod
    def register(cls, name: str, model_cls: Type[nn.Module]) -> None:
        """Register a new architecture.

        Args:
            name: String identifier to use with :meth:`create`.
            model_cls: ``nn.Module`` subclass.
        """
        if not issubclass(model_cls, nn.Module):
            raise TypeError(f"{model_cls} must be a subclass of nn.Module")
        cls._models[name] = model_cls

    @classmethod
    def get_default_kwargs(cls, name: str) -> dict:
        """Return the default kwargs used in the search space.

        Useful so that the grid search stays comparable across trials —
        changing dropout via the search overrides only the dropout field,
        everything else keeps its default value.
        """
        defaults = {
            "baseline_mlp": {"dropout": 0.2},
            "deep_mlp": {"dropout": 0.3},
            "simple_cnn": {"dropout": 0.5},
            "resnet_mini": {},
        }
        return defaults.get(name, {})


__all__ = ["ModelFactory"]