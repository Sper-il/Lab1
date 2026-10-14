"""Callbacks for early stopping and training checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0001,
        best_path: str | Path = "models/checkpoints/trained_model.pt",
        last_path: str | Path = "models/checkpoints/last_model.pt",
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if patience <= 0:
            raise ValueError("patience must be greater than zero")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_path = Path(best_path)
        self.last_path = Path(last_path)
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value and value <= self.best_value - self.min_delta
        return value > self.best_value and value >= self.best_value + self.min_delta

    def _checkpoint(
        self,
        epoch: int,
        metrics: Mapping[str, float],
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        history: Mapping[str, list[float]],
        model_name: str | None,
        hyperparameters: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "model_name": model_name or model.__class__.__name__,
            "metrics": dict(metrics),
            "monitor": self.monitor,
            "best_metric": self.best_value,
            "best_epoch": self.best_epoch,
            "history": dict(history),
            "hyperparameters": dict(hyperparameters or {}),
        }

    def step(
        self,
        epoch: int,
        metrics: Mapping[str, float],
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        history: Mapping[str, list[float]] | None = None,
        model_name: str | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> bool:
        """Persist checkpoints and return whether training should stop."""
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' is missing")
        value = float(metrics[self.monitor])
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"Monitored metric '{self.monitor}' must be finite")

        history = history or {}
        improved = self._is_improvement(value)
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        checkpoint = self._checkpoint(
            epoch,
            metrics,
            model,
            optimizer,
            history,
            model_name,
            hyperparameters,
        )
        self.last_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.last_path)
        if improved:
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.best_path)

        return self.bad_epochs >= self.patience

    def restore_best(self, model: nn.Module, device: str | torch.device = "cpu") -> dict[str, Any]:
        """Load best model weights and return the complete checkpoint."""
        if not self.best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {self.best_path}")
        checkpoint = torch.load(self.best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint


__all__ = ["EarlyStopping"]
