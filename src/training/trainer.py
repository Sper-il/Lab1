"""Core training and validation loop for classification models."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


class Trainer:
    """Run supervised classification epochs and coordinate callbacks."""

    HISTORY_KEYS = ("train_loss", "val_loss", "train_acc", "val_acc")

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable,
        val_loader: Iterable,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str | torch.device = "cpu",
        callbacks: list[Any] | None = None,
        model_name: str | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.callbacks = list(callbacks or [])
        self.model_name = model_name
        self.hyperparameters = dict(hyperparameters or {})
        self.history: dict[str, list[float]] = {key: [] for key in self.HISTORY_KEYS}

    @staticmethod
    def _ensure_non_empty(loader: Iterable, name: str) -> None:
        try:
            is_empty = len(loader) == 0
        except TypeError:
            is_empty = False
        if is_empty:
            raise ValueError(f"{name} loader is empty")

    def train_epoch(self) -> tuple[float, float]:
        """Run one optimizer epoch and return sample-weighted loss and accuracy."""
        self._ensure_non_empty(self.train_loader, "train")
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("train loader is empty")
        return total_loss / total_samples, 100.0 * total_correct / total_samples

    def validate(self) -> tuple[float, float]:
        """Evaluate one validation epoch without changing model parameters."""
        self._ensure_non_empty(self.val_loader, "validation")
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                batch_size = targets.size(0)
                total_loss += float(loss.item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == targets).sum().item())
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError("validation loader is empty")
        return total_loss / total_samples, 100.0 * total_correct / total_samples

    def train(self, num_epochs: int) -> dict[str, list[float]]:
        """Run training, persist callback checkpoints, and restore best weights."""
        if num_epochs <= 0:
            raise ValueError("num_epochs must be greater than zero")

        self.history = {key: [] for key in self.HISTORY_KEYS}
        for epoch in range(1, num_epochs + 1):
            started = time.perf_counter()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["train_acc"].append(float(train_acc))
            self.history["val_acc"].append(float(val_acc))

            metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            }
            should_stop = False
            for callback in self.callbacks:
                callback_stop = callback.step(
                    epoch=epoch,
                    metrics=metrics,
                    model=self.model,
                    optimizer=self.optimizer,
                    history=self.history,
                    model_name=self.model_name,
                    hyperparameters=self.hyperparameters,
                )
                should_stop = should_stop or bool(callback_stop)

            elapsed = time.perf_counter() - started
            LOGGER.info(
                "Epoch %d/%d - train_loss=%.6f val_loss=%.6f train_acc=%.2f%% val_acc=%.2f%% (%.2fs)",
                epoch,
                num_epochs,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                elapsed,
            )
            if should_stop:
                LOGGER.info("Early stopping triggered at epoch %d", epoch)
                break

        for callback in self.callbacks:
            best_path = getattr(callback, "best_path", None)
            restore_best = getattr(callback, "restore_best", None)
            if callable(restore_best) and best_path is not None and best_path.exists():
                restore_best(self.model, self.device)
                break
        return self.history


__all__ = ["Trainer"]
