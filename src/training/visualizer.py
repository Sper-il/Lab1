"""Training-history plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingVisualizer:
    """Render loss and accuracy curves from a Trainer history mapping."""

    REQUIRED_KEYS = ("train_loss", "val_loss", "train_acc", "val_acc")

    @classmethod
    def _validate_history(cls, history: Mapping[str, Sequence[float]]) -> int:
        missing = [key for key in cls.REQUIRED_KEYS if key not in history]
        if missing:
            raise ValueError(f"history is missing required keys: {', '.join(missing)}")

        lengths = []
        for key in cls.REQUIRED_KEYS:
            values = history[key]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"history['{key}'] must be a sequence")
            lengths.append(len(values))
        if not lengths or lengths[0] == 0:
            raise ValueError("history must contain at least one epoch")
        if len(set(lengths)) != 1:
            raise ValueError("history series must have equal lengths")
        return lengths[0]

    @classmethod
    def plot_loss_curve(
        cls, history: Mapping[str, Sequence[float]], save_path: str | Path
    ) -> Path:
        epochs = cls._validate_history(history)
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        axis = range(1, epochs + 1)
        ax.plot(axis, history["train_loss"], label="Train loss")
        ax.plot(axis, history["val_loss"], label="Validation loss")
        ax.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    @classmethod
    def plot_accuracy_curve(
        cls, history: Mapping[str, Sequence[float]], save_path: str | Path
    ) -> Path:
        epochs = cls._validate_history(history)
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        axis = range(1, epochs + 1)
        ax.plot(axis, history["train_acc"], label="Train accuracy")
        ax.plot(axis, history["val_acc"], label="Validation accuracy")
        ax.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Training and validation accuracy")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path


__all__ = ["TrainingVisualizer"]
