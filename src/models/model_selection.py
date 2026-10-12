"""
Model Selection
===============
Quick comparison between the four architectures trained with a *fixed*
mini configuration. This answers the question "which architecture family
should the search start from?" before the more expensive hyper-parameter
search runs.

Design (per ``docs/PHASE2_PLAN.md``)
------------------------------------
* Fixed optimiser (Adam, lr=1e-3), fixed batch size (64), fixed epochs
  (``--epochs``  flag, default 3 — we are not selecting hyperparameters
  here, just architectures).
* Each architecture trains for the same number of epochs on the same
  train/val split, then we report validation accuracy + parameter count.
* The result is a small markdown table that becomes a section of the
  Người 1 deliverable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model_factory import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SelectionEntry:
    """Per-architecture result for the comparison table."""

    name: str
    num_params: int
    train_acc: float
    val_acc: float
    train_loss: float
    val_loss: float
    epoch_time_s: float
    history: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "params": self.num_params,
            "train_acc": round(self.train_acc, 2),
            "val_acc": round(self.val_acc, 2),
            "train_loss": round(self.train_loss, 4),
            "val_loss": round(self.val_loss, 4),
            "epoch_time_s": round(self.epoch_time_s, 2),
        }


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (avg_loss, accuracy%) over ``loader``."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / max(n, 1), 100.0 * correct / max(n, 1)


def _train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer, device: torch.device
) -> tuple[float, float]:
    """Train one epoch. Returns (avg_loss, accuracy%)."""
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return total_loss / max(n, 1), 100.0 * correct / max(n, 1)


def _train_and_score(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> SelectionEntry:
    """Train a single architecture for ``epochs`` epochs with fixed lr."""
    model = ModelFactory.create(model_name)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    epoch_times = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = _train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = _evaluate(model, val_loader, device)
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
             "val_loss": val_loss, "val_acc": val_acc}
        )
        logger.info(
            f"[{model_name}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% ({dt:.1f}s)"
        )

    final = history[-1]
    return SelectionEntry(
        name=model_name,
        num_params=n_params,
        train_acc=final["train_acc"],
        val_acc=final["val_acc"],
        train_loss=final["train_loss"],
        val_loss=final["val_loss"],
        epoch_time_s=sum(epoch_times) / len(epoch_times),
        history=history,
    )


def compare_architectures(
    model_names: Iterable[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[SelectionEntry]:
    """Train each architecture in ``model_names`` and rank by val_acc.

    Args:
        model_names: Architecture names to evaluate.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of epochs to train each architecture.
        lr: Fixed learning rate (we don't tune hyperparams here).
        device: Torch device (auto-detected ``cpu`` if ``None``).

    Returns:
        A list of :class:`SelectionEntry`, sorted by ``val_acc`` descending.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: list[SelectionEntry] = []
    for name in model_names:
        try:
            entry = _train_and_score(name, train_loader, val_loader, epochs, lr, device)
            results.append(entry)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{name}] failed: {exc}")
            continue

    results.sort(key=lambda r: r.val_acc, reverse=True)
    return results


def results_to_markdown(results: list[SelectionEntry]) -> str:
    """Render the comparison as a small markdown table."""
    lines = [
        "# Architecture Comparison",
        "",
        "| Model | Params | Train Acc | Val Acc | Train Loss | Val Loss | s/epoch |",
        "|-------|--------|-----------|---------|------------|----------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.num_params:,} | {r.train_acc:.2f}% | {r.val_acc:.2f}% | "
            f"{r.train_loss:.4f} | {r.val_loss:.4f} | {r.epoch_time_s:.2f} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "SelectionEntry",
    "compare_architectures",
    "results_to_markdown",
]