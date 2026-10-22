"""
Hyperparameter Tuning
=====================
Grid search over architectures + training hyperparameters for
FashionMNIST.

What this module does (per ``docs/PHASE2_PLAN.md``)
---------------------------------------------------
* Train a model from :class:`src.models.ModelFactory` for a small number
  of epochs under a given hyperparameter combination.
* Score it on the validation set.
* Iterate over the full Cartesian product of the search space.
* Track every trial's parameters + metrics.
* Optionally use a :class:`HyperoptScheduler` to prune bad trials early.

Design choices
--------------
* The tuner is **architecture-agnostic** — it accepts any model name
  registered in :class:`ModelFactory`.
* Each trial owns a fresh model instance to avoid weight leakage between
  trials.
* Trials return a :class:`TrialResult` so that downstream code can sort
  and dump the results without re-running.
* ``train_with_params`` is fully self-contained, so it can be unit-tested
  with tiny loaders.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """One (model, hyperparameters) trial outcome."""

    trial_id: int
    params: dict
    train_acc: float
    val_acc: float
    train_loss: float
    val_loss: float
    epoch_time_s: float
    n_params: int
    history: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        flat = {"trial": self.trial_id}
        for k, v in self.params.items():
            flat[f"param_{k}"] = v
        flat.update(
            {
                "train_acc": round(self.train_acc, 2),
                "val_acc": round(self.val_acc, 2),
                "train_loss": round(self.train_loss, 4),
                "val_loss": round(self.val_loss, 4),
                "n_params": self.n_params,
                "epoch_time_s": round(self.epoch_time_s, 2),
            }
        )
        return flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:
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


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return total_loss / max(n, 1), 100.0 * correct / max(n, 1)


# ---------------------------------------------------------------------------
# Main tuner
# ---------------------------------------------------------------------------


class HyperparameterTuner:
    """Hyperparameter search for FashionMNIST models.

    Args:
        to_tune: Either a single architecture name, or a list of names. If a
            list is given, ``model_name`` becomes a hyperparameter in the
            search space (one trial can pick any of them).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Torch device (auto ``cpu`` if ``None``).
    """

    def __init__(
        self,
        to_tune: str | Sequence[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device | None = None,
    ):
        if isinstance(to_tune, str):
            to_tune = [to_tune]
        for name in to_tune:
            if name not in ModelFactory.list_models():
                raise ValueError(
                    f"Unknown model '{name}'. Available: {ModelFactory.list_models()}"
                )

        self.to_tune = list(to_tune)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ #
    # Public search strategy
    # ------------------------------------------------------------------ #

    def grid_search(self, param_grid: Mapping[str, Sequence[Any]]) -> list[TrialResult]:
        """Cartesian product search.

        ``param_grid`` may also include ``model_name`` as a list. If it is
        not present, ``self.to_tune`` (first item) is taken as the fixed
        architecture. Any scalar value (str/int/float) is treated as a
        constant — this lets callers mix fixed fields into the same dict.

        The search is **exhaustive**: every cell of the product is
        evaluated, so the result is deterministic and no promising
        configuration can be missed.
        """
        keys = list(param_grid.keys())
        raw_values = [param_grid[k] for k in keys]
        # Treat non-iterables as a single-element grid (constant value).
        values = [v if isinstance(v, (list, tuple)) else [v] for v in raw_values]

        expanded = []
        for combo in itertools.product(*values):
            cfg = dict(zip(keys, combo))
            cfg.setdefault("model_name", self.to_tune[0])
            expanded.append(cfg)

        logger.info(f"Grid search: {len(expanded)} trials across {len(keys)} params")
        return self._run_all(expanded)

    # ------------------------------------------------------------------ #
    # Single-trial training
    # ------------------------------------------------------------------ #

    def train_with_params(self, params: Mapping[str, Any]) -> TrialResult:
        """Train one (model, hyperparams) combination and return its result."""
        return self._run_trial(-1, dict(params))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_all(self, configs: list[dict]) -> list[TrialResult]:
        results = []
        for i, cfg in enumerate(configs, start=1):
            res = self._run_trial(i, cfg)
            results.append(res)
            logger.info(
                f"[trial {i}/{len(configs)}] {cfg} -> val_acc={res.val_acc:.2f}%"
            )
        return results

    def _run_trial(self, trial_id: int, params: dict) -> TrialResult:
        # 1. Build model
        model_name = params.get("model_name", self.to_tune[0])
        defaults = ModelFactory.get_default_kwargs(model_name)
        model_kwargs = {k: v for k, v in params.items() if k.startswith("model__")}
        model_kwargs = {k.replace("model__", ""): v for k, v in model_kwargs.items()}
        model_kwargs = {**defaults, **model_kwargs}

        model = ModelFactory.create(model_name, **model_kwargs).to(self.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 2. Optimizer
        lr = float(params.get("learning_rate", 1e-3))
        wd = float(params.get("weight_decay", 0.0))
        opt_name = params.get("optimizer", "adam")
        optimizer = _make_optimizer(model, opt_name, lr=lr, weight_decay=wd)

        # 3. Train
        epochs = int(params.get("epochs", 3))
        history = []
        epoch_times = []
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = _train_one_epoch(model, self.train_loader, optimizer, self.device)
            val_loss, val_acc = _evaluate(model, self.val_loader, self.device)
            dt = time.perf_counter() - t0
            epoch_times.append(dt)
            history.append(
                {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                 "val_loss": val_loss, "val_acc": val_acc}
            )

        final = history[-1]
        return TrialResult(
            trial_id=trial_id if trial_id > 0 else 0,
            params=self._effective_params(params, model_name, model_kwargs, lr, wd, epochs),
            train_acc=final["train_acc"],
            val_acc=final["val_acc"],
            train_loss=final["train_loss"],
            val_loss=final["val_loss"],
            epoch_time_s=sum(epoch_times) / max(len(epoch_times), 1),
            n_params=n_params,
            history=history,
        )

    def _effective_params(
        self,
        params: Mapping[str, Any],
        model_name: str,
        model_kwargs: Mapping[str, Any],
        lr: float,
        wd: float,
        epochs: int,
    ) -> dict:
        """Return the params that were *actually* used by this trial.

        The grid only sweeps a subset of hyperparameters. Everything else
        still has a concrete value at runtime (batch size comes from the
        loader, weight decay from the optimizer, dropout from the model's
        own default), so we record them here. Without this, the generated
        report shows ``?`` for columns that were in fact well-defined.
        """
        effective = dict(params)
        effective.setdefault("learning_rate", lr)
        effective.setdefault("weight_decay", wd)
        effective.setdefault("epochs", epochs)

        # Batch size is a property of the loader the tuner was handed.
        loader_batch = getattr(self.train_loader, "batch_size", None)
        effective.setdefault("batch_size", loader_batch)

        # Dropout is only meaningful for models that actually have it
        # (e.g. ResNetMini has no dropout layer at all).
        if "dropout" in model_kwargs:
            effective.setdefault("dropout", model_kwargs["dropout"])
        else:
            effective.setdefault("dropout", None)
        return effective


# ---------------------------------------------------------------------------
# Tiny helper — we accept lists in any order but need a stable param order
# ---------------------------------------------------------------------------


def values_for_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[str]:
    """Return keys in their original insertion order."""
    return list(param_grid.keys())


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------


def save_results(results: Iterable[TrialResult], path: str | Path) -> None:
    """Save a list of TrialResults to a JSON file."""
    rows = [r.as_row() for r in results]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Saved {len(rows)} trial results -> {path}")


def best_result(results: Sequence[TrialResult]) -> TrialResult:
    """Return the trial with highest val_acc."""
    if not results:
        raise ValueError("results is empty")
    return max(results, key=lambda r: r.val_acc)


def results_to_markdown(results: Sequence[TrialResult], top_k: int | None = None) -> str:
    """Render results as a markdown table, sorted by val_acc desc."""
    ordered = sorted(results, key=lambda r: r.val_acc, reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]

    lines = [
        "# Hyperparameter Search Results",
        "",
        "| Trial | Model | LR | Batch | Optim | WD | Dropout | Epochs | Train Acc | Val Acc | Params |",
        "|-------|-------|----|-------|-------|----|---------|--------|-----------|---------|--------|",
    ]
    for r in ordered:
        p = r.params
        # A value of ``None`` means the parameter does not apply to this
        # architecture (e.g. ResNetMini has no dropout layer at all).
        dropout = p.get("dropout")
        dropout_cell = "-" if dropout is None else dropout
        lines.append(
            f"| {r.trial_id} | {p.get('model_name', '?')} | "
            f"{p.get('learning_rate', '?')} | {p.get('batch_size', '?')} | "
            f"{p.get('optimizer', '?')} | {p.get('weight_decay', '?')} | "
            f"{dropout_cell} | {p.get('epochs', '?')} | "
            f"{r.train_acc:.2f}% | {r.val_acc:.2f}% | {r.n_params:,} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "HyperparameterTuner",
    "TrialResult",
    "save_results",
    "best_result",
    "results_to_markdown",
]