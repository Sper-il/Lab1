"""Reusable training orchestration utilities and configuration helpers."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.data.split import DEFAULT_VAL_RATIO, split_pipeline
from src.models.model_factory import ModelFactory
from src.training.callbacks import EarlyStopping
from src.training.losses import create_loss
from src.training.optimizers import create_optimizer
from src.training.trainer import Trainer
from src.training.visualizer import TrainingVisualizer


_REQUIRED_FIELDS = {
    "model": ("name", "dropout", "num_classes"),
    "training": (
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "loss",
        "weight_decay",
        "device",
        "seed",
        "num_workers",
        "augment",
    ),
    "early_stopping": ("monitor", "mode", "patience", "min_delta"),
}


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    for block, fields in _REQUIRED_FIELDS.items():
        section = _require_mapping(config.get(block), block)
        for field in fields:
            if field not in section:
                raise ValueError(f"Missing required field: {block}.{field}")

    model = config["model"]
    training = config["training"]
    early = config["early_stopping"]
    if not isinstance(model["name"], str) or not model["name"]:
        raise ValueError("model.name must be a non-empty string")
    if (
        not isinstance(model["dropout"], (int, float))
        or not math.isfinite(float(model["dropout"]))
        or not 0 <= model["dropout"] < 1
    ):
        raise ValueError("model.dropout must be in [0, 1)")
    if isinstance(model["num_classes"], bool) or not isinstance(model["num_classes"], int) or model["num_classes"] <= 0:
        raise ValueError("model.num_classes must be a positive integer")

    positive_ints = ("epochs", "batch_size")
    for field in positive_ints:
        if isinstance(training[field], bool) or not isinstance(training[field], int) or training[field] <= 0:
            raise ValueError(f"training.{field} must be a positive integer")
    nonnegative_ints = ("seed", "num_workers")
    for field in nonnegative_ints:
        if isinstance(training[field], bool) or not isinstance(training[field], int) or training[field] < 0:
            raise ValueError(f"training.{field} must be a non-negative integer")
    for field in ("learning_rate", "weight_decay"):
        if (
            not isinstance(training[field], (int, float))
            or not math.isfinite(float(training[field]))
            or training[field] < 0
        ):
            raise ValueError(f"training.{field} must be a non-negative number")
    if training["learning_rate"] == 0:
        raise ValueError("training.learning_rate must be greater than zero")
    if training["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be one of: auto, cpu, cuda")
    if not isinstance(training["optimizer"], str) or training["optimizer"].lower() not in {"adam", "sgd"}:
        raise ValueError("training.optimizer must be one of: adam, sgd")
    if not isinstance(training["loss"], str) or training["loss"].lower() not in {"cross_entropy", "crossentropyloss", "ce"}:
        raise ValueError("training.loss must be cross_entropy")
    if not isinstance(training["augment"], bool):
        raise ValueError("training.augment must be a boolean")

    if early["mode"] not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be 'min' or 'max'")
    if isinstance(early["patience"], bool) or not isinstance(early["patience"], int) or early["patience"] <= 0:
        raise ValueError("early_stopping.patience must be a positive integer")
    if (
        not isinstance(early["min_delta"], (int, float))
        or not math.isfinite(float(early["min_delta"]))
        or early["min_delta"] < 0
    ):
        raise ValueError("early_stopping.min_delta must be non-negative")
    if not isinstance(early["monitor"], str) or not early["monitor"]:
        raise ValueError("early_stopping.monitor must be a non-empty string")
    return copy.deepcopy(dict(config))


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML training configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return _validate_config(loaded)


def apply_overrides(config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    """Return a copied config with only explicitly supplied CLI values applied."""
    updated = copy.deepcopy(dict(config))
    training = updated.setdefault("training", {})
    mappings = {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "learning_rate",
        "device": "device",
        "num_workers": "num_workers",
    }
    for argument, field in mappings.items():
        value = getattr(args, argument, None)
        if value is not None:
            training[field] = value
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        updated["output_dir"] = str(output_dir)
    return _validate_config(updated)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda while refusing unavailable explicitly requested CUDA."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cpu, cuda")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(config: Mapping[str, Any]) -> tuple[torch.nn.Module, dict[str, list[float]], dict[str, Any]]:
    """Assemble and execute the complete training pipeline."""
    effective_config = _validate_config(config)
    training = effective_config["training"]
    model_config = effective_config["model"]
    early_config = effective_config["early_stopping"]
    set_random_seed(training["seed"])
    device = resolve_device(training["device"])

    train_loader, val_loader, _test_loader, _split_config = split_pipeline(
        data_dir=effective_config.get("data_dir", "data/processed"),
        batch_size=training["batch_size"],
        val_ratio=DEFAULT_VAL_RATIO,
        num_workers=training["num_workers"],
    )
    model = ModelFactory.create(
        model_config["name"],
        dropout=model_config["dropout"],
        num_classes=model_config["num_classes"],
    )
    criterion = create_loss(training["loss"])
    optimizer = create_optimizer(
        training["optimizer"],
        model.parameters(),
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )

    checkpoint_dir = Path(effective_config.get("checkpoint_dir", "models/checkpoints"))
    callback = EarlyStopping(
        monitor=early_config["monitor"],
        mode=early_config["mode"],
        patience=early_config["patience"],
        min_delta=early_config["min_delta"],
        best_path=checkpoint_dir / "trained_model.pt",
        last_path=checkpoint_dir / "last_model.pt",
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        callbacks=[callback],
        model_name=model_config["name"],
        hyperparameters=effective_config,
    )
    started = time.perf_counter()
    history = trainer.train(training["epochs"])
    elapsed = time.perf_counter() - started

    output_dir = Path(effective_config.get("output_dir", "results/runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    TrainingVisualizer.plot_loss_curve(history, output_dir / "loss_curve.png")
    TrainingVisualizer.plot_accuracy_curve(history, output_dir / "accuracy_curve.png")

    metadata = {
        "device": str(device),
        "elapsed_seconds": elapsed,
        "completed_epochs": len(history["train_loss"]),
        "best_epoch": callback.best_epoch,
        "best_metric": callback.best_value,
        "config": effective_config,
        "artifacts": {
            "best_checkpoint": str(callback.best_path),
            "last_checkpoint": str(callback.last_path),
            "history": str(history_path),
            "loss_curve": str(output_dir / "loss_curve.png"),
            "accuracy_curve": str(output_dir / "accuracy_curve.png"),
        },
    }
    return trainer.model, history, metadata


__all__ = [
    "apply_overrides",
    "load_config",
    "resolve_device",
    "run_training",
    "set_random_seed",
]
