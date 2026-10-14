from pathlib import Path
import argparse
import random

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

import run_training as training_pipeline
from run_training import apply_overrides, load_config, resolve_device, set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_model_config_matches_training_contract():
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"] == {
        "name": "simple_cnn",
        "dropout": 0.5,
        "num_classes": 10,
    }
    assert config["training"] == {
        "epochs": 15,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "loss": "cross_entropy",
        "device": "auto",
        "seed": 42,
        "num_workers": 0,
        "augment": False,
    }
    assert config["early_stopping"] == {
        "monitor": "val_loss",
        "mode": "min",
        "patience": 5,
        "min_delta": 0.0001,
    }


def test_load_config_validates_schema_and_field_paths(tmp_path: Path):
    valid_path = tmp_path / "valid.yaml"
    valid_path.write_text(
        """
model: {name: simple_cnn, dropout: 0.5, num_classes: 10}
training: {epochs: 15, batch_size: 128, learning_rate: 0.001, optimizer: adam, loss: cross_entropy, weight_decay: 0.0, device: auto, seed: 42, num_workers: 0, augment: false}
early_stopping: {monitor: val_loss, mode: min, patience: 5, min_delta: 0.0001}
""",
        encoding="utf-8",
    )
    assert load_config(valid_path)["model"]["name"] == "simple_cnn"

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("training: []", encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        load_config(invalid_path)

    wrong_type_path = tmp_path / "wrong_type.yaml"
    wrong_type_path.write_text(
        "model: {name: simple_cnn, dropout: 0.5, num_classes: 10}\n"
        "training: {epochs: 0}\n"
        "early_stopping: {monitor: val_loss, mode: min, patience: 5, min_delta: 0.0001}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training.batch_size"):
        load_config(wrong_type_path)

    unsupported_path = tmp_path / "unsupported.yaml"
    unsupported_path.write_text(
        "model: {name: simple_cnn, dropout: 0.5, num_classes: 10}\n"
        "training: {epochs: 1, batch_size: 2, learning_rate: .nan, optimizer: rmsprop, loss: mse, weight_decay: 0.0, device: auto, seed: 42, num_workers: 0, augment: false}\n"
        "early_stopping: {monitor: val_loss, mode: min, patience: 5, min_delta: 0.0001}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(unsupported_path)
    unsupported_path.write_text(
        "model: {name: simple_cnn, dropout: 0.5, num_classes: 10}\n"
        "training: {epochs: 1, batch_size: 2, learning_rate: 0.001, optimizer: rmsprop, loss: mse, weight_decay: 0.0, device: auto, seed: 42, num_workers: 0, augment: false}\n"
        "early_stopping: {monitor: val_loss, mode: min, patience: 5, min_delta: 0.0001}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="optimizer"):
        load_config(unsupported_path)
    bool_path = tmp_path / "bool.yaml"
    bool_path.write_text(
        "model: {name: simple_cnn, dropout: 0.5, num_classes: 10}\n"
        "training: {epochs: true, batch_size: 2, learning_rate: 0.001, optimizer: adam, loss: cross_entropy, weight_decay: 0.0, device: auto, seed: 42, num_workers: 0, augment: false}\n"
        "early_stopping: {monitor: val_loss, mode: min, patience: 5, min_delta: 0.0001}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training.epochs"):
        load_config(bool_path)


def test_apply_overrides_does_not_mutate_source_config():
    source = {
        "model": {"name": "simple_cnn", "dropout": 0.5, "num_classes": 10},
        "training": {
            "epochs": 15,
            "batch_size": 128,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "cross_entropy",
            "weight_decay": 0.0,
            "device": "auto",
            "seed": 42,
            "num_workers": 0,
            "augment": False,
        },
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 5, "min_delta": 0.0001},
    }
    args = argparse.Namespace(epochs=4, batch_size=32, lr=0.0005, device="cpu", num_workers=2, output_dir="out")
    updated = apply_overrides(source, args)

    assert updated is not source
    assert updated["training"]["epochs"] == 4
    assert updated["training"]["batch_size"] == 32
    assert updated["training"]["learning_rate"] == 0.0005
    assert updated["training"]["device"] == "cpu"
    assert updated["training"]["num_workers"] == 2
    assert updated["output_dir"] == "out"
    assert source["training"]["epochs"] == 15


def test_resolve_device_and_seed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")

    set_random_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(2))
    set_random_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(2))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def _integration_config(tmp_path: Path) -> dict:
    return {
        "model": {"name": "simple_cnn", "dropout": 0.5, "num_classes": 10},
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.01,
            "optimizer": "adam",
            "loss": "cross_entropy",
            "weight_decay": 0.0,
            "device": "cpu",
            "seed": 42,
            "num_workers": 0,
            "augment": False,
        },
        "early_stopping": {"monitor": "val_loss", "mode": "min", "patience": 5, "min_delta": 0.0001},
        "output_dir": str(tmp_path / "runs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }


def test_run_training_smoke_persists_artifacts_and_consistent_state(monkeypatch, tmp_path: Path):
    x = torch.randn(6, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4, 5])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    monkeypatch.setattr(training_pipeline, "split_pipeline", lambda **_: (loader, loader, loader, None))
    monkeypatch.setattr(
        training_pipeline.ModelFactory,
        "create",
        lambda name, **kwargs: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)),
    )

    model, history, metadata = training_pipeline.run_training(_integration_config(tmp_path))
    output_dir = tmp_path / "runs"
    checkpoint_dir = tmp_path / "checkpoints"
    assert isinstance(model, torch.nn.Module)
    assert set(history) == {"train_loss", "val_loss", "train_acc", "val_acc"}
    assert (checkpoint_dir / "trained_model.pt").exists()
    assert (checkpoint_dir / "last_model.pt").exists()
    assert (output_dir / "training_history.json").exists()
    assert (output_dir / "loss_curve.png").exists()
    assert (output_dir / "accuracy_curve.png").exists()
    assert metadata["completed_epochs"] == len(history["train_loss"])

    best = torch.load(checkpoint_dir / "trained_model.pt", map_location="cpu", weights_only=True)
    for key, value in best["model_state_dict"].items():
        assert torch.equal(model.state_dict()[key], value)
    assert best["epoch"] == metadata["best_epoch"]
    saved_history = __import__("json").loads((output_dir / "training_history.json").read_text(encoding="utf-8"))
    assert saved_history == history


def test_training_public_exports_preserve_person1_api():
    from src.training import HyperparameterTuner, Trainer, TrainingVisualizer

    assert HyperparameterTuner is not None
    assert Trainer is not None
    assert TrainingVisualizer is not None
