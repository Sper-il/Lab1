from pathlib import Path

import pytest
import torch

from src.training.callbacks import EarlyStopping


def _objects():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_early_stopping_rejects_invalid_options(tmp_path: Path):
    with pytest.raises(ValueError):
        EarlyStopping(patience=0, best_path=tmp_path / "best.pt")
    with pytest.raises(ValueError):
        EarlyStopping(min_delta=-0.1, best_path=tmp_path / "best.pt")
    with pytest.raises(ValueError):
        EarlyStopping(mode="sideways", best_path=tmp_path / "best.pt")


def test_min_delta_and_patience_control_stopping(tmp_path: Path):
    model, optimizer = _objects()
    callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
        min_delta=0.0001,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )

    assert callback.step(1, {"val_loss": 0.5}, model, optimizer, {}) is False
    assert callback.step(2, {"val_loss": 0.49995}, model, optimizer, {}) is False
    assert callback.bad_epochs == 1
    assert callback.step(3, {"val_loss": 0.4999}, model, optimizer, {}) is False
    assert callback.bad_epochs == 0


def test_zero_min_delta_requires_strict_directional_change(tmp_path: Path):
    model, optimizer = _objects()
    callback = EarlyStopping(
        patience=2,
        min_delta=0.0,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    assert callback.step(1, {"val_loss": 0.5}, model, optimizer, {}) is False
    assert callback.step(2, {"val_loss": 0.5}, model, optimizer, {}) is False
    assert callback.bad_epochs == 1


def test_best_and_last_checkpoints_have_metadata(tmp_path: Path):
    model, optimizer = _objects()
    callback = EarlyStopping(
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    history = {"train_loss": [0.4], "val_loss": [0.3]}
    config = {"training": {"epochs": 15}}

    callback.step(
        1,
        {"val_loss": 0.3},
        model,
        optimizer,
        history,
        model_name="simple_cnn",
        hyperparameters=config,
    )
    callback.step(
        2,
        {"val_loss": 0.4},
        model,
        optimizer,
        history,
        model_name="simple_cnn",
        hyperparameters=config,
    )

    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert best["epoch"] == 1
    assert last["epoch"] == 2
    for checkpoint in (best, last):
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["model_name"] == "simple_cnn"
        assert checkpoint["history"] == history
        assert checkpoint["hyperparameters"] == config


def test_restore_best_loads_best_weights(tmp_path: Path):
    model, optimizer = _objects()
    callback = EarlyStopping(
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    callback.step(1, {"val_loss": 0.2}, model, optimizer, {})
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    callback.restore_best(model)
    for key, value in best_state.items():
        assert torch.equal(model.state_dict()[key], value)
