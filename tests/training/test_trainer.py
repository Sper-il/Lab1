from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.callbacks import EarlyStopping
from src.training.trainer import Trainer


def _loaders(batch_size: int = 2):
    x = torch.randn(6, 1, 28, 28)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size), DataLoader(dataset, batch_size=batch_size)


def _trainer(tmp_path: Path, batch_size: int = 2):
    train_loader, val_loader = _loaders(batch_size)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = EarlyStopping(
        patience=5,
        best_path=tmp_path / "best.pt",
        last_path=tmp_path / "last.pt",
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device="cpu",
        callbacks=[callback],
        model_name="tiny",
        hyperparameters={"epochs": 2},
    )
    return trainer, callback


def test_train_epoch_updates_parameters_and_returns_weighted_metrics(tmp_path: Path):
    trainer, _ = _trainer(tmp_path, batch_size=4)
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    loss, accuracy = trainer.train_epoch()
    assert torch.isfinite(torch.tensor(loss))
    assert 0 <= accuracy <= 100
    assert any(not torch.equal(before[key], value) for key, value in trainer.model.state_dict().items())


def test_validate_does_not_update_parameters(tmp_path: Path):
    trainer, _ = _trainer(tmp_path)
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    loss, accuracy = trainer.validate()
    assert torch.isfinite(torch.tensor(loss))
    assert 0 <= accuracy <= 100
    assert trainer.model.training is False
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(before[key], value)


def test_train_rejects_empty_loader_and_non_positive_epochs(tmp_path: Path):
    trainer, _ = _trainer(tmp_path)
    empty = DataLoader(TensorDataset(torch.empty(0, 1, 28, 28), torch.empty(0, dtype=torch.long)))
    trainer.train_loader = empty
    with pytest.raises(ValueError, match="empty"):
        trainer.train_epoch()
    with pytest.raises(ValueError, match="num_epochs"):
        trainer.train(0)


def test_train_returns_history_and_restores_best_weights(tmp_path: Path):
    trainer, callback = _trainer(tmp_path)
    scripted = iter(
        [
            (0.2, 80.0),
            (0.4, 85.0),
            (0.3, 85.0),
            (0.5, 85.0),
            (0.4, 82.0),
            (0.6, 80.0),
        ]
    )
    trainer.train_epoch = lambda: next(scripted)
    trainer.validate = lambda: next(scripted)

    history = trainer.train(3)
    assert set(history) == {"train_loss", "val_loss", "train_acc", "val_acc"}
    assert len(history["train_loss"]) == 3
    assert callback.best_epoch == 1
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=True)
    for key, value in best["model_state_dict"].items():
        assert torch.equal(trainer.model.state_dict()[key], value)
