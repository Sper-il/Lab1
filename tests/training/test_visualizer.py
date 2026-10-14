from pathlib import Path

import pytest

from src.training.visualizer import TrainingVisualizer


HISTORY = {
    "train_loss": [1.2, 0.8, 0.5],
    "val_loss": [1.1, 0.9, 0.7],
    "train_acc": [45.0, 65.0, 80.0],
    "val_acc": [40.0, 60.0, 75.0],
}


@pytest.mark.parametrize(
    "history",
    [
        {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
        {"train_loss": [1.0], "val_loss": [1.0], "train_acc": [1.0]},
        {**HISTORY, "val_acc": [40.0, 60.0]},
    ],
)
def test_visualizer_rejects_invalid_history(history, tmp_path: Path):
    with pytest.raises(ValueError):
        TrainingVisualizer.plot_loss_curve(history, tmp_path / "loss.png")


def test_visualizer_writes_loss_and_accuracy_pngs(tmp_path: Path):
    loss_path = TrainingVisualizer.plot_loss_curve(HISTORY, tmp_path / "plots" / "loss.png")
    accuracy_path = TrainingVisualizer.plot_accuracy_curve(
        HISTORY, tmp_path / "plots" / "accuracy.png"
    )

    assert loss_path == tmp_path / "plots" / "loss.png"
    assert accuracy_path == tmp_path / "plots" / "accuracy.png"
    assert loss_path.exists() and loss_path.stat().st_size > 0
    assert accuracy_path.exists() and accuracy_path.stat().st_size > 0
