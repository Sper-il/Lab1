import pytest
import torch

from src.training.optimizers import create_optimizer


def test_create_adam_uses_requested_hyperparameters():
    model = torch.nn.Linear(4, 2)
    optimizer = create_optimizer(
        "adam", model.parameters(), learning_rate=0.001, weight_decay=0.01
    )
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 0.001
    assert optimizer.defaults["weight_decay"] == 0.01


def test_create_sgd_uses_momentum():
    model = torch.nn.Linear(4, 2)
    optimizer = create_optimizer("sgd", model.parameters(), learning_rate=0.01)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.9


@pytest.mark.parametrize("name", ["unknown", "", "rmsprop"])
def test_create_optimizer_rejects_unknown_name(name):
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        create_optimizer(name, model.parameters(), learning_rate=0.001)


def test_create_optimizer_rejects_invalid_hyperparameters():
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="learning_rate"):
        create_optimizer("adam", model.parameters(), learning_rate=0)
    with pytest.raises(ValueError, match="weight_decay"):
        create_optimizer("adam", model.parameters(), learning_rate=0.001, weight_decay=-1)
