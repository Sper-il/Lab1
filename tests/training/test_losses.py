import pytest
import torch.nn as nn

from src.training.losses import create_loss


def test_create_cross_entropy_loss():
    assert isinstance(create_loss("cross_entropy"), nn.CrossEntropyLoss)


def test_create_loss_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown_loss"):
        create_loss("unknown_loss")
