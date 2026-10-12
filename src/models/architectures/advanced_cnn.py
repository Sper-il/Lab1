"""
ResNetMini Architecture (Advanced CNN)
========================================
A small ResNet-style network built from scratch for this lab (not from
torchvision.models). It uses pre-activation residual blocks with optional
projection shortcuts, which is what the design document in
``docs/PHASE2_PLAN.md`` specifies.

Why a ResNet for FashionMNIST?
-------------------------------
* MNIST-like datasets are not really "deep", but residual connections
  still help: they keep gradients flowing through the small stem and let
  us stack 6 conv layers without the network collapsing.
* We pick widths (16 → 32 → 64) that match the input resolution (28×28
  shrinks to 14×14 then 7×7) so that the parameter count stays around
  ~270k — small enough to train fast on CPU, large enough to compete
  with a deeper vanilla CNN.

Spatial dimension trace:
    Input  : (B, 1, 28, 28)
    Stem   : (B, 16, 28, 28)
    Layer1 : (B, 16, 28, 28)
    Layer2 : (B, 32, 14, 14)
    Layer3 : (B, 64, 7, 7)
    AvgPool: (B, 64, 1, 1)
    Output : (B, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block with optional projection shortcut.

    Uses Conv-BN-ReLU-Conv-BN with an additive shortcut. The shortcut is
    a 1x1 convolution when shapes don't match (different stride or
    channel count) and an identity when they do.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: identity when shapes match, otherwise 1x1 conv to
        # match both channels and stride.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNetMini(nn.Module):
    """Mini-ResNet for FashionMNIST.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (1 for FashionMNIST).
        base_width:  Width multiplier for the stem (16 by default).
        num_blocks_per_layer: How many residual blocks per stage.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 1,
        base_width: int = 16,
        num_blocks_per_layer: int = 2,
    ):
        super().__init__()

        self.in_channels = base_width
        self.conv1 = nn.Conv2d(
            in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_width)

        widths = [base_width, base_width * 2, base_width * 4]   # 16, 32, 64
        strides = [1, 2, 2]

        self.layer1 = self._make_layer(widths[0], num_blocks_per_layer, stride=strides[0])
        self.layer2 = self._make_layer(widths[1], num_blocks_per_layer, stride=strides[1])
        self.layer3 = self._make_layer(widths[2], num_blocks_per_layer, stride=strides[2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[2] * BasicBlock.expansion, num_classes)

        self.num_classes = num_classes
        self.in_channels_input = in_channels
        self.base_width = base_width

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels_input}, num_classes={self.num_classes}, "
            f"base_width={self.base_width}"
        )