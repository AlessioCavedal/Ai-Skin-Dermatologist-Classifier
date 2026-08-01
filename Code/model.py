"""Le due architetture: la CNN che ho progettato e il backbone pre-addestrato."""

from __future__ import annotations

import torch
from torch import nn

ARCHITECTURES = ["cnn_v1", "resnet18"]


class SkinCancerCNNv1(nn.Module):
    """
    Cinque blocchi Conv-BatchNorm-ReLU-MaxPool piu' un classificatore denso.

    Dei 5.831.298 parametri, 4.260.738 stanno nel primo layer lineare: e' il
    punto piu' esposto al sovradattamento, ed e' per questo che subito dopo
    c'e' un dropout a 0.5. Insieme al weight decay e all'early stopping sul
    validation, tiene la capacita' della rete sotto controllo.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.mlp = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        return self.mlp(x)


def build_model(arch: str, num_classes: int = 2, dropout: float = 0.4) -> nn.Module:
    if arch == "cnn_v1":
        return SkinCancerCNNv1(num_classes)
    if arch == "resnet18":
        from torchvision import models

        # Fine-tuning completo: sostituisco solo la testa e riaddestro tutto.
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, num_classes))
        return net
    raise ValueError(f"Architettura non riconosciuta: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
