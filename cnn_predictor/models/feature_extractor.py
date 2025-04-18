import torchvision
import torch
from torch import nn


def ResNet18() -> nn.Module:
    resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    return nn.Sequential(*(list(resnet.children())[:-2]))


def EfficientNet() -> nn.Module:
    efnet = torchvision.models.efficientnet_b0(pretrained=True)
    return efnet.features


class CustomExtractor(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return features
