import torchvision
from torch import nn


def ResNet18() -> nn.Module:
    resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    return nn.Sequential(*(list(resnet.children())[:-2]))
