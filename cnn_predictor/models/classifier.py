from torch import Tensor, nn


class Classifier(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)

        # resnet18 produces 512 channels on the last feature extraction layer
        self.proj = nn.Linear(512, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.flatten(start_dim=-3, end_dim=-1)  # (B, C)
        x = self.dropout(x)
        x = self.proj(x)
        return x
