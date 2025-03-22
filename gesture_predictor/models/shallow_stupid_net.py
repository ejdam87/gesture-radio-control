from torch import nn
import torch


class ShallowStupidNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(85, 5)

    def forward(self, x: torch.tensor) -> torch.tensor:
        logits = self.linear_layer(x).squeeze(1)
        return logits
