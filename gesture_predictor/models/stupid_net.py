from torch import nn
import torch


class StupidNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(85, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        logits = self.linear_relu_stack(x).squeeze(1)
        return logits


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        if m.out_features == 5:  # Final layer
            nn.init.xavier_uniform_(m.weight)  # Xavier for final layer
            nn.init.zeros_(m.bias)  # Initialize bias to 0
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He init for hidden layers
            nn.init.zeros_(m.bias)  # Bias init
