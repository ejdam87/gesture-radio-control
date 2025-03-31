from torch import Tensor, nn


class GestureCNN(nn.Module):
    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        classes = self.classifier(features)
        return classes
