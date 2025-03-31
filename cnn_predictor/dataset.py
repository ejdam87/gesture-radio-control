import torch
import pandas as pd
from torchvision.transforms import v2
from PIL import Image


class ImageDataset():
    """Dataset of hand images."""

    def __init__(
            self,
            images: pd.Series,
            labels: pd.Series,
            dev: torch.device,
            transforms: v2.Compose | None = None
        ) -> None:
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.dev = dev
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        im_path = self.images.iloc[index]
        label = self.labels.iloc[index]
        im = Image.open(im_path)
        im_tensor = self.transforms(im)

        return im_tensor.to(self.dev), torch.tensor(label).to(self.dev)
