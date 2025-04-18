import torch
from torchvision.transforms import v2
from PIL import Image


class ImageDataset():
    """Dataset of hand images."""

    def __init__(
            self,
            images: list[str],
            labels: list[int],
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
        im_path = self.images[index]
        label = self.labels[index]
        im = Image.open(im_path)
        # im = im.convert("RGB")
        im_tensor = self.transforms(im)

        return im_tensor.to(self.dev), torch.tensor(label).to(self.dev)
