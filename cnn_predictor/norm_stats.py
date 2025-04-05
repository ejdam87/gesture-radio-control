from cnn_predictor.dataset import ImageDataset

import torch
from torchvision.transforms import v2


def compute_norm_stats(images: list[str]) -> tuple[torch.tensor, torch.tensor]:
    dummy = [-1 for _ in range(len(images))]
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]
    )

    dataset = ImageDataset(images, dummy, torch.device("cpu"), transforms)
    means = []
    stds = []

    for i in range( len(dataset) ):
        im, _ = dataset[i]
        means.append(im.mean((1, 2)))
        stds.append(im.std((1, 2)))

    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)
    return mean, std
