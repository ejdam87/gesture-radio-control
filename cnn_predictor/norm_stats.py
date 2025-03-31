from cnn_predictor.dataset import ImageDataset

import torch
from torchvision.transforms import v2

import sys
from pathlib import Path


def compute_norm_stats(images: list[Path]) -> tuple[torch.tensor, torch.tensor]:
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


def main() -> None:
    """
    Computed mean and std for normalization for images from given folders.

    Example usage:
        python -m cnn_predictor.norm_stats path/to/ims1 path/to/ims2 ...
    """

    if len(sys.argv) < 2:
        raise ValueError("Needs at least 1 input path!")

    image_paths = []
    for folder in sys.argv[1:]:
        fold_path = Path(folder)
        image_paths.extend( list(fold_path.glob("*")) )
    
    print( compute_norm_stats(image_paths) )

if __name__ == "__main__":
    main()
