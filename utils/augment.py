""" Offline data augmentations. """

import sys
import os
from pathlib import Path

import numpy as np
import albumentations as A
from PIL import Image


TRANSFORMS = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(sigma_limit=(0.5, 1), p=0.5),
        A.RGBShift(p=0.5),
        A.GaussNoise(std_range=(0.01, 0.08), p=0.5)
    ]
)

ROUNDS = 3

def rem_augmented(path: Path) -> None:
    for file in path.glob("*"):
        if "aug" in file.stem:
            os.remove(str(file))

def main() -> None:
    """
    Augments all the images in directory given by the argument

    Usage:
        python utils/augment.py path/to/the/image/folder
    """

    paths = list( Path(sys.argv[1]).glob("*") )
    for i in range(ROUNDS):
        for path in paths:
            im = Image.open(path)
            arr = np.array(im)
            augmented = TRANSFORMS(image=arr)["image"]
            target_path = path.with_stem(path.stem + f"_aug_{i}")
            Image.fromarray(augmented).save(target_path)

if __name__ == "__main__":
    main()
    # rem_augmented( Path(sys.argv[1]) )
