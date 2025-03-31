import pandas as pd

import json
import sys
from pathlib import Path


LABELS_PATH = "data/label_dict.json"


def main() -> None:
    """
    Creates pairs of (path to image, label of image) for CNN training

    Usage:
        python cnn_predictor/combine_datasets.py path/to/image/folder1 path/to/image/folder2 ... out/path.csv
    """

    with open(LABELS_PATH, "r") as f:
        label_dict = json.load(f)["labels"]

    image_paths = []
    labels = []

    for folder in sys.argv[1:-1]:
        folder_path = Path(folder)
        label = label_dict[folder_path.name]

        for im_path in folder_path.glob("*"):
            image_paths.append(im_path)
            labels.append(label)

    out = sys.argv[-1]
    pd.DataFrame({"paths": image_paths, "labels": labels}).to_csv(out)


if __name__ == "__main__":
    main()
