import pandas as pd

import json
import sys
from pathlib import Path

from utils.constants import LABELS_ALL_PATH


def main() -> None:
    """
    Creates pairs of (path to image, label of image) for CNN training

    Usage:
        python cnn_predictor/create_df.py path/to/image/folder1 path/to/image/folder2 ... out/path.csv
    """

    with open(LABELS_ALL_PATH, "r") as f:
        label_dict = json.load(f)["labels"]

    image_paths = []
    labels = []

    for folder in sys.argv[1:-1]:
        folder_path = Path(folder)

        # all non-desired gestures are in clas 'no_gesture'
        key = folder_path.name if folder_path.name in label_dict else "no_gesture"
        label = label_dict[key]

        for im_path in folder_path.glob("*"):
            image_paths.append(im_path)
            labels.append(label)

    out = sys.argv[-1]
    pd.DataFrame({"paths": image_paths, "labels": labels}).to_csv(out, index=False)


if __name__ == "__main__":
    main()
