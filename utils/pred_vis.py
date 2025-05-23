import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.constants import LABELS_ALL_PATH


def visualize(im_path: str, pred: int) -> None:

    with open(LABELS_ALL_PATH, "r") as f:
        label_dict = json.load(f)["inv_labels"]

    im = Image.open(im_path)
    im_arr = np.asarray(im)
    plt.imshow(im_arr)
    plt.title(f"Prediction: {label_dict[str(pred)]}")
    plt.show()
