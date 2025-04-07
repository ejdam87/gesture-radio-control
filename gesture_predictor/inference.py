from gesture_predictor.models.stupid_net import StupidNet
from hand_graph_extraction.get_hand_graph import process_image
from hand_graph_extraction.normalize_hand_graph import rescale_landmarks
from utils.persistency import load_model
from utils.constants import LABELS_ALL_PATH

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import sys
import json


def run_model(model_path: str, stats_path: str, im_path: str) -> int:
    model = StupidNet(14)
    load_model(model, model_path)
    model.eval()

    # TODO: restructure hand graph extraction to be able to use it as a library


def visualize(im_path: str, pred: int) -> None:

    with open(LABELS_ALL_PATH, "r") as f:
        label_dict = json.load(f)["inv_labels"]

    im = Image.open(im_path)
    im_arr = np.asarray(im)
    plt.imshow(im_arr)
    plt.title(f"Prediction: {label_dict[str(pred)]}")
    plt.show()

if __name__ == "__main__":
    model_path = sys.argv[1]
    stats_path = sys.argv[2]
    im_path = sys.argv[3]

    pred = run_model(model_path, stats_path, im_path)
    visualize(im_path, pred)
