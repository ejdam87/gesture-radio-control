from gesture_predictor.models.stupid_net import StupidNet

from hand_graph_extraction.get_hand_graph import process_image
from hand_graph_extraction.normalize_hand_graph import rescale_landmarks, find_corners
from hand_graph_extraction.constants import LANDMARKS_COUNT

from utils.persistency import load_model
from utils.constants import LABELS_ALL_PATH

from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

import sys
import json
from pathlib import Path


def run_model(model_path: str, im_path: str) -> int:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = StupidNet(14).to(dev)
    load_model(model, model_path)
    model.eval()

    raw_landmarks = process_image(im_path)
    if not raw_landmarks:
        return 5

    columns = ["path", "hand"] + [f"{axis}{i}" for i in range(21) for axis in "xyzc"]
    df = pd.DataFrame([raw_landmarks], columns=columns)
    df[["x_min", "x_max", "y_min", "y_max"]] = df.apply(find_corners, axis=1)
    columns = [f"{axis}{i}" for i in range(LANDMARKS_COUNT) for axis in "xy"]
    df[columns] = df.apply(rescale_landmarks, axis=1)
    norm_landmarks = df.iloc[0]
    norm_landmarks = norm_landmarks.filter(regex=r'^[xyzc]\d|hand')
    norm_landmarks = torch.tensor(norm_landmarks, dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        pred = model(norm_landmarks)

    class_id = pred.argmax(1).item()
    return class_id


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
    im_dir = sys.argv[2]

    for im_path in Path(im_dir).glob("*"):
        pred = run_model(model_path, im_path)
        visualize(im_path, pred)
