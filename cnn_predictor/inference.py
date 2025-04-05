import sys
import json

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.persistency import load_model
from cnn_predictor.models import Classifier, ResNet18, GestureCNN
from utils.constants import LABELS_PATH
from torchvision.transforms import v2


def run_model(model_path: str, stats_path: str, im_path: str) -> int:
    model = GestureCNN(ResNet18(), Classifier(5))
    load_model(model, model_path)
    model.eval()

    with open(stats_path, "r") as f:
        stats = json.load(f)
        means = stats["means"]
        stds = stats["stds"]

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=means, std=stds)
        ]
    )

    im = Image.open(im_path)
    im_tensor = transforms(im)
    pred = model( im_tensor.unsqueeze(0) )
    class_id = pred.argmax(1).item()
    return class_id


def visualize(im_path: str, pred: int) -> None:

    with open(LABELS_PATH, "r") as f:
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
