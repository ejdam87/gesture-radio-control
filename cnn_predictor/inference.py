import sys
import json

import torch
import albumentations as A
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torchvision.transforms import v2

from utils.persistency import load_model
from utils.pred_vis import visualize

from cnn_predictor.models import Classifier, ResNet18, GestureCNN, CustomExtractor

GRAYSALE = False

def cnn_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = GestureCNN(CustomExtractor(1) if GRAYSALE else ResNet18(), Classifier(128 if GRAYSALE else 512, 14))
    load_model(model, model_path)
    model = model.to(device)
    model.eval()
    return model

def cnn_pred(model: torch.nn.Module, frame: NDArray, transforms: A.Compose, device: torch.device) -> int:
    im = transforms(frame)
    with torch.no_grad():
        pred = model(im.unsqueeze(0).to(device))

    class_id = pred.argmax(1).item()
    return class_id


def run_model(model_path: str, stats_path: str, im_path: str, device: torch.device) -> int:
    model = cnn_model(model_path, device)

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
    frame = np.asarray(im)
    return cnn_pred(model, frame, transforms, device)


if __name__ == "__main__":
    model_path = sys.argv[1]
    stats_path = sys.argv[2]
    im_path = sys.argv[3]
    device = torch.device("cpu")

    pred = run_model(model_path, stats_path, im_path, device)
    visualize(im_path, pred)
