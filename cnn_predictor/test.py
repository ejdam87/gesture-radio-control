from cnn_predictor.inference import cnn_model
from cnn_predictor.dataset import ImageDataset
from utils.training import test_loop

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import json
import sys


def test_model(model_path: str, stats_path: str, data_path: str) -> None:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = cnn_model(model_path, device=dev)

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

    df = pd.read_csv(data_path)
    ds = ImageDataset(df["paths"], df["labels"], dev, transforms)
    dl = DataLoader(ds, batch_size=16)
    test_loop(dl, model, torch.nn.CrossEntropyLoss())


if __name__ == "__main__":
    test_model( sys.argv[1], sys.argv[2], sys.argv[3] )
