""" Training CNN model """
import torch
import pandas as pd
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from utils.training import train_loop, test_loop
from utils.persistency import save_model
from utils.constants import LABELS_ALL_PATH

from cnn_predictor.dataset import ImageDataset
from cnn_predictor.models import GestureCNN, Classifier, CustomExtractor, ResNet18
from cnn_predictor.norm_stats import compute_norm_stats

import sys
import json
from pathlib import Path


GRAYSALE = False # whether to train grayscale or rgb model


def perform_training(images_df: pd.DataFrame, out_path: str) -> None:

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", dev)

    # --- Hyper-params
    batch_size = 16
    epochs = 5
    lr = 0.001
    # ---

    # --- Data balancing (without no_gesture)
    with open(LABELS_ALL_PATH, "r") as f:
        label_dict = json.load(f)["labels"]

    no_gesture = images_df[ images_df[ "labels" ] == label_dict["no_gesture"] ]
    rest = images_df[ images_df[ "labels" ] != label_dict["no_gesture"] ]
    X = rest[["paths"]]
    y = rest["labels"]

    ros = RandomOverSampler(random_state=42)

    X_resampled, y_resampled = ros.fit_resample(X, y)
    rest = pd.concat(
        [pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name="labels")],
        axis=1
    )

    images_df = pd.concat( [no_gesture, rest], axis=0 )
    # ---
    print(images_df["labels"].value_counts())

    # --- Data handling
    train, val = train_test_split(images_df, test_size=0.1, stratify=images_df["labels"])
    print("Number of train images", len(train))
    print("Number of validation images", len(val))

    image_paths_train = list(train["paths"])
    labels_train = list(train["labels"])

    means, stds = compute_norm_stats(image_paths_train)
    print("computed train means:", means)
    print("computed train stds:", stds)

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=means, std=stds)
        ]
    )

    train_dataset = ImageDataset(image_paths_train, labels_train, dev, transforms)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    image_paths_val = list(val["paths"])
    labels_val = list(val["labels"])
    val_dataset = ImageDataset(image_paths_val, labels_val, dev, transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    # ---

    # --- Training
    model = GestureCNN( CustomExtractor(1) if GRAYSALE else ResNet18(), Classifier(128 if GRAYSALE else 512, 14) ).to(dev)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(val_dataloader, model, loss_fn)
    print("Done!")
    # ---

    # --- Persistency
    dest = Path(out_path)
    dest.mkdir(parents=True, exist_ok=True)

    print("Saving model to", dest / Path("cnn_model.pth"))
    save_model(model, str(dest / Path("cnn_model.pth")))
    print("Saved!")

    print("Saving norm stats used to", dest / Path("norm_stats.json"))
    with open(dest / Path("norm_stats.json"), "w") as f:
        json.dump({"means": means.tolist(), "stds": stds.tolist()}, f)
    print("Saved!")
    # ---

if __name__ == "__main__":
    """
    Usage: python -m cnn_predictor.train path/to/cnn_df.csv dir/where/to/store/the/output
    """

    # pass the path to the result of `create_df.py` as an argument
    df = pd.read_csv(sys.argv[1])
    out = sys.argv[2]
    perform_training(df, out)
