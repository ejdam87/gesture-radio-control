import torch
import pandas as pd
from torch import nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.training import train_loop, test_loop
from cnn_predictor.dataset import ImageDataset
from cnn_predictor.models import GestureCNN, ResNet18, Classifier


import sys


transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.4408, 0.4542, 0.4400], std=[0.2707, 0.2813, 0.2789])
    ]
)


def perform_training(images_df: pd.DataFrame) -> None:

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 8
    epochs = 15

    train, val = train_test_split(images_df, test_size=0.1)

    image_paths_train = train["paths"]
    labels_train = train["labels"]
    train_dataset = ImageDataset(image_paths_train, labels_train, dev, transforms)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    image_paths_val = val["paths"]
    labels_val = val["labels"]
    val_dataset = ImageDataset(image_paths_val, labels_val, dev, transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = GestureCNN( ResNet18(), Classifier(5) ).to(dev)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(val_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":

    # pass the result of `combine_datasets.py` as the argument
    df = pd.read_csv(sys.argv[1])
    perform_training(df)
