import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import HandDataset
from models.stupid_net import StupidNet, init_weights
from models.shallow_stupid_net import ShallowStupidNet
from utils.training import train_loop, test_loop


def main(hagrid_path: str, fabia_path: str) -> None:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on {dev} device.")

    hagrid_df = pd.read_parquet(hagrid_path)
    hagrid_x = hagrid_df.filter(regex=r'^[xyzc]\d|hand|label')
    hagrid_train, hagrid_test = train_test_split(hagrid_x, test_size=0.2, random_state=42, stratify=hagrid_df["label"])

    fabia_df = pd.read_parquet(fabia_path)
    fabia_x = fabia_df.filter(regex=r'^[xyzc]\d|hand|label')
    fabia_train, fabia_test = train_test_split(fabia_x, test_size=0.2, random_state=42, stratify=fabia_df["label"])

    learning_rate = 0.01
    batch_size = 16
    epochs = 15

    hagrid_train_dataset = HandDataset(hagrid_train, dev)
    hagrid_test_dataset = HandDataset(hagrid_test, dev)

    fabia_train_dataset = HandDataset(fabia_train, dev)
    fabia_test_dataset = HandDataset(fabia_test, dev)

    train_dataloader = DataLoader(
        ConcatDataset([hagrid_train_dataset, fabia_train_dataset]),
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        ConcatDataset([hagrid_test_dataset, fabia_test_dataset]),
        batch_size=batch_size
    )

    fabia_train_dataloader = DataLoader(
        fabia_train_dataset,
        batch_size=batch_size
    )

    fabia_test_dataloader = DataLoader(
        fabia_test_dataset,
        batch_size=batch_size
    )

    # model = StupidNet().to(dev)
    # model.apply(init_weights)
    model = ShallowStupidNet().to(dev)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(fabia_train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(fabia_test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    main(
        r"data\landmarks\hagrid_color\normalized\normalized_combined.parquet",
        r"data\landmarks\fabia_esp\normalized\normalized_combined.parquet"
    )
