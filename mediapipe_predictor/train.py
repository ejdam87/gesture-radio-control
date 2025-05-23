import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from mediapipe_predictor.dataset import HandDataset
from mediapipe_predictor.models.linear_net import LinearNet
from utils.persistency import save_model
from utils.training import train_loop, test_loop

import sys
from pathlib import Path


def main(data_path: str, out_path: str) -> None:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on {dev} device.")

    df = pd.read_parquet(data_path)
    df = df.filter(regex=r'^[xyzc]\d|hand|label')
    train, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])

    learning_rate = 0.01
    batch_size = 16
    epochs = 30

    train_dataset = HandDataset(train, dev)
    test_dataset = HandDataset(test, dev)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size
    )

    val_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    model = LinearNet(14).to(dev)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(val_dataloader, model, loss_fn)
    print("Done!")

    # --- Persistency
    dest = Path(out_path)
    dest.mkdir(parents=True, exist_ok=True)

    print("Saving model to", dest / Path("mediapipe_model.pth"))
    save_model(model, str(dest / Path("mediapipe_model.pth")))
    print("Saved!")
    # ---

if __name__ == "__main__":
    data = sys.argv[1]
    out = sys.argv[2]
    main(data, out)
