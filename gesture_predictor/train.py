import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from gesture_predictor.dataset import HandDataset
from gesture_predictor.models.stupid_net import StupidNet, init_weights
from utils.training import train_loop, test_loop


def main(data_path: str) -> None:
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

    model = StupidNet(14).to(dev)
    # model.apply(init_weights)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        test_loop(val_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    main(
        r"data\dfs\mediapipe\mediapipe_all_df.parquet"
    )
