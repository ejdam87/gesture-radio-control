import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import HandDataset
from models.stupid_net import StupidNet, init_weights
from models.shallow_stupid_net import ShallowStupidNet

def train_loop(
        dataloader: DataLoader,
        model: nn.Module, 
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int) -> None:

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y.long())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module) -> None:

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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
