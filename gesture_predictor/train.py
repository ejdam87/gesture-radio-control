import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import HandDataset
from model import StupidNet, init_weights


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


def main(dataset_path: str) -> None:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dev = torch.device("cpu")
    print(f"Running on {dev} device.")

    df = pd.read_parquet(dataset_path)
    X = df.filter(regex=r'^[xyzc]\d|hand|label')
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42, stratify=df["label"])

    learning_rate = 0.01
    batch_size = 256
    epochs = 10

    train_dataset = HandDataset(X_train, dev)
    test_dataset = HandDataset(X_test, dev)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = StupidNet().to(dev)
    model.apply(init_weights)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, dev)
        test_loop(test_dataloader, model, loss_fn, dev)
    print("Done!")    

if __name__ == "__main__":
    main("C:\\Users\\JankoHraskoAKAJovyan\\path\\to\\normalized\\combined\\dataset.parquet")
