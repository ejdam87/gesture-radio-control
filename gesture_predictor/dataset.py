import torch
import pandas as pd


class HandDataset():
    """Dataset of hand landmarks."""

    def __init__(self, dataset_df: pd.DataFrame, dev: torch.device) -> None:
        self.samples_df = dataset_df
        self.dev = dev

    def __len__(self) -> int:
        return len(self.samples_df)

    def __getitem__(self, index: int) -> tuple[torch.tensor, torch.tensor]:
        sample = self.samples_df.iloc[index]
        label = torch.tensor(sample["label"]).to(self.dev)

        landmarks = sample.filter(regex=r'^[xyzc]\d|hand')
        landmarks = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.dev)

        return landmarks, label
