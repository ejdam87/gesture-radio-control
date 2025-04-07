import sys
import json
from pathlib import Path

import pandas as pd


LABELS_PATH = "data/label_dicts/label_dict_all.json"


def combine() -> None:
    """Combine multiple datasets into one.
    
    Combines multiple datasets of normalized hand landmarks into one.
    
    Example usage: python ./combine_datasets.py ./dataset1.parquet ./dataset2.parquet ... ./combined_output.parquet
    """
    if len(sys.argv) < 2:
        raise ValueError("Specify at least one input parquet file. Last one is output.")

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)["labels"]

    out_df = pd.read_parquet(Path(sys.argv[1]))
    out_df["label"] = 0
    for i in range(1, len(sys.argv) - 1):
        path = Path(sys.argv[i])
        gesture = "_".join( token for token in path.stem.split("_")[1:] ) # assuming file name like ".../xxxx_gesture_name.parquet"

        df = pd.read_parquet(path)
        df["label"] = labels[gesture]
        out_df = pd.concat([out_df, df], ignore_index=True)

    out_df.to_parquet(Path(sys.argv[-1]))

if __name__ == "__main__":
    combine()
