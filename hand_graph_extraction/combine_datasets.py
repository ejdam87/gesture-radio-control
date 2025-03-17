import sys
from pathlib import Path
import pandas as pd

def combine():
    """Combine multiple datasets into one.
    
    Combines multiple datasets of normalized hand landmarks into one
    and labels them based on the order on the command line.
    
    Example usage: python ./combine_datasets.py ./dataset1.parquet ./dataset2.parquet ... ./combined_output.parquet
    """
    if len(sys.argv) < 2:
        raise ValueError("Specify at least one input parquet file. Last one is output.")

    out_df = pd.read_parquet(Path(sys.argv[1]))
    out_df["label"] = 0
    for i in range(2, len(sys.argv) - 1):
        df = pd.read_parquet(Path(sys.argv[i]))
        df["label"] = i - 1
        out_df = pd.concat([out_df, df], ignore_index=True)

    out_df.to_parquet(Path(sys.argv[-1]))

if __name__ == "__main__":
    combine()