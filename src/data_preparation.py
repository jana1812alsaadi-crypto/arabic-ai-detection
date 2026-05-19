# Phase 1: download the KFUPM-JRCAI Arabic generated abstracts dataset
# from Hugging Face and save each split as a local CSV in data/raw.

import os
import pandas as pd
from datasets import load_dataset

LOCAL_RAW_DIR = "data/raw"
DATASET_NAME = "KFUPM-JRCAI/arabic-generated-abstracts"
SPLITS = ["by_polishing", "from_title", "from_title_and_content"]


def download_dataset():
    os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
    print(f"Downloading dataset: {DATASET_NAME}")

    for split in SPLITS:
        print(f"Processing split: {split}")
        try:
            ds = load_dataset(DATASET_NAME, split=split)
            df = ds.to_pandas()
            out_path = os.path.join(LOCAL_RAW_DIR, f"{split}.csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            print(f"  Saved {len(df)} rows to {out_path}")
        except Exception as e:
            print(f"  Error on split '{split}': {e}")

    print("Download complete.")


def inspect_local_data():
    print("Local data overview:")
    for split in SPLITS:
        path = os.path.join(LOCAL_RAW_DIR, f"{split}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  {split}: {len(df)} rows, columns = {list(df.columns)}")


if __name__ == "__main__":
    download_dataset()
    inspect_local_data()
