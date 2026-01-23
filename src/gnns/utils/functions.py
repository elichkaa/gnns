import os
import polars as pl
from datasets import Dataset
from loguru import logger
import json
from typing import Any


def save_df_to_json(df: pl.DataFrame, path: str) -> None:
    logger.success(f"Saved dataframe as json to {path}")
    df.write_json(path)


def save_df_to_parquet(df: pl.DataFrame, path: str) -> None:
    logger.success(f"Saved dataframe as parquet to {path}")
    df.write_parquet(path)


def save_dataset_to_json(ds: Dataset, path: str) -> None:
    logger.success(f"Saved dataset as json to {path}")
    ds.to_json(path)


def save_dataset_to_parquet(ds: Dataset, path: str) -> None:
    logger.success(f"Saved dataset as parquet to {path}")
    ds.to_parquet(path)


def save_to_json(data, path: str) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(data, indent=2))
    logger.success(f"Saved json to {path}")


def read_chunks(path: str) -> list[dict[str, Any]]:
    content = []
    with open(path) as f:
        content = json.load(f)
    return content


def folder_exists(path: str) -> bool:
    return os.path.isdir(path)


def read_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def read_json(path: str) -> pl.DataFrame:
    with open(path) as f:
        d = json.load(f)
        return d


def read_jsonl(path: str) -> pl.DataFrame:
    with open(path) as f:
        return [json.loads(line) for line in f]
