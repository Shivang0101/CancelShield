"""
src/data/loader.py
==================
CancelShield Data Loader
Loads the Hotel Booking Demand dataset, validates schema,
removes leakage columns, and performs time-based train/val/test split.

Key design decisions:
- Time-based split (not random) to prevent future-leak in temporal data
- Explicit leakage column removal before any split
- Schema validation raises loudly on missing columns
- Returns named DataFrames for downstream traceability
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def validate_schema(df: pd.DataFrame, expected_columns: list) -> None:
    """
    Assert all expected columns are present in the DataFrame.
    Raises ValueError with a clear message listing missing columns.
    """
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing columns: {sorted(missing)}\n"
            f"Dataset has: {sorted(df.columns.tolist())}"
        )
    logger.info("Schema validation passed — all %d expected columns present.", len(expected_columns))

def remove_leakage(df: pd.DataFrame, leakage_cols: list = None) -> pd.DataFrame:
    """
    Drop columns that encode the outcome AFTER the fact.
    reservation_status   — directly says 'Canceled' / 'Check-Out'
    reservation_status_date — the date of that status event

    Including these would give a model 99%+ AUC and be completely useless
    in production where predictions are made BEFORE the booking resolves.
    """
    if leakage_cols is None:
        leakage_cols = ["reservation_status", "reservation_status_date", "assigned_room_type"]
    existing = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=existing)
    logger.info("Removed %d leakage column(s): %s", len(existing), existing)
    return df

def load_raw(path: str = None, config: dict = None) -> pd.DataFrame:
    """
    Load the raw CSV, validate schema, clean obvious issues.

    Steps:
      1. Read CSV
      2. Validate schema (all expected columns present)
      3. Drop duplicate rows
      4. Drop rows where both adults and children are 0 (invalid bookings)
      5. Clip adr < 0 to 0 (negative ADR is a data entry error)
      6. Parse arrival_date_month to a consistent format
    """
    if config is None:
        config = load_config()

    if path is None:
        path = ROOT / config["data"]["raw_path"]

    logger.info("Loading raw data from %s ...", path)
    df = pd.read_csv(path)
    logger.info("Raw shape: %s", df.shape)

    validate_schema(df, config["data"]["expected_columns"])

    # Remove exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows.", n_before - len(df))

    # Remove clearly invalid bookings (0 guests)
    invalid_mask = (df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0)
    df = df[~invalid_mask]
    logger.info("Dropped %d zero-guest rows.", invalid_mask.sum())

    # Clip negative ADR
    df["adr"] = df["adr"].clip(lower=0)

    # Fill missing children/babies with 0 (common in this dataset)
    df["children"] = df["children"].fillna(0).astype(int)
    df["babies"] = df["babies"].fillna(0).astype(int)

    # Fill missing agent/company with 0 (no agent / no company)
    df["agent"] = df["agent"].fillna(0).astype(int)
    df["company"] = df["company"].fillna(0).astype(int)

    # Fill missing country with 'Unknown'
    df["country"] = df["country"].fillna("Unknown")

    logger.info("Clean shape after basic fixes: %s", df.shape)
    return df

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def build_arrival_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a single arrival_date column from the three date components.
    This is required for time-based splitting.
    """
    df = df.copy()
    month_num = df["arrival_date_month"].map(MONTH_MAP)
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + month_num.astype(str).str.zfill(2)
        + "-"
        + df["arrival_date_day_of_month"].astype(str).str.zfill(2),
        errors="coerce",
    )
    nat_count = df["arrival_date"].isna().sum()
    if nat_count > 0:
        logger.warning("Could not parse %d arrival dates — dropping those rows.", nat_count)
        df = df.dropna(subset=["arrival_date"])
    logger.info("Arrival date range: %s → %s", df["arrival_date"].min(), df["arrival_date"].max())
    return df

def time_split(df: pd.DataFrame, config: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset by arrival date — NOT by random index.

    Why? Random splits allow 'future' bookings from 2017 to appear in
    training data, inflating AUC because the model has seen the same
    booking period from multiple angles.

    Split logic (configurable in config.yaml):
      Train : arrival_year <= 2016           (~67k rows, 2015-2016)
      Val   : arrival_year == 2017, month <= 6  (~24k rows, Jan-Jun 2017)
      Test  : arrival_year == 2017, month >= 7  (~28k rows, Jul-Aug 2017)
    """
    if config is None:
        config = load_config()
    split_cfg = config["data"]["time_split"]

    if "arrival_date" not in df.columns:
        df = build_arrival_date(df)

    train_mask = df["arrival_date_year"] <= split_cfg["train_end_year"]
    val_mask = (df["arrival_date_year"] == split_cfg["val_year"]) & \
               (df["arrival_date_month"].map(MONTH_MAP) <= split_cfg["val_end_month"])
    test_mask = (df["arrival_date_year"] == split_cfg["val_year"]) & \
                (df["arrival_date_month"].map(MONTH_MAP) >= split_cfg["test_start_month"])

    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()

    logger.info("Time split → Train: %d | Val: %d | Test: %d", len(train), len(val), len(test))
    logger.info(
        "Cancel rate → Train: %.1f%% | Val: %.1f%% | Test: %.1f%%",
        train["is_canceled"].mean() * 100,
        val["is_canceled"].mean() * 100,
        test["is_canceled"].mean() * 100,
    )
    return train, val, test

def get_splits(path: str = None) -> Dict[str, pd.DataFrame]:
    """
    Full pipeline: load → validate → remove leakage → split.

    Returns
    -------
    dict with keys: 'train', 'val', 'test', 'full'
    All DataFrames have leakage columns removed and arrival_date column added.
    """
    config = load_config()

    df = load_raw(path, config)
    df = remove_leakage(df, config["data"]["leakage_columns"])
    df = build_arrival_date(df)

    train, val, test = time_split(df, config)

    return {
        "train": train,
        "val": val,
        "test": test,
        "full": df,
    }

def dataset_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the loaded dataset for EDA."""
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Hotels         : {df['hotel'].value_counts().to_dict()}")
    if "is_canceled" in df.columns:
        cr = df["is_canceled"].mean() * 100
        print(f"Cancel rate    : {cr:.1f}%")
    if "adr" in df.columns:
        print(f"ADR (EUR)      : mean={df['adr'].mean():.1f}, std={df['adr'].std():.1f}, "
              f"min={df['adr'].min():.1f}, max={df['adr'].max():.1f}")
    print(f"Missing values : {df.isna().sum().sum()}")
    print("=" * 60)


if __name__ == "__main__":
    splits = get_splits()
    dataset_summary(splits["full"])
    print("\nSplit shapes:")
    for name, split_df in splits.items():
        print(f"  {name:8s}: {split_df.shape}")