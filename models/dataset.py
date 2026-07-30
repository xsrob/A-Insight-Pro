"""
A-Insight Pro
PyTorch Dataset for time-series stock data
- Rolling window sequences with strict temporal splits
- No look-ahead bias
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_DIR = "features"

# Default feature columns (normalized versions)
DEFAULT_FEATURES = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma10", "ma20", "ma60",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "rsi_14", "macd", "macd_signal", "macd_histogram",
    "bollinger_pct_b", "bollinger_bandwidth",
    "obv", "mfi_14", "atr_14",
    "volume_ratio_20d",
    "price_position_20d", "price_position_60d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma60_ratio",
    "csi300_return_1d", "csi300_return_5d", "csi300_volatility_20d",
    "market_breadth"
]


def load_all_features(feature_dir=FEATURE_DIR):
    """Load all stock feature files into one DataFrame."""
    files = glob.glob(os.path.join(feature_dir, "*.csv"))
    print(f"Loading {len(files)} feature files...")

    all_data = []
    for f in files:
        code = os.path.basename(f).replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "future_return" not in df.columns or len(df) < 120:
                continue
            df["code"] = code
            df["date"] = pd.to_datetime(df["date"])
            all_data.append(df)
        except Exception as e:
            print(f"  Skip {code}: {e}")

    if not all_data:
        raise ValueError("No valid feature files found!")

    result = pd.concat(all_data, ignore_index=True)
    print(f"  Total rows: {len(result)}, codes: {result['code'].nunique()}")
    return result


def get_available_features(df):
    """Return feature columns that actually exist in the DataFrame."""
    feature_cols = []
    for col in DEFAULT_FEATURES:
        if col in df.columns:
            feature_cols.append(col)
    # Fallback: use normalized columns if available
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    if norm_cols:
        return norm_cols
    return feature_cols


class StockSequenceDataset(Dataset):
    """
    Creates (seq_len, n_features) -> (1,) pairs for LSTM training.

    Each sample is a 60-day window of features, predicting the
    future_return of the last day in the window.
    """

    def __init__(self, df, seq_len=60, feature_cols=None):
        """
        Args:
            df: DataFrame with 'code', 'date', feature columns, 'future_return'
            seq_len: Number of trading days in each sequence
            feature_cols: List of column names to use as features
        """
        self.seq_len = seq_len

        if feature_cols is None:
            feature_cols = get_available_features(df)
        self.feature_cols = [c for c in feature_cols if c in df.columns]

        print(f"  Features: {len(self.feature_cols)}")
        print(f"  First 5: {self.feature_cols[:5]}")

        # Build sequences per stock (preserving time order)
        self.sequences = []
        self.targets = []

        df = df.sort_values(["code", "date"])

        for code, group in df.groupby("code"):
            group = group.sort_values("date").reset_index(drop=True)

            if len(group) < seq_len + 1:
                continue

            features = group[self.feature_cols].values.astype(np.float32)
            targets = group["future_return"].values.astype(np.float32)

            # Replace inf/nan
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            targets = np.nan_to_num(targets, nan=0.0)

            # Create rolling windows
            for i in range(len(group) - seq_len):
                seq = features[i:i + seq_len]  # [seq_len, n_features]
                tgt = targets[i + seq_len - 1]  # future_return at last position
                self.sequences.append(seq)
                self.targets.append(tgt)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        # Clip extreme targets
        self.targets = np.clip(self.targets, -0.3, 0.3)

        print(f"  Total sequences: {len(self.sequences)}")
        print(f"  Target mean: {self.targets.mean():.4f}, std: {self.targets.std():.4f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.targets[idx]])
        )


def split_by_date(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split data by date (not random) to prevent look-ahead bias.
    Returns (train_df, val_df, test_df).
    """
    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].unique()
    dates = sorted(dates)

    n = len(dates)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train_df = df[df["date"].isin(train_dates)]
    val_df = df[df["date"].isin(val_dates)]
    test_df = df[df["date"].isin(test_dates)]

    print(f"  Train: {train_dates[0].date()} -> {train_dates[-1].date()} ({len(train_df)} rows)")
    print(f"  Val:   {val_dates[0].date()} -> {val_dates[-1].date()} ({len(val_df)} rows)")
    print(f"  Test:  {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_df)} rows)")

    return train_df, val_df, test_df
