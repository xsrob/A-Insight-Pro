"""
Enrich feature files with cross-sectional rank factors and multi-horizon targets.
Run after feature_engine.py has generated base features.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from collections import defaultdict

FEATURE_DIR = "features"


def add_multi_horizon_targets(stock_dict):
    """Add 10-day and 20-day forward return targets."""
    print("  Adding multi-horizon targets (10d, 20d)...")
    for code, df in stock_dict.items():
        close = df["close"]
        df["future_return_10d"] = close.shift(-10) / close - 1
        df["future_return_20d"] = close.shift(-20) / close - 1
    return stock_dict


def add_cross_sectional_ranks(stock_dict):
    """
    Add cross-sectional rank factors for key metrics on each date.
    These capture RELATIVE value vs peers — the most powerful factor category.
    """
    print("  Computing cross-sectional rank factors...")

    # Metrics to compute cross-sectional ranks for
    # Using columns that already exist in feature files
    metrics = {
        "return_5d": "return_5d",
        "return_20d": "return_20d",
        "volatility": "volatility",
        "rsi": "rsi",
        "volume_ratio_20d": "volume_ratio_20d",
        "ma20_ratio": "ma20_ratio",
        "max_ret_5d": "max_ret_5d",
        "reversal_1d": "reversal_1d",
        "amihud_illiq": "amihud_illiq",
        "idio_vol_20d": "idio_vol_20d",
    }

    # Collect (date, code, metric_value) for each metric
    metric_data = {m: defaultdict(dict) for m in metrics}
    all_dates = set()

    for code, df in stock_dict.items():
        for _, row in df.iterrows():
            date = row["date"]
            all_dates.add(date)
            for metric_name, col_name in metrics.items():
                if col_name in df.columns:
                    val = row[col_name]
                    if pd.notna(val):
                        metric_data[metric_name][date][code] = val

    # For each date, rank stocks cross-sectionally
    sorted_dates = sorted(all_dates)
    rank_columns = {m: defaultdict(dict) for m in metrics}

    for date in sorted_dates:
        for metric_name in metrics:
            date_data = metric_data[metric_name].get(date, {})
            if len(date_data) < 20:
                continue

            # Rank stocks: higher value = higher rank (1.0 = top)
            sorted_codes = sorted(date_data, key=date_data.get)
            n = len(sorted_codes)
            for rank_idx, code in enumerate(sorted_codes):
                rank_columns[metric_name][code][date] = rank_idx / (n - 1) if n > 1 else 0.5

    # Merge ranks into stock DataFrames
    print(f"    Merging ranks for {len(metrics)} metrics across {len(sorted_dates)} dates...")
    for code, df in stock_dict.items():
        for metric_name in metrics:
            col_name = f"cs_rank_{metric_name}"
            df[col_name] = 0.5  # Default: median

            if code in rank_columns[metric_name]:
                rank_map = rank_columns[metric_name][code]
                for idx, row in df.iterrows():
                    date_val = row["date"]
                    if date_val in rank_map:
                        df.at[idx, col_name] = rank_map[date_val]

    print(f"    Added {len(metrics)} cross-sectional rank factors")
    return stock_dict


def add_ranked_targets_multi_horizon(stock_dict):
    """Add cross-sectional ranked targets for 5d, 10d, 20d horizons."""
    print("  Computing cross-sectional ranked targets (5d/10d/20d)...")

    for horizon, target_col in [(5, "future_return"), (10, "future_return_10d"), (20, "future_return_20d")]:
        rank_col = f"future_return_rank_{horizon}d"

        # Collect all (date, code, return) triples
        records = []
        for code, df in stock_dict.items():
            if target_col not in df.columns:
                continue
            for _, row in df.iterrows():
                if pd.notna(row.get(target_col)):
                    records.append({
                        "date": row["date"],
                        "code": code,
                        "return": row[target_col],
                    })

        if not records:
            continue

        rank_df = pd.DataFrame(records)
        rank_df[rank_col] = rank_df.groupby("date")["return"].rank(pct=True)

        # Build lookup
        rank_lookup = {}
        for _, row in rank_df.iterrows():
            rank_lookup[(str(row["date"]), str(row["code"]).zfill(6))] = row[rank_col]

        # Write back
        for code, df in stock_dict.items():
            df[rank_col] = 0.5
            for idx, row in df.iterrows():
                key = (str(row["date"]), code)
                if key in rank_lookup:
                    df.at[idx, rank_col] = rank_lookup[key]

        n_dates = rank_df["date"].nunique()
        print(f"    {horizon}d ranked target: {len(rank_df)} records, {n_dates} dates")

    return stock_dict


def enrich():
    """Main enrichment pipeline."""
    print("=" * 60)
    print("Enriching Feature Files")
    print("=" * 60)

    # Load all feature files
    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    print(f"Loading {len(files)} feature files...")

    stock_dict = {}
    for i, fname in enumerate(files):
        code = fname.replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, fname), encoding="utf-8-sig")
            if len(df) >= 60:
                stock_dict[code] = df
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(files)}...")

    print(f"  Valid stocks: {len(stock_dict)}")

    # Step 1: Add multi-horizon targets
    stock_dict = add_multi_horizon_targets(stock_dict)

    # Step 2: Add cross-sectional rank factors
    stock_dict = add_cross_sectional_ranks(stock_dict)

    # Step 3: Add ranked targets for all horizons
    stock_dict = add_ranked_targets_multi_horizon(stock_dict)

    # Save all
    print(f"\n  Saving {len(stock_dict)} enriched feature files...")
    for code, df in stock_dict.items():
        df.to_csv(os.path.join(FEATURE_DIR, f"{code}.csv"), index=False, encoding="utf-8-sig")

    # Count new columns
    sample = stock_dict[list(stock_dict.keys())[0]]
    cs_cols = [c for c in sample.columns if c.startswith("cs_rank_")]
    target_cols = [c for c in sample.columns if "future_return_rank" in c or "future_return_10" in c or "future_return_20" in c]
    print(f"  New cross-sectional factors: {len(cs_cols)}")
    print(f"  New target columns: {len(target_cols)}")
    print(f"  Total columns: {len(sample.columns)}")
    print("=" * 60)
    print("Enrichment complete! Retrain with: python main.py --train-rf")
    print("=" * 60)


if __name__ == "__main__":
    enrich()
