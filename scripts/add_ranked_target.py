"""
Pre-compute cross-sectional ranked future_return and add to feature files.
This converts the noisy raw 5-day return into a stable rank [0,1] target.

Run once after feature engineering, before training.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

FEATURE_DIR = "features"

def add_ranked_targets():
    print("=" * 60)
    print("Adding Cross-Sectional Ranked Targets to Feature Files")
    print("=" * 60)

    # Load all feature files
    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    print(f"Loading {len(files)} feature files...")

    stock_data = {}
    for i, fname in enumerate(files):
        code = fname.replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, fname), encoding="utf-8-sig")
            if "future_return" in df.columns and "date" in df.columns and len(df) >= 60:
                stock_data[code] = df
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(files)}...")

    print(f"  Valid stocks: {len(stock_data)}")

    # Collect all (date, code, future_return) triples
    records = []
    for code, df in stock_data.items():
        for _, row in df.iterrows():
            if pd.notna(row.get("future_return")):
                records.append({
                    "date": row["date"],
                    "code": code,
                    "future_return": row["future_return"],
                })

    print(f"  Records: {len(records)}")
    rank_df = pd.DataFrame(records)

    # Rank within each date
    print("  Computing cross-sectional ranks per date...")
    rank_df["future_return_rank"] = rank_df.groupby("date")["future_return"].rank(pct=True)
    n_dates = rank_df["date"].nunique()
    print(f"  Ranked across {n_dates} dates")

    # Build lookup: {(date, code): rank}
    rank_lookup = {}
    for _, row in rank_df.iterrows():
        rank_lookup[(str(row["date"]), str(row["code"]).zfill(6))] = row["future_return_rank"]

    # Write back to feature files
    print("  Writing ranked targets to feature files...")
    updated = 0
    for code, df in stock_data.items():
        df["future_return_rank"] = np.nan
        for idx, row in df.iterrows():
            key = (str(row["date"]), code)
            if key in rank_lookup:
                df.at[idx, "future_return_rank"] = rank_lookup[key]

        df.to_csv(os.path.join(FEATURE_DIR, f"{code}.csv"), index=False, encoding="utf-8-sig")
        updated += 1
        if updated % 100 == 0:
            print(f"    Updated {updated}/{len(stock_data)}...")

    print(f"  Updated {updated} feature files")
    print("=" * 60)
    print("Ranked targets added! Now retrain with: python main.py --train-rf")
    print("=" * 60)


if __name__ == "__main__":
    add_ranked_targets()
