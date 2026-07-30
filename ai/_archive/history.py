"""
A-Insight Pro
AI Prediction History System V1.1

Changes from V1:
- Deduplicates before saving: removes existing entries for today's date
- Normalizes codes to 6-digit format
"""

import os
import pandas as pd
from datetime import datetime

RANK_FILE = "reports/final_stock_rank.csv"
HISTORY_FILE = "reports/prediction_history.csv"


def save_history():
    """Save today's predictions to history, deduplicating by date+code."""
    print("Reading AI predictions...")

    if not os.path.exists(RANK_FILE):
        print("No prediction file found")
        return

    df = pd.read_csv(RANK_FILE, encoding="utf-8-sig")
    if df.empty:
        print("Predictions empty")
        return

    today = datetime.now().strftime("%Y-%m-%d")

    # Save ALL stocks (was: only top 20)
    # Build today's history
    history = pd.DataFrame()
    history["date"] = today
    history["code"] = df["code"].astype(str).str.zfill(6)
    history["predict_percent"] = df["predict_percent"]
    history["AI_SCORE"] = df["AI_SCORE"]
    history["LEVEL"] = df["LEVEL"]

    # Load existing history and remove duplicates for today
    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig", dtype=str)
        old["code"] = old["code"].astype(str).str.zfill(6)

        # Remove existing entries for today's date
        before = len(old)
        old = old[old["date"] != today]
        removed = before - len(old)
        if removed > 0:
            print(f"Removed {removed} existing entries for {today}")

        history = pd.concat([old, history], ignore_index=True)
    else:
        print("Creating new history file")

    history.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

    print("=" * 40)
    print(f"Prediction history saved: {len(history)} total rows")
    print(f"  Today: {today}, {len(df)} stocks added")
    print(f"  File: {HISTORY_FILE}")
    print("=" * 40)


if __name__ == "__main__":
    save_history()
