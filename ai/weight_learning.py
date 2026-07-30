"""
A-Insight Pro
AI Dynamic Weight Learning V3.0 — Per-Stock Calibration

Changes from V2.1:
- Per-stock adjustment factors (not just global)
- Stock-specific bias correction
- Confidence-weighted aggregation
- Outputs both per-stock and global adjustments
"""

import os, json
import pandas as pd
import numpy as np

FEEDBACK_FILE = "reports/ai_learning_feedback.csv"
OUTPUT_FILE = "reports/predict_adjust.json"
PER_STOCK_OUTPUT = "reports/per_stock_adjust.json"
os.makedirs("reports", exist_ok=True)


def learn_weight():
    """Learn prediction adjustment factors from feedback — per-stock + global."""
    print("=" * 60)
    print("AI Weight Learning V3.0 — Per-Stock Calibration")
    print("=" * 60)

    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback data available")
        save_defaults()
        return 1.0, {}

    df = pd.read_csv(FEEDBACK_FILE, encoding="utf-8-sig")
    if df.empty:
        print("Empty feedback")
        save_defaults()
        return 1.0, {}

    required_cols = ["code", "success_rate", "avg_error", "bias", "samples"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        save_defaults()
        return 1.0, {}

    df["code"] = df["code"].astype(str).str.zfill(6)

    # ── Per-stock adjustment ──
    per_stock = {}
    for _, row in df.iterrows():
        code = str(row["code"]).zfill(6)
        success_rate = float(row["success_rate"])
        avg_error = float(row["avg_error"])
        bias = float(row["bias"])
        samples = int(row["samples"])

        # Confidence weight from sample size
        confidence = min(1.0, samples / 30.0)

        # Compute adjustment multiplier
        multiplier = 1.0

        # Bias correction
        if bias > 5:
            multiplier = 0.75
        elif bias > 2:
            multiplier = 0.88
        elif bias < -5:
            multiplier = 1.20
        elif bias < -2:
            multiplier = 1.10

        # Error magnitude cap
        if avg_error > 8:
            multiplier = min(multiplier, 0.80)
        elif avg_error > 5:
            multiplier = min(multiplier, 0.90)

        # Success rate floor
        if success_rate < 0.3:
            multiplier = min(multiplier, 0.75)

        # Clamp
        multiplier = max(0.60, min(1.30, multiplier))

        per_stock[code] = {
            "multiplier": round(multiplier, 3),
            "confidence": round(confidence, 3),
            "success_rate": round(success_rate, 3),
            "avg_error": round(avg_error, 2),
            "bias": round(bias, 2),
            "samples": samples,
        }

    # ── Global factor (weighted average) ──
    if "sample_weight" in df.columns:
        weights = df["sample_weight"].fillna(0.1)
    else:
        weights = df["samples"].fillna(1) / df["samples"].max()

    if "bias" in df.columns:
        global_bias = np.average(df["bias"].fillna(0), weights=weights)
    else:
        global_bias = 0

    global_mae = np.average(df["avg_error"].fillna(0), weights=weights)
    global_success = np.average(df["success_rate"].fillna(0.5), weights=weights)

    global_factor = 1.0
    if global_bias > 5:
        global_factor = 0.82
    elif global_bias > 2:
        global_factor = 0.90
    elif global_bias < -5:
        global_factor = 1.15
    elif global_bias < -2:
        global_factor = 1.08

    if global_mae > 7:
        global_factor = min(global_factor, 0.88)
    elif global_mae > 5:
        global_factor = min(global_factor, 0.95)

    if global_success < 0.3:
        global_factor = min(global_factor, 0.85)

    global_factor = max(0.70, min(1.30, global_factor))

    # ── Save ──
    # Per-stock adjustments
    with open(PER_STOCK_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(per_stock, f, ensure_ascii=False, indent=2)

    # Global factor (backward compatible)
    global_result = {
        "predict_factor": round(global_factor, 3),
        "avg_error": round(float(global_mae), 4),
        "bias": round(float(global_bias), 4),
        "avg_success_rate": round(float(global_success), 4),
        "n_stocks_with_feedback": len(df),
        "per_stock_available": True,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(global_result, f, ensure_ascii=False, indent=2)

    # ── Print ──
    n_reliable = sum(1 for v in per_stock.values() if v["confidence"] >= 0.5)
    n_unreliable = len(per_stock) - n_reliable

    print(f"  Stocks analyzed:       {len(per_stock)}")
    print(f"  Reliable (≥15 samples): {n_reliable}")
    print(f"  Low sample (<15):       {n_unreliable}")
    print(f"  Global factor:          {global_factor:.3f}")
    print(f"  Global bias:            {global_bias:+.2f}%")
    print(f"  Global MAE:             {global_mae:.2f}%")
    print(f"  Global success rate:    {global_success:.1%}")
    print(f"  ──────────────────────────")
    print(f"  Per-stock range:")
    multipliers = [v["multiplier"] for v in per_stock.values()]
    print(f"    Min: {min(multipliers):.3f}  Max: {max(multipliers):.3f}  "
          f"Median: {np.median(multipliers):.3f}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Output: {PER_STOCK_OUTPUT}")
    print("=" * 60)

    return global_factor, per_stock


def save_defaults():
    """Save default adjustment files when no feedback exists."""
    defaults = {
        "predict_factor": 1.0,
        "avg_error": 0.0,
        "bias": 0.0,
        "avg_success_rate": 0.5,
        "n_stocks_with_feedback": 0,
        "per_stock_available": False,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(defaults, f, ensure_ascii=False, indent=2)
    with open(PER_STOCK_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({}, f)
    print(f"  Saved default adjustments (no feedback yet)")


if __name__ == "__main__":
    learn_weight()
