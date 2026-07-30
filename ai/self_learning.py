"""
A-Insight Pro
AI Self-Learning Feedback System V2.2

Changes from V2.1:
- Supports three-tier accuracy: 准确(Hit) / 方向正确(Dir) / 失败(Miss)
- Directional success = 准确 + 方向正确 (both had correct direction)
- Calculates MAE from abs_error column when available
- Better risk level assessment
"""

import os
import numpy as np
import pandas as pd

# Prefer historical review (30+ samples/stock) over daily simulate (1 sample/stock)
HISTORICAL_REVIEW = "reports/historical_review.csv"
DAILY_REVIEW = "reports/simulate_review.csv"
OUTPUT_FILE = "reports/ai_learning_feedback.csv"


def load_review():
    """Load review data, preferring historical (rich) over daily (sparse)."""
    # Prefer historical review with 30+ samples/stock
    review_file = HISTORICAL_REVIEW if os.path.exists(HISTORICAL_REVIEW) else DAILY_REVIEW
    if not os.path.exists(review_file):
        print("No review data found")
        return None

    print(f"Loading review: {os.path.basename(review_file)}")
    df = pd.read_csv(review_file, encoding="utf-8-sig")
    if df.empty:
        return None

    df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def analyze_stock(group):
    """Compute per-stock accuracy statistics."""
    samples = len(group)

    # Directional success = "准确" + "方向正确" (both predicted correct direction)
    # Legacy "成功" support for backward compatibility
    if "成功" in group["result"].values:
        success = group["result"].eq("成功").sum()
        direction_success = success  # old binary system
    else:
        # New three-tier system
        success = group["result"].eq("准确").sum()  # hit rate
        direction_success = group["result"].isin(["准确", "方向正确"]).sum()  # directional accuracy

    hit_rate = success / samples if samples > 0 else 0
    directional_rate = direction_success / samples if samples > 0 else 0

    # Average prediction error (use abs_error if available)
    if "abs_error" in group.columns:
        avg_error = group["abs_error"].mean()
    elif "error" in group.columns:
        avg_error = group["error"].abs().mean()
    else:
        avg_error = abs(group["actual_return"] - group["predict_percent"]).mean()

    # Signed bias: positive = over-predict, negative = under-predict
    if "error" in group.columns:
        bias = group["error"].mean()
    else:
        bias = (group["predict_percent"] - group["actual_return"]).mean()

    return {
        "samples": samples,
        "hit_count": int(success),
        "direction_count": int(direction_success),
        "hit_rate": round(hit_rate, 4),           # 准确率 (strict)
        "success_rate": round(directional_rate, 4), # 方向准确率 (broad, for backward compat)
        "avg_error": round(float(avg_error), 4),
        "bias": round(float(bias), 4),
    }


def calculate_confidence(row):
    """Sample confidence weight — more samples = higher confidence."""
    max_samples = 30
    weight = row["samples"] / max_samples
    return round(min(weight, 1.0), 3)


def calculate_adjust(row):
    """Calculate score adjustment based on historical accuracy."""
    rate = row["success_rate"]  # directional accuracy
    avg_error = row["avg_error"]
    bias = row.get("bias", 0)

    adjust = 0

    # Directional accuracy adjustment
    if rate >= 0.7:
        adjust += 8
    elif rate >= 0.5:
        adjust += 3
    elif rate < 0.3:
        adjust -= 8

    # Error magnitude adjustment (higher error = lower confidence)
    if avg_error > 8:
        adjust -= 5
    elif avg_error < 2:
        adjust += 5

    # Bias adjustment (systematic over/under prediction)
    if bias > 5:
        adjust += 3   # Consistently over-predicts → boost score to compensate
    elif bias < -5:
        adjust -= 3   # Consistently under-predicts → lower score

    return round(max(-10, min(10, adjust)), 2)


def risk_level(row):
    """Assess prediction reliability."""
    rate = row["success_rate"]
    samples = row["samples"]

    if samples < 3:
        return "样本不足"
    elif rate >= 0.7:
        return "低风险"
    elif rate >= 0.4:
        return "中风险"
    else:
        return "高风险"


def learning():
    """Run self-learning feedback loop."""
    print("=" * 50)
    print("AI Self-Learning V2.2")
    print("=" * 50)

    df = load_review()
    if df is None:
        print("No data to learn from")
        return

    print(f"Review samples: {len(df)}")

    records = []
    for code, group in df.groupby("code"):
        data = analyze_stock(group)
        data["code"] = code
        records.append(data)

    result = pd.DataFrame(records)

    if result.empty:
        print("No stocks to analyze")
        return

    # Sample confidence
    result["sample_weight"] = result.apply(calculate_confidence, axis=1)

    # Raw adjustment
    result["confidence_adjust"] = result.apply(calculate_adjust, axis=1)

    # Final adjustment (weighted by sample size)
    result["final_adjust"] = (result["confidence_adjust"] * result["sample_weight"]).round(2)

    # Risk level
    result["risk_level"] = result.apply(risk_level, axis=1)

    # Sort by reliability
    result = result.sort_values(
        by=["final_adjust", "success_rate"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Save
    os.makedirs("reports", exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Summary
    n = len(result)
    avg_rate = result["success_rate"].mean()
    avg_error = result["avg_error"].mean()

    print("=" * 50)
    print(f"Self-Learning Complete — {n} stocks analyzed")
    print(f"  Avg Directional Accuracy: {avg_rate:.1%}")
    print(f"  Avg Prediction Error:     {avg_error:.2f}%")
    print(f"  Low Risk:  {(result['risk_level']=='低风险').sum()}")
    print(f"  Mid Risk:  {(result['risk_level']=='中风险').sum()}")
    print(f"  High Risk: {(result['risk_level']=='高风险').sum()}")
    print(f"  Insufficient: {(result['risk_level']=='样本不足').sum()}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    learning()
