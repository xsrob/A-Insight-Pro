"""
A-Insight Pro
AI Review System V2.0 — Scientific Accuracy Measurement

Changes from V1:
- Accuracy now compares prediction vs actual (direction + magnitude)
- Directional accuracy: did we predict the right direction?
- Precision accuracy: is the magnitude within tolerance?
- Three-tier result: 准确(Hit) / 方向正确(Direction) / 失败(Miss)
- Aggregated stats: MAE, RMSE, Win Rate, Directional Accuracy
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

HISTORY_FILE = "reports/prediction_history.csv"
FEATURE_DIR = "features"
REPORT_FILE = "reports/review_report.csv"


def classify_result(predict, actual, tolerance_pct=50):
    """
    Classify prediction accuracy with three tiers:

    - "准确" (Hit):     direction matches AND magnitude error < tolerance%
    - "方向正确" (Dir):  direction matches but magnitude error >= tolerance%
    - "失败" (Miss):    direction DOES NOT match
    """
    if predict == 0:
        return "失败", abs(predict - actual)

    # Direction match
    same_direction = (predict > 0 and actual > 0) or (predict < 0 and actual < 0)

    if not same_direction:
        return "失败", abs(predict - actual)

    # Direction matched, check magnitude
    if predict != 0:
        relative_error = abs(predict - actual) / abs(predict) * 100
    else:
        relative_error = 100

    if relative_error <= tolerance_pct:
        return "准确", abs(predict - actual)
    else:
        return "方向正确", abs(predict - actual)


def review():
    """Run review comparing historical predictions to actual returns."""
    print("=" * 60)
    print("AI Review V2.0 — Scientific Accuracy")
    print("=" * 60)

    if not os.path.exists(HISTORY_FILE):
        print("No prediction history found")
        return

    history = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")

    # Deduplicate: keep first occurrence of each (date, code) pair
    if "date" in history.columns and "code" in history.columns:
        history["code"] = history["code"].astype(str).str.zfill(6)
        before = len(history)
        history = history.drop_duplicates(subset=["date", "code"], keep="first")
        if before > len(history):
            print(f"Deduplicated: {before} → {len(history)} rows")

    if len(history) == 0:
        print("No reviewable data")
        return

    print(f"Predictions to review: {len(history)}")
    results = []
    skipped = 0

    for _, row in history.iterrows():
        code = str(row["code"]).zfill(6)
        predict = float(row["predict_percent"])

        file = os.path.join(FEATURE_DIR, f"{code}.csv")
        if not os.path.exists(file):
            skipped += 1
            continue

        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
            df["date"] = pd.to_datetime(df["date"])
            predict_date = pd.to_datetime(row["date"])

            # Find the 5th trading day after prediction (model predicts 5-day return)
            future = df[df["date"] > predict_date]
            if len(future) < 5:
                # Not enough future data yet — need 5 trading days
                skipped += 1
                continue

            close_before = df[df["date"] == predict_date]
            if len(close_before) == 0:
                close_before = df[df["date"] <= predict_date]
                if len(close_before) == 0:
                    skipped += 1
                    continue
                close_before = close_before.iloc[-1:]

            close_before_val = close_before["close"].iloc[-1]
            # 5-day forward return = close[+5] / close[0] - 1
            close_after_val = future["close"].iloc[4]  # 5th trading day

            actual_return = (close_after_val / close_before_val - 1) * 100
            error = predict - actual_return

            # Classify with scientific accuracy
            result_label, abs_error = classify_result(predict, actual_return)

            results.append({
                "date": str(row["date"]),
                "code": code,
                "predict_percent": round(predict, 4),
                "actual_return": round(actual_return, 4),
                "abs_error": round(abs_error, 4),
                "error": round(error, 4),
                "result": result_label,
            })

        except Exception as e:
            print(f"  {code}: {e}")
            skipped += 1

    if len(results) == 0:
        print("=" * 40)
        print("No reviewable trades yet (need future data)")
        print("=" * 40)
        return

    report = pd.DataFrame(results)

    # ---- Aggregate Statistics ----
    n = len(report)
    n_hit = (report["result"] == "准确").sum()
    n_dir = (report["result"] == "方向正确").sum()
    n_miss = (report["result"] == "失败").sum()

    # Directional accuracy: 准确 + 方向正确
    directional_accuracy = (n_hit + n_dir) / n * 100
    hit_rate = n_hit / n * 100

    # Error metrics
    mae = report["abs_error"].mean()
    rmse = np.sqrt((report["error"] ** 2).mean())

    # Mean signed error (bias): positive = over-predict, negative = under-predict
    bias = report["error"].mean()

    print("=" * 60)
    print(f"Review Complete — {n} predictions evaluated ({skipped} skipped)")
    print(f"  准确(Hit):         {n_hit:4d}  ({hit_rate:.1f}%)")
    print(f"  方向正确(Dir):     {n_dir:4d}  ({(n_dir/n*100):.1f}%)")
    print(f"  失败(Miss):        {n_miss:4d}  ({(n_miss/n*100):.1f}%)")
    print(f"  方向准确率:        {directional_accuracy:.1f}%")
    print(f"  MAE:              {mae:.4f}%")
    print(f"  RMSE:             {rmse:.4f}%")
    print(f"  预测偏差(Bias):    {bias:+.4f}% {'(over-predict)' if bias > 0 else '(under-predict)'}")
    print("=" * 60)

    # Save detailed report
    report.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    print(f"Detailed report: {REPORT_FILE}")

    # Save summary
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_predictions": n,
        "hits": int(n_hit),
        "direction_correct": int(n_dir),
        "misses": int(n_miss),
        "directional_accuracy_pct": round(directional_accuracy, 2),
        "hit_rate_pct": round(hit_rate, 2),
        "mae_pct": round(float(mae), 4),
        "rmse_pct": round(float(rmse), 4),
        "bias_pct": round(float(bias), 4),
    }
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(os.path.dirname(REPORT_FILE), "review_summary.csv")
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    print(f"Summary: {summary_file}")

    return report


if __name__ == "__main__":
    review()
