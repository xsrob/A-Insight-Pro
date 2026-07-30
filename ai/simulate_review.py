"""
A-Insight Pro
Simulated Trading Review V2.0 — Scientific Accuracy

Changes from V1:
- Three-tier accuracy: 准确(Hit) / 方向正确(Dir) / 失败(Miss)
- Processes ALL stocks (was: only top 20)
- MAE, RMSE, directional accuracy, hit rate statistics
- Prediction bias detection (over-predict / under-predict)
- Deduplication of input data
"""

import os
import numpy as np
import pandas as pd

PREDICT_FILE = "reports/final_stock_rank.csv"
OUTPUT_FILE = "reports/simulate_review.csv"
SUMMARY_FILE = "reports/simulate_review_summary.csv"


def classify_result(predict, actual, tolerance_pct=50):
    """
    Three-tier accuracy classification:

    - "准确" (Hit):     direction matches AND magnitude error < tolerance%
    - "方向正确" (Dir):  direction matches but magnitude error >= tolerance%
    - "失败" (Miss):    direction DOES NOT match
    """
    if predict == 0:
        return "失败", abs(predict - actual)

    same_direction = (predict > 0 and actual > 0) or (predict < 0 and actual < 0)

    if not same_direction:
        return "失败", abs(predict - actual)

    if predict != 0:
        relative_error = abs(predict - actual) / abs(predict) * 100
    else:
        relative_error = 100

    if relative_error <= tolerance_pct:
        return "准确", abs(predict - actual)
    else:
        return "方向正确", abs(predict - actual)


def run():
    """Run simulated review against latest feature data."""
    print("=" * 60)
    print("AI Simulated Review V2.0 — Scientific Accuracy")
    print("=" * 60)

    if not os.path.exists(PREDICT_FILE):
        print("No AI prediction file found")
        return

    df = pd.read_csv(PREDICT_FILE, encoding="utf-8-sig", dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)

    # Deduplicate
    if len(df) > 0:
        before = len(df)
        df = df.drop_duplicates(subset=["code"], keep="first")
        if before > len(df):
            print(f"Deduplicated: {before} → {len(df)} stocks")

    print(f"Evaluating {len(df)} stocks...")

    results = []
    skipped = 0

    for _, row in df.iterrows():
        code = row["code"]
        feature_file = f"features/{code}.csv"

        if not os.path.exists(feature_file):
            skipped += 1
            continue

        try:
            feature = pd.read_csv(feature_file, encoding="utf-8-sig")
            # Model predicts 5-day forward return (future_return = close.shift(-5)/close - 1)
            # Need at least 6 rows: today + 5 days back for actual 5-day return
            if len(feature) < 6:
                skipped += 1
                continue

            today = feature.iloc[-1]
            five_days_ago = feature.iloc[-6]

            # Actual 5-day return (what the model should have predicted)
            actual_return = (today["close"] / five_days_ago["close"] - 1) * 100
            predict = float(row.get("predict_percent", 0))

            result_label, abs_error = classify_result(predict, actual_return)

            results.append({
                "date": str(today.get("date", "")),
                "code": code,
                "predict_percent": round(predict, 4),
                "actual_return": round(actual_return, 4),
                "abs_error": round(abs_error, 4),
                "error": round(predict - actual_return, 4),
                "result": result_label,
            })

        except Exception as e:
            print(f"  {code}: {e}")
            skipped += 1

    if not results:
        print("No review results (all skipped)")
        return

    review = pd.DataFrame(results)

    # ---- Aggregate Statistics ----
    n = len(review)
    n_hit = (review["result"] == "准确").sum()
    n_dir = (review["result"] == "方向正确").sum()
    n_miss = (review["result"] == "失败").sum()

    directional_accuracy = (n_hit + n_dir) / n * 100
    hit_rate = n_hit / n * 100
    mae = review["abs_error"].mean()
    rmse = np.sqrt((review["error"] ** 2).mean())
    bias = review["error"].mean()

    print("=" * 60)
    print(f"Simulated Review Complete — {n} stocks ({skipped} skipped)")
    print(f"  准确(Hit):         {n_hit:4d}  ({hit_rate:.1f}%)")
    print(f"  方向正确(Dir):     {n_dir:4d}  ({(n_dir/n*100):.1f}%)")
    print(f"  失败(Miss):        {n_miss:4d}  ({(n_miss/n*100):.1f}%)")
    print(f"  方向准确率:        {directional_accuracy:.1f}%")
    print(f"  MAE:              {mae:.4f}%")
    print(f"  RMSE:             {rmse:.4f}%")
    print(f"  预测偏差(Bias):    {bias:+.4f}% {'(over-predict)' if bias > 0 else '(under-predict)'}")
    print("=" * 60)

    # Save review
    os.makedirs("reports", exist_ok=True)
    review.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Output: {OUTPUT_FILE}")

    # Save summary
    summary = pd.DataFrame([{
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": n,
        "hits": int(n_hit),
        "direction_correct": int(n_dir),
        "misses": int(n_miss),
        "directional_accuracy_pct": round(directional_accuracy, 2),
        "hit_rate_pct": round(hit_rate, 2),
        "mae_pct": round(float(mae), 4),
        "rmse_pct": round(float(rmse), 4),
        "bias_pct": round(float(bias), 4),
    }])
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    print(f"Summary: {SUMMARY_FILE}")

    # Top/bottom accuracy examples
    print(f"\nBest predictions:")
    best = review.nsmallest(5, "abs_error")
    for _, r in best.iterrows():
        print(f"  {r['code']} pred:{r['predict_percent']:+.2f}% "
              f"actual:{r['actual_return']:+.2f}% error:{r['abs_error']:.2f}% [{r['result']}]")

    print(f"\nWorst predictions:")
    worst = review.nlargest(5, "abs_error")
    for _, r in worst.iterrows():
        print(f"  {r['code']} pred:{r['predict_percent']:+.2f}% "
              f"actual:{r['actual_return']:+.2f}% error:{r['abs_error']:.2f}% [{r['result']}]")

    return review


if __name__ == "__main__":
    run()
