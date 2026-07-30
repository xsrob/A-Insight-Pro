"""
A-Insight Pro
Historical Prediction Review V1.0

Walks back through historical feature data for each stock,
generates mock predictions, and compares against actual 5-day returns.

This solves the "样本不足" problem by generating 10-50 review samples
per stock without waiting for real trading days to pass.

Output: reports/historical_review.csv (compatible with self_learning.py)
"""

import os, sys, glob, json, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_FILE = "models/stock_model.pkl"
FEATURE_NAMES_FILE = "models/feature_names.json"
FEATURE_DIR = "features"
REPORT_DIR = "reports"
OUTPUT_FILE = os.path.join(REPORT_DIR, "historical_review.csv")
SUMMARY_FILE = os.path.join(REPORT_DIR, "historical_review_summary.csv")

os.makedirs(REPORT_DIR, exist_ok=True)

# Sampling: review every Nth row to get independent 5-day samples
SAMPLE_STEP = 30  # ~15-30 samples per stock (faster than 10)
MIN_SAMPLES_PER_STOCK = 3


def load_model():
    """Load trained model and feature names."""
    model = joblib.load(MODEL_FILE)
    if os.path.exists(FEATURE_NAMES_FILE):
        with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as f:
            info = json.load(f)
        feature_names = info.get("feature_names", [])
    else:
        feature_names = []
    return model, feature_names


def classify_result(predict, actual, tolerance_pct=50):
    """Three-tier accuracy classification."""
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


def review_one_stock(code, df, model, feature_names):
    """
    Review a single stock's historical predictions.

    Walks through the DataFrame in steps of SAMPLE_STEP,
    predicts at each point, and compares to actual 5-day return.
    """
    results = []
    if len(df) < SAMPLE_STEP + 6:  # Need at least one sample + 5 days ahead
        return results

    # Ensure feature columns exist
    available = [c for c in feature_names if c in df.columns]
    if len(available) < 10:
        return results

    for i in range(0, len(df) - 6, SAMPLE_STEP):
        # Features at prediction point
        row = df.iloc[i]
        # Target: 5-day forward return
        close_now = row.get("close", 0)
        close_future = df.iloc[i + 5].get("close", 0)

        if close_now <= 0 or close_future <= 0:
            continue

        actual_return = (close_future / close_now - 1) * 100

        # Build feature vector
        try:
            X = pd.DataFrame([row[available].fillna(0).values], columns=available)
            X = X.fillna(0).replace([float("inf"), -float("inf")], 0)
            raw_pred = float(model.predict(X)[0])
        except Exception:
            continue

        # Simple factor adjustment (same as predict.py's logic)
        factor_adj = 0.0
        try:
            close_v = float(row.get("close", 0))
            ma20_v = float(row.get("ma20", close_v))
            if close_v > 0 and ma20_v > 0:
                trend = (close_v - ma20_v) / ma20_v
                if abs(trend) > 0.02:
                    factor_adj += np.clip(trend, -0.10, 0.10) * 0.10
            ma5_v = float(row.get("ma5", 0))
            if ma5_v > 0 and ma20_v > 0:
                factor_adj += (0.005 if ma5_v > ma20_v else -0.005)
            rsi_v = float(row.get("rsi", 50))
            if rsi_v > 75:
                factor_adj -= 0.008
            elif rsi_v < 25:
                factor_adj += 0.008
            vol_v = float(row.get("volatility", 0))
            if vol_v > 0.05:
                factor_adj -= 0.005
        except (ValueError, TypeError):
            pass

        factor_adj = np.clip(factor_adj, -0.03, 0.03)
        predicted_return = raw_pred + factor_adj
        predicted_return = max(-0.12, min(0.12, predicted_return))
        predict_percent = predicted_return * 100

        result_label, abs_error = classify_result(predict_percent, actual_return)

        results.append({
            "date": str(row.get("date", "")),
            "code": code,
            "predict_percent": round(predict_percent, 4),
            "actual_return": round(actual_return, 4),
            "abs_error": round(abs_error, 4),
            "error": round(predict_percent - actual_return, 4),
            "result": result_label,
        })

    return results


def run(sample_step=SAMPLE_STEP, max_stocks=None):
    """
    Run historical review across all stocks.

    Args:
        sample_step: Review every Nth row (default 10, ~20-50 samples/stock)
        max_stocks: Limit to N stocks (None = all)
    """
    print("=" * 60)
    print("Historical Review V1.0 — Bootstrapping Review Samples")
    print("=" * 60)

    # Load model
    print("Loading model...")
    model, feature_names = load_model()
    print(f"  Features expected: {len(feature_names)}")

    # Find feature files
    files = glob.glob(os.path.join(FEATURE_DIR, "*.csv"))
    if max_stocks:
        files = files[:max_stocks]

    print(f"  Stocks to review: {len(files)}")
    print(f"  Sample step: every {sample_step} rows")
    print(f"  Reviewing...")

    all_results = []
    stock_sample_counts = {}
    skipped = 0

    for i, f in enumerate(files):
        code = os.path.basename(f).replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if len(df) < sample_step + 6:
                skipped += 1
                continue

            results = review_one_stock(code, df, model, feature_names)
            if results:
                all_results.extend(results)
                stock_sample_counts[code] = len(results)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)} stocks, {len(all_results)} samples...")

        except Exception as e:
            skipped += 1
            continue

    if not all_results:
        print("No review results generated")
        return

    review = pd.DataFrame(all_results)

    # ── Aggregate Statistics ──
    n = len(review)
    n_hit = (review["result"] == "准确").sum()
    n_dir = (review["result"] == "方向正确").sum()
    n_miss = (review["result"] == "失败").sum()

    directional_accuracy = (n_hit + n_dir) / n * 100
    hit_rate = n_hit / n * 100
    mae = review["abs_error"].mean()
    rmse = np.sqrt((review["error"] ** 2).mean())
    bias = review["error"].mean()

    avg_samples = np.mean(list(stock_sample_counts.values())) if stock_sample_counts else 0

    print("=" * 60)
    print(f"Historical Review Complete")
    print(f"  Stocks reviewed:    {len(stock_sample_counts)} ({skipped} skipped)")
    print(f"  Total samples:      {n}")
    print(f"  Avg samples/stock:  {avg_samples:.1f}")
    print(f"  Stocks with >=5:    {sum(1 for v in stock_sample_counts.values() if v >= 5)}")
    print(f"  --- Accuracy ---")
    print(f"  准确(Hit):          {n_hit:5d}  ({hit_rate:.1f}%)")
    print(f"  方向正确(Dir):      {n_dir:5d}  ({(n_dir/n*100):.1f}%)")
    print(f"  失败(Miss):         {n_miss:5d}  ({(n_miss/n*100):.1f}%)")
    print(f"  方向准确率:         {directional_accuracy:.1f}%")
    print(f"  MAE:               {mae:.4f}%")
    print(f"  RMSE:              {rmse:.4f}%")
    print(f"  预测偏差(Bias):     {bias:+.4f}%")
    print("=" * 60)

    # Save to historical_review.csv (separate from daily simulate_review.csv)
    review.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Output: {OUTPUT_FILE} ({n} samples)")

    # Save summary
    summary = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": len(stock_sample_counts),
        "total_samples": n,
        "avg_samples_per_stock": round(avg_samples, 1),
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

    # Show sample count distribution
    counts = pd.Series(stock_sample_counts)
    print(f"\nSample count distribution:")
    for bucket in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 200)]:
        n_stocks = ((counts >= bucket[0]) & (counts < bucket[1])).sum()
        if n_stocks > 0:
            print(f"  [{bucket[0]:3d}-{bucket[1]:3d}): {n_stocks} stocks")

    return review


if __name__ == "__main__":
    run()
