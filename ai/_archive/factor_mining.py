
"""
A-Insight Pro
Multi-Factor Mining Engine V1.0

What it does:
1. Computes 25+ candidate alpha factors
2. Calculates IC (Information Coefficient) for each factor vs future returns
3. Ranks factors by predictive power
4. Dynamically adjusts scoring weights based on IC

IC = rank correlation between factor_value(t) and future_return(t+5)
Positive IC → factor predicts returns correctly
|IC| > 0.02 → statistically meaningful
|IC| > 0.05 → strong predictor
"""

import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datetime import datetime

FEATURE_DIR = "features"
REPORT_DIR = "reports"
OUTPUT_FILE = os.path.join(REPORT_DIR, "factor_ic_analysis.csv")
WEIGHTS_FILE = os.path.join(REPORT_DIR, "factor_weights.json")
os.makedirs(REPORT_DIR, exist_ok=True)


def calc_momentum_factors(df):
    """Momentum group: short/medium/long term returns."""
    close = df["close"]
    factors = {}
    for period, name in [(5, "mom_5d"), (10, "mom_10d"), (20, "mom_20d"), (60, "mom_60d")]:
        if len(close) > period:
            factors[name] = close.pct_change(period)
    return factors


def calc_volatility_factors(df):
    """Volatility group: various risk measures."""
    ret = df["close"].pct_change()
    factors = {}
    for period in [5, 10, 20]:
        if len(ret) > period:
            factors[f"vol_{period}d"] = ret.rolling(period).std()
    # Volatility of volatility
    if "vol_20d" in factors:
        factors["vol_of_vol"] = factors["vol_20d"].rolling(20).std()
    return factors


def calc_volume_factors(df):
    """Volume/ liquidity group."""
    volume = df["volume"]
    close = df["close"]
    factors = {}
    # Volume ratio
    if len(volume) > 20:
        factors["vol_ratio"] = volume / volume.rolling(20).mean()
    # Volume trend
    if len(volume) > 5:
        factors["vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()
    # Turnover proxy (volume * close)
    amount = volume * close
    if len(amount) > 20:
        factors["amount_ratio"] = amount / amount.rolling(20).mean()
    # Volume-price correlation
    if len(close) > 20:
        ret = close.pct_change()
        factors["vp_corr"] = ret.rolling(20).corr(volume.pct_change())
    return factors


def calc_technical_factors(df):
    """Technical indicator group."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    factors = {}

    # RSI divergence (distance from 50)
    if "rsi_14" in df.columns:
        factors["rsi_divergence"] = abs(df["rsi_14"] - 50) / 50

    # MACD strength
    if "macd_histogram" in df.columns and "close" in df.columns:
        factors["macd_strength"] = df["macd_histogram"] / close

    # Bollinger position
    if "bollinger_pct_b" in df.columns:
        factors["bb_position"] = df["bollinger_pct_b"]

    # Price position within range
    if len(high) > 20 and len(low) > 20:
        hh = high.rolling(20).max()
        ll = low.rolling(20).min()
        factors["price_position"] = (close - ll) / (hh - ll + 1e-10)

    # Distance from MA20
    if "ma20_ratio" in df.columns:
        factors["ma20_distance"] = df["ma20_ratio"]

    # ATR ratio
    if "atr_14" in df.columns:
        factors["atr_ratio"] = df["atr_14"] / close

    return factors


def calc_quality_factors(df):
    """Quality/Value proxy group (from available data)."""
    close = df["close"]
    factors = {}

    # Return consistency (rolling Sharpe proxy)
    ret = close.pct_change()
    if len(ret) > 20:
        roll_mean = ret.rolling(20).mean()
        roll_std = ret.rolling(20).std()
        factors["sharpe_proxy"] = roll_mean / (roll_std + 1e-10)

    # Max drawdown over period
    if len(close) > 20:
        roll_max = close.rolling(20).max()
        factors["drawdown_20d"] = (close / roll_max - 1)

    # Up/down capture ratio
    if len(ret) > 20:
        up = ret.clip(lower=0).rolling(20).sum()
        down = abs(ret.clip(upper=0).rolling(20).sum())
        factors["up_down_ratio"] = up / (down + 1e-10)

    return factors


def calc_all_factors(df):
    """Compute all candidate factors for a single stock."""
    if len(df) < 60:
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)

    all_factors = {}
    all_factors.update(calc_momentum_factors(df))
    all_factors.update(calc_volatility_factors(df))
    all_factors.update(calc_volume_factors(df))
    all_factors.update(calc_technical_factors(df))
    all_factors.update(calc_quality_factors(df))

    result = pd.DataFrame(all_factors, index=df.index)
    result["date"] = df["date"]

    # Target: future 5-day return
    result["future_return"] = df["close"].shift(-5) / df["close"] - 1

    return result.dropna()


def compute_ic():
    """
    Compute Information Coefficient for all factors across all stocks.
    IC = Spearman rank correlation between factor(t) and future_return(t+5)
    """
    print("=" * 60)
    print("Factor Mining Engine V1.0 - IC Analysis")
    print("=" * 60)

    if not os.path.exists(FEATURE_DIR):
        print("No feature directory found")
        return

    feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    print(f"Feature files: {len(feature_files)}")

    # Sample stocks for speed (first 100)
    sample_files = feature_files[:100]

    all_ic = {}

    for i, f in enumerate(sample_files):
        code = f.replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, f), encoding="utf-8-sig")
            if len(df) < 120:
                continue

            factors_df = calc_all_factors(df)
            if factors_df.empty or "future_return" not in factors_df.columns:
                continue

            target = factors_df["future_return"].dropna()

            for col in factors_df.columns:
                if col in ("date", "future_return"):
                    continue
                factor_vals = factors_df[col].dropna()
                # Align indices
                common_idx = factor_vals.index.intersection(target.index)
                if len(common_idx) < 50:
                    continue

                try:
                    ic, pval = spearmanr(
                        factor_vals.loc[common_idx].rank(),
                        target.loc[common_idx].rank()
                    )
                    if np.isnan(ic):
                        continue
                except Exception:
                    continue

                if col not in all_ic:
                    all_ic[col] = []
                all_ic[col].append(ic)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(sample_files)} stocks...")

        except Exception as e:
            continue

    print(f"\n  Stocks analyzed: {i+1}")

    # Aggregate ICs
    ic_summary = []
    for factor, ics in all_ic.items():
        ics_arr = np.array(ics)
        ic_mean = ics_arr.mean()
        ic_std = ics_arr.std()
        ic_ir = ic_mean / (ic_std + 1e-10)  # Information Ratio
        hit_rate = (ics_arr > 0).mean()  # % of stocks where IC > 0

        ic_summary.append({
            "factor": factor,
            "ic_mean": round(ic_mean, 5),
            "ic_std": round(ic_std, 5),
            "ic_ir": round(ic_ir, 3),
            "hit_rate": round(hit_rate, 3),
            "abs_ic_mean": round(abs(ic_mean), 5),
            "n_stocks": len(ics_arr),
        })

    ic_df = pd.DataFrame(ic_summary).sort_values("abs_ic_mean", ascending=False)

    # ==========================================
    # Select significant factors
    # ==========================================
    # Threshold: |IC| > 0.02 and hit_rate > 0.5
    significant = ic_df[
        (ic_df["abs_ic_mean"] > 0.02) &
        (ic_df["hit_rate"] > 0.5)
    ]

    if significant.empty:
        # Relax threshold
        significant = ic_df[
            (ic_df["abs_ic_mean"] > 0.01) &
            (ic_df["hit_rate"] > 0.45)
        ]

    # ==========================================
    # Generate dynamic weights
    # ==========================================
    if not significant.empty:
        sig = significant.copy()
        sig["weight"] = sig["abs_ic_mean"] / sig["abs_ic_mean"].sum()
        sig["weight"] = sig["weight"].round(4)

        # Normalize to sum to 1
        weights_dict = dict(zip(sig["factor"], sig["weight"]))
    else:
        weights_dict = {"fallback": 1.0}

    # ==========================================
    # Output
    # ==========================================
    ic_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "n_stocks_analyzed": len(sample_files),
            "n_factors_tested": len(ic_df),
            "n_significant": len(significant),
            "weights": weights_dict,
        }, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n  Total factors tested: {len(ic_df)}")
    print(f"  Significant factors (|IC| > 0.02): {len(significant)}")
    print(f"\n  Top 10 factors by IC:")
    print(f"  {'Factor':<20s} {'IC Mean':>8s} {'IR':>8s} {'Hit Rate':>8s}")
    print(f"  {'-'*44}")
    for _, row in ic_df.head(10).iterrows():
        print(f"  {row['factor']:<20s} {row['ic_mean']:>+8.4f} {row['ic_ir']:>8.3f} {row['hit_rate']:>8.3f}")

    print(f"\n  Dynamic weights saved: {WEIGHTS_FILE}")
    print(f"  IC analysis saved: {OUTPUT_FILE}")
    print("=" * 60)

    return ic_df


if __name__ == "__main__":
    compute_ic()
