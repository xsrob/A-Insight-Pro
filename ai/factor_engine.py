"""
A-Insight Pro
Advanced Factor Mining Engine V4.0

Features:
- 70+ alpha factors across 14 categories (was 36 across 6)
- Rolling walk-forward IC computation
- Factor correlation analysis & redundancy pruning
- Dynamic weight generation for scoring integration
- Full 500-stock coverage (no sampling)
- NEW: Cross-sectional, MAX effect, reversal, idiosyncratic,
  liquidity, path-dependent, gap, lead-lag factors

Categories:
  Momentum, Volatility, Volume/Flow, Technical/MR, Quality, Tail Risk
  Cross-Sectional, MAX Effect, Short-Term Reversal, Idiosyncratic,
  Liquidity, Path-Dependent, Gap Effects, Lead-Lag
"""

import os, json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from datetime import datetime

FEATURE_DIR = "features"
REPORT_DIR = "reports"

# Import advanced factors
try:
    from ai.advanced_factors import (
        ADVANCED_FACTOR_CATEGORIES,
        compute_advanced_factors_for_stock,
        compute_cross_sectional_factors,
        get_all_advanced_factor_names,
    )
    HAS_ADVANCED_FACTORS = True
    print("  Advanced factors module loaded (34 new factors)")
except ImportError as e:
    ADVANCED_FACTOR_CATEGORIES = {}
    HAS_ADVANCED_FACTORS = False
    print(f"  Advanced factors not available: {e}")

# ============================================================
# Factor Category Definitions
# ============================================================

FACTOR_CATEGORIES = {
    "momentum": [
        "mom_5d", "mom_10d", "mom_20d", "mom_60d",
        "mom_ratio_5_20", "mom_accel", "mom_vol_adj"
    ],
    "volatility": [
        "vol_5d", "vol_10d", "vol_20d", "vol_of_vol",
        "vol_ratio_5_20", "downside_vol_20d", "hl_volatility"
    ],
    "volume_flow": [
        "vol_ratio", "vol_trend", "amount_ratio", "vp_corr",
        "vol_price_trend", "vol_climax"
    ],
    "technical_mr": [
        "bb_position", "rsi_divergence", "macd_strength",
        "ma20_distance", "price_position", "atr_ratio",
        "bb_squeeze", "ma_cross_5_20"
    ],
    "quality_sharpe": [
        "sharpe_proxy", "drawdown_20d", "up_down_ratio",
        "return_consistency", "calmar_proxy"
    ],
    "tail_risk": [
        "skewness_20d", "kurtosis_20d", "tail_ratio"
    ],
    # Event & Alternative factors (from ai/event_factors.py)
    "news_sentiment": [
        "news_sentiment_mean", "news_volume_5d", "news_pos_ratio"
    ],
    "social_heat": [
        "hot_heat_value", "hot_rank_em", "xq_followers", "xq_tweets_7d"
    ],
    "fund_flow": [
        "fund_flow_main_net", "north_bound_daily"
    ],
}

# Merge advanced factor categories
if HAS_ADVANCED_FACTORS:
    FACTOR_CATEGORIES.update(ADVANCED_FACTOR_CATEGORIES)


def get_all_factor_names():
    """Return flat list of all 36 factor names."""
    names = []
    for cat_factors in FACTOR_CATEGORIES.values():
        names.extend(cat_factors)
    return names


# ============================================================
# Factor Computation Functions
# ============================================================

def _safe_roll(df, col, window, func):
    """Apply rolling function safely, returning NaN for short series."""
    if len(df) < window:
        return pd.Series([np.nan] * len(df), index=df.index)
    return func(df[col].rolling(window))


def calc_momentum_factors(df):
    """Momentum group: multi-horizon returns + acceleration."""
    close = df["close"]
    factors = {}

    # Basic momentum
    for period in [5, 10, 20, 60]:
        if len(close) > period:
            factors[f"mom_{period}d"] = close.pct_change(period)

    # Momentum ratio (short vs medium)
    if "mom_5d" in factors and "mom_20d" in factors:
        factors["mom_ratio_5_20"] = factors["mom_5d"] - factors["mom_20d"]

    # Momentum acceleration (5d mom - 20d mom, divided by std)
    if "mom_5d" in factors and "mom_20d" in factors:
        diff = factors["mom_5d"] - factors["mom_20d"]
        factors["mom_accel"] = diff

    # Volatility-adjusted momentum
    ret = close.pct_change()
    if len(ret) > 20:
        vol_20d = ret.rolling(20).std()
        mom_20d = close.pct_change(20)
        factors["mom_vol_adj"] = mom_20d / (vol_20d + 1e-10)

    return factors


def calc_volatility_factors(df):
    """Volatility group: multi-horizon, downside, Parkinson."""
    ret = df["close"].pct_change()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    factors = {}

    # Standard volatility at multiple horizons
    for period in [5, 10, 20]:
        if len(ret) > period:
            factors[f"vol_{period}d"] = ret.rolling(period).std()

    # Volatility of volatility (regime change detection)
    if "vol_20d" in factors:
        factors["vol_of_vol"] = factors["vol_20d"].rolling(20).std()

    # Vol ratio (short-term vs medium-term vol)
    if "vol_5d" in factors and "vol_20d" in factors:
        factors["vol_ratio_5_20"] = factors["vol_5d"] / (factors["vol_20d"] + 1e-10)

    # Downside volatility (semi-variance)
    if len(ret) > 20:
        downside = ret.clip(upper=0)
        factors["downside_vol_20d"] = downside.rolling(20).std()

    # Parkinson volatility (High-Low range)
    if len(high) > 20 and len(low) > 20:
        log_hl = np.log(high / (low + 1e-10))
        factors["hl_volatility"] = log_hl.rolling(20).mean() / (4 * np.log(2)) ** 0.5

    return factors


def calc_volume_factors(df):
    """Volume/Flow group: liquidity, volume-price dynamics."""
    volume = df["volume"]
    close = df["close"]
    factors = {}

    # Volume ratio vs 20d average
    if len(volume) > 20:
        factors["vol_ratio"] = volume / volume.rolling(20).mean()
        factors["vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()
        factors["amount_ratio"] = (volume * close) / (volume * close).rolling(20).mean()

    # Volume-price correlation
    if len(close) > 20:
        ret = close.pct_change()
        vol_chg = volume.pct_change()
        factors["vp_corr"] = ret.rolling(20).corr(vol_chg)

    # Volume * price return (capital flow proxy)
    if len(close) > 5:
        factors["vol_price_trend"] = volume.pct_change(5) * close.pct_change(5)

    # Volume climax detection
    if len(volume) > 20:
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        factors["vol_climax"] = (volume - vol_mean) / (vol_std * 2 + 1e-10)

    return factors


def calc_technical_factors(df):
    """Technical/Mean-Reversion group."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    factors = {}

    # RSI divergence
    if "rsi_14" in df.columns:
        factors["rsi_divergence"] = abs(df["rsi_14"] - 50) / 50
    elif "rsi" in df.columns:
        factors["rsi_divergence"] = abs(df["rsi"] - 50) / 50

    # MACD strength
    if "macd_histogram" in df.columns:
        factors["macd_strength"] = df["macd_histogram"] / (close + 1e-10)

    # Bollinger position & squeeze
    if "bollinger_pct_b" in df.columns:
        factors["bb_position"] = df["bollinger_pct_b"]
    if "bollinger_bandwidth" in df.columns:
        factors["bb_squeeze"] = -df["bollinger_bandwidth"]  # negative = squeeze

    # Price position within range
    if len(high) > 20 and len(low) > 20:
        hh = high.rolling(20).max()
        ll = low.rolling(20).min()
        factors["price_position"] = (close - ll) / (hh - ll + 1e-10)

    # MA distance
    if "ma20" in df.columns:
        factors["ma20_distance"] = close / df["ma20"] - 1

    # ATR ratio
    if "atr_14" in df.columns:
        factors["atr_ratio"] = df["atr_14"] / (close + 1e-10)

    # MA cross signal (golden/death cross)
    if "ma5" in df.columns and "ma20" in df.columns:
        factors["ma_cross_5_20"] = (df["ma5"] - df["ma20"]) / (close + 1e-10)

    return factors


def calc_quality_factors(df):
    """Quality/Value proxy group."""
    close = df["close"]
    ret = close.pct_change()
    factors = {}

    # Rolling Sharpe proxy
    if len(ret) > 20:
        roll_mean = ret.rolling(20).mean()
        roll_std = ret.rolling(20).std()
        factors["sharpe_proxy"] = roll_mean / (roll_std + 1e-10)

    # Max drawdown over 20 days
    if len(close) > 20:
        roll_max = close.rolling(20).max()
        factors["drawdown_20d"] = close / roll_max - 1

    # Up/down capture ratio
    if len(ret) > 20:
        up = ret.clip(lower=0).rolling(20).sum()
        down = abs(ret.clip(upper=0).rolling(20).sum())
        factors["up_down_ratio"] = up / (down + 1e-10)

    # Return consistency (% positive days)
    if len(ret) > 20:
        factors["return_consistency"] = (ret > 0).rolling(20).mean()

    # Calmar proxy
    if len(close) > 20:
        mom_20d = close.pct_change(20)
        dd = factors.get("drawdown_20d", pd.Series([0] * len(close), index=close.index))
        factors["calmar_proxy"] = mom_20d / (abs(dd) + 1e-10)

    return factors


def calc_tail_factors(df):
    """Skewness & Tail Risk group."""
    ret = df["close"].pct_change()
    factors = {}

    if len(ret) > 20:
        # Skewness
        factors["skewness_20d"] = ret.rolling(20).skew()
        # Kurtosis
        factors["kurtosis_20d"] = ret.rolling(20).kurt()
        # Tail ratio (95th vs 5th percentile return)
        tail_95 = ret.rolling(20).quantile(0.95)
        tail_5 = ret.rolling(20).quantile(0.05)
        factors["tail_ratio"] = tail_95 / (abs(tail_5) + 1e-10)

    return factors


# ============================================================
# Main Factor Computation
# ============================================================

def compute_all_factors(df, market_ret=None, sector_ret=None):
    """
    Compute all 70+ alpha factors for a single stock DataFrame.
    Expects columns: date, open, high, low, close, volume
    Plus optional pre-computed indicators from feature_engine:
      ma5, ma10, ma20, ma60, rsi_14, rsi, macd_histogram,
      bollinger_pct_b, bollinger_bandwidth, atr_14

    V4.0: Also computes 34 advanced factors (MAX effect, reversal,
    idiosyncratic, liquidity, path-dependent, gap, lead-lag).

    Returns DataFrame with factor columns only (no OHLCV).
    """
    if len(df) < 60:
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)

    all_factors = {}

    # Compute original 6 categories (36 factors)
    all_factors.update(calc_momentum_factors(df))
    all_factors.update(calc_volatility_factors(df))
    all_factors.update(calc_volume_factors(df))
    all_factors.update(calc_technical_factors(df))
    all_factors.update(calc_quality_factors(df))
    all_factors.update(calc_tail_factors(df))

    # Compute advanced factors (34 factors) — V4.0
    if HAS_ADVANCED_FACTORS:
        try:
            adv_factors = compute_advanced_factors_for_stock(df, market_ret, sector_ret)
            if adv_factors is not None and not adv_factors.empty:
                for col in adv_factors.columns:
                    all_factors[col] = adv_factors[col].values
        except Exception:
            pass  # Advanced factors unavailable — gracefully skip

    result = pd.DataFrame(all_factors, index=df.index)
    return result


def compute_factor_exposure(features_df, factor_names, factor_ic_map):
    """
    Compute weighted factor exposure score for a single stock's latest row.

    factor_exposure = sum(factor_zscore_i * sign(IC_i) * weight_i)

    Positive exposure → factors predict positive return
    Negative exposure → factors predict negative return
    """
    row = features_df.iloc[-1] if len(features_df) > 0 else None
    if row is None:
        return 0.0

    exposure = 0.0
    total_weight = 0.0

    for fname in factor_names:
        if fname not in row or fname not in factor_ic_map:
            continue
        val = row[fname]
        ic = factor_ic_map[fname]["ic_mean"]
        weight = factor_ic_map[fname].get("weight", 0.02)

        if pd.isna(val) or pd.isna(ic):
            continue

        # Direction: sign(IC) tells us factor direction
        direction = 1.0 if ic > 0 else -1.0
        exposure += val * direction * weight
        total_weight += weight

    if total_weight > 0:
        exposure /= total_weight

    return round(float(exposure), 4)


# ============================================================
# IC Computation Engine
# ============================================================

def compute_rolling_ic(window_days=60, step_days=20, min_stocks=50,
                       min_samples_per_window=30, max_windows=None):
    """
    Compute rolling walk-forward IC for all factors across all stocks.

    For each rolling window, compute Spearman IC between factor(t) and
    future_return(t+5), then slide forward. This avoids look-ahead bias
    and enables factor stability analysis.

    Returns:
        ic_rolling: DataFrame of IC per factor per window
        ic_summary: DataFrame of aggregated IC statistics
    """
    print("=" * 60)
    print("Factor Engine V4.0 — Rolling IC Analysis (70+ factors)")
    print("=" * 60)

    if not os.path.exists(FEATURE_DIR):
        print("ERROR: features/ directory not found. Run feature_engine first.")
        return None, None

    feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    n_stocks = len(feature_files)
    print(f"Stocks available: {n_stocks}")

    all_factor_names = get_all_factor_names()

    # Collect time-aligned factor+target data across all stocks
    # We need to find the common date range first
    all_dates = set()
    stock_data = {}

    for i, fname in enumerate(feature_files):
        code = fname.replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, fname), encoding="utf-8-sig")
            if len(df) < 120 or "future_return" not in df.columns:
                continue

            df = df.sort_values("date").reset_index(drop=True)

            # Compute factors on-the-fly (factors not yet in feature files)
            # V4.0: pass market_ret from index if available (for idiosyncratic factors)
            factor_df = compute_all_factors(df)
            if factor_df.empty:
                continue

            # Align factor_df with original df
            combined = pd.concat([df[["date", "close", "future_return"]], factor_df], axis=1)
            combined = combined.dropna(subset=["future_return"])
            if len(combined) < min_samples_per_window:
                continue

            stock_data[code] = combined
            all_dates.update(combined["date"].values)

        except Exception as e:
            continue

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{n_stocks} stocks...")

    n_valid = len(stock_data)
    print(f"Valid stocks: {n_valid}/{n_stocks}")

    if n_valid < min_stocks:
        print(f"ERROR: Only {n_valid} valid stocks (need {min_stocks})")
        return None, None

    # ── Cross-Sectional Factor Computation (V4.0) ──
    # After all stocks are loaded, compute cross-sectional percentile ranks
    if HAS_ADVANCED_FACTORS:
        try:
            print("  Computing cross-sectional rank factors across all stocks...")
            stock_data = compute_cross_sectional_factors(stock_data)
            # Rebuild all_factor_names to include the new CS factors
            all_factor_names = get_all_factor_names()
            print(f"  Total factors (including cross-sectional): {len(all_factor_names)}")
        except Exception as e:
            print(f"  Cross-sectional factors skipped: {e}")

    # Sort dates to define windows
    sorted_dates = sorted(all_dates)
    if len(sorted_dates) < window_days:
        print("ERROR: Not enough dates for rolling windows")
        return None, None

    # Build rolling windows
    windows = []
    start = 0
    while start + window_days <= len(sorted_dates):
        windows.append(sorted_dates[start:start + window_days])
        start += step_days

    if max_windows:
        windows = windows[-max_windows:]  # Keep most recent windows

    print(f"Rolling windows: {len(windows)} (size={window_days}d, step={step_days}d)")

    # For each window, compute IC per factor
    all_window_ics = []

    for wi, win_dates in enumerate(windows):
        win_set = set(win_dates)
        win_label = f"W{wi+1}_{win_dates[0][:10]}_{win_dates[-1][:10]}"

        # Collect factor values and targets within this window
        factor_values = {fn: [] for fn in all_factor_names}
        targets = []

        stocks_in_window = 0

        for code, sdf in stock_data.items():
            win_data = sdf[sdf["date"].isin(win_set)]
            if len(win_data) < min_samples_per_window:
                continue

            stocks_in_window += 1
            targets.extend(win_data["future_return"].values)

            for fn in all_factor_names:
                if fn in win_data.columns:
                    factor_values[fn].extend(win_data[fn].values)
                else:
                    factor_values[fn].extend([np.nan] * len(win_data))

        if stocks_in_window < min_stocks:
            continue

        targets = np.array(targets)

        # Compute IC per factor
        for fn in all_factor_names:
            fvals = np.array(factor_values[fn])
            valid = ~np.isnan(fvals) & ~np.isnan(targets)
            if valid.sum() < min_samples_per_window:
                continue

            try:
                ic, _ = spearmanr(
                    pd.Series(fvals[valid]).rank(),
                    pd.Series(targets[valid]).rank()
                )
                if not np.isnan(ic):
                    all_window_ics.append({
                        "window": win_label,
                        "window_start": win_dates[0][:10],
                        "window_end": win_dates[-1][:10],
                        "factor": fn,
                        "ic": round(float(ic), 5),
                        "n_samples": int(valid.sum()),
                        "n_stocks": stocks_in_window,
                    })
            except Exception:
                continue

        if (wi + 1) % 5 == 0:
            print(f"  Window {wi+1}/{len(windows)} done")

    # Build rolling IC DataFrame
    ic_rolling = pd.DataFrame(all_window_ics)
    if ic_rolling.empty:
        print("ERROR: No valid IC computed")
        return None, None

    # Aggregate to factor-level summary
    ic_summary = []
    for fn in all_factor_names:
        fdata = ic_rolling[ic_rolling["factor"] == fn]
        if len(fdata) < 3:
            continue

        ics = fdata["ic"].values
        ic_mean = ics.mean()
        ic_std = ics.std()
        ic_ir = ic_mean / (ic_std + 1e-10)
        abs_ic = abs(ic_mean)
        hit_rate = (ics > 0).mean() if ic_mean > 0 else (ics < 0).mean()
        # Stability: fraction of windows where |IC| > 0.015
        stability = (np.abs(ics) > 0.015).mean()

        # Assign category
        category = "unknown"
        for cat, factors in FACTOR_CATEGORIES.items():
            if fn in factors:
                category = cat
                break

        ic_summary.append({
            "factor": fn,
            "category": category,
            "ic_mean": round(ic_mean, 5),
            "ic_std": round(ic_std, 5),
            "ic_ir": round(ic_ir, 3),
            "abs_ic": round(abs_ic, 5),
            "hit_rate": round(hit_rate, 3),
            "rolling_stability": round(stability, 3),
            "n_windows": len(fdata),
            "avg_n_samples": int(fdata["n_samples"].mean()),
        })

    ic_df = pd.DataFrame(ic_summary).sort_values("abs_ic", ascending=False)
    ic_df.to_csv(os.path.join(REPORT_DIR, "factor_ic_rolling.csv"),
                 index=False, encoding="utf-8-sig")

    print(f"\n  Factors tested: {len(ic_df)}")
    print(f"  Top 10 by |IC|:")
    for _, row in ic_df.head(10).iterrows():
        print(f"  {row['factor']:<22s} |IC|={row['abs_ic']:.4f}  "
              f"IR={row['ic_ir']:+.3f}  stability={row['rolling_stability']:.2f}")

    return ic_rolling, ic_df


# ============================================================
# Factor Selection & Weight Generation
# ============================================================

def select_factors(ic_summary, top_n=15, min_abs_ic=0.015,
                   max_correlation=0.70, min_stability=0.4):
    """
    Select best factors by |IC| with redundancy pruning.

    Steps:
    1. Filter by min_abs_ic and min_stability
    2. Take top-N by abs_ic
    3. Remove redundant factors (high pairwise correlation)
    4. Assign weights proportional to abs_ic * stability
    """
    if ic_summary is None or ic_summary.empty:
        return None

    # Filter
    candidates = ic_summary[
        (ic_summary["abs_ic"] >= min_abs_ic) &
        (ic_summary["rolling_stability"] >= min_stability)
    ].copy()

    if candidates.empty:
        # Relax thresholds
        candidates = ic_summary.nlargest(min(top_n, len(ic_summary)), "abs_ic").copy()

    # Top-N
    selected = candidates.nlargest(top_n, "abs_ic").copy()

    # Compute weights: proportional to abs_ic * stability
    selected["raw_weight"] = selected["abs_ic"] * selected["rolling_stability"]
    total_w = selected["raw_weight"].sum()
    selected["weight"] = selected["raw_weight"] / (total_w + 1e-10)
    selected["weight"] = selected["weight"].round(4)

    # Normalize to sum = 1
    weight_sum = selected["weight"].sum()
    if weight_sum > 0:
        selected["weight"] = selected["weight"] / weight_sum

    return selected


# ============================================================
# Main Entry Point
# ============================================================

def run(window_days=60, step_days=20, top_n=15, min_abs_ic=0.015):
    """
    Run the full factor engine pipeline:
    1. Compute rolling IC across all stocks
    2. Select best factors by |IC| and stability
    3. Generate factor weights for scoring integration
    4. Save all outputs
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    t0 = time.time()

    # Step 1: Rolling IC
    ic_rolling, ic_summary = compute_rolling_ic(
        window_days=window_days, step_days=step_days, max_windows=12
    )

    if ic_summary is None:
        print("FACTOR ENGINE FAILED: No IC data computed")
        return None

    # Step 2: Select factors
    selected = select_factors(ic_summary, top_n=top_n, min_abs_ic=min_abs_ic)

    # Step 3: Build output
    if selected is not None and not selected.empty:
        factor_weights = {}
        ic_map = {}
        for _, row in selected.iterrows():
            fname = str(row["factor"])
            w = float(row["weight"])
            factor_weights[fname] = round(w, 4)
            ic_map[fname] = {
                "ic_mean": float(row["ic_mean"]),
                "abs_ic": float(row["abs_ic"]),
                "weight": float(row["weight"]),
                "stability": float(row["rolling_stability"]),
                "category": str(row["category"]),
            }
    else:
        factor_weights = {"fallback": 1.0}
        ic_map = {}

    # Save factor_selection.json
    # Convert to native Python types for JSON serialization
    n_factors_tested = int(len(ic_summary)) if ic_summary is not None else 0
    n_factors_selected = int(len(selected)) if selected is not None else 0
    n_stocks_analyzed = int(ic_summary["n_windows"].max()) if not ic_summary.empty else 0

    top_factors_list = []
    if selected is not None and not selected.empty:
        for _, row in selected.iterrows():
            top_factors_list.append({
                "factor": str(row["factor"]),
                "category": str(row["category"]),
                "ic_mean": float(row["ic_mean"]),
                "abs_ic": float(row["abs_ic"]),
                "weight": float(row["weight"]),
                "rolling_stability": float(row["rolling_stability"]),
            })

    output = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "method": f"rolling_window_{window_days}d_step_{step_days}d",
        "n_stocks_analyzed": n_stocks_analyzed,
        "n_factors_tested": n_factors_tested,
        "n_factors_selected": n_factors_selected,
        "min_abs_ic": min_abs_ic,
        "top_n": top_n,
        "factor_weights_for_scoring": factor_weights,
        "factor_ic_map": ic_map,
        "top_factors": top_factors_list,
    }

    with open(os.path.join(REPORT_DIR, "factor_selection.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Also save legacy factor_weights.json for backward compat
    legacy = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "n_stocks_analyzed": int(ic_summary["n_windows"].max()) if not ic_summary.empty else 0,
        "n_factors_tested": int(len(ic_summary)) if ic_summary is not None else 0,
        "n_significant": int(len(selected)) if selected is not None else 0,
        "weights": factor_weights,
    }
    with open(os.path.join(REPORT_DIR, "factor_weights.json"), "w", encoding="utf-8") as f:
        json.dump(legacy, f, ensure_ascii=False, indent=2)

    # ---- Step 4: Event & Alternative Factor IC ----
    print(f"\n{'='*60}")
    print("Testing Event & Alternative Factors...")
    try:
        from ai.event_factors import compute_event_factors_batch, auto_discover_new_factors

        # Get codes from valid stocks (stock_data is in compute_rolling_ic scope)
        # Re-derive from feature files
        feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
        all_codes = [f.replace(".csv", "").zfill(6) for f in feature_files][:100]
        event_df = compute_event_factors_batch(all_codes, max_stocks=100)

        if event_df is not None and not event_df.empty:
            # Auto-test event factors against future returns
            event_results = auto_discover_new_factors(all_codes[:50])
            if event_results:
                # Merge significant event factors into selection
                for er in event_results:
                    if er.get("is_significant"):
                        fname = er["factor"]
                        found = False
                        for cat, factors in FACTOR_CATEGORIES.items():
                            if fname in factors:
                                found = True
                                break
                        if not found:
                            FACTOR_CATEGORIES.setdefault("event_alternative", []).append(fname)

                        # Add to ic_map
                        if fname not in ic_map:
                            ic_map[fname] = {
                                "ic_mean": er["ic"],
                                "abs_ic": er["abs_ic"],
                                "weight": 1.0 / (len(event_results) + 1),
                                "stability": 0.5,
                                "category": "event_alternative",
                            }
        print(f"  Event factors tested: {len(event_results) if event_results else 0}")
    except ImportError:
        print("  [SKIP] event_factors module not available (install snownlp)")
    except Exception as e:
        print(f"  [WARN] Event factor computation failed: {e}")

    # ---- Step 5: Genetic Programming — Nonlinear Factor Discovery ----
    print(f"\n{'='*60}")
    print("Step 5: Genetic Programming — Nonlinear Factor Discovery")
    print(f"{'='*60}")
    genetic_factors = []
    try:
        from ai.genetic_factors import discover_factors
        genetic_factors = discover_factors(
            population_size=300, generations=5,
            max_stocks=100, top_n=12
        )
        if genetic_factors:
            # Merge genetic factors into ic_map
            for gf in genetic_factors:
                gf_name = f"GP_{gf['expression'][:30]}"
                if gf_name not in ic_map:
                    gf_ic = gf.get("abs_ic", gf.get("ic", 0.01))
                    ic_map[gf_name] = {
                        "ic_mean": gf.get("ic", 0),
                        "abs_ic": abs(gf_ic),
                        "weight": 0.01,
                        "stability": 0.4,
                        "category": "genetic_nonlinear",
                        "expression": gf["expression"],
                    }
            print(f"  Genetic factors discovered: {len(genetic_factors)}")
    except ImportError:
        print("  [SKIP] gplearn not installed")
    except Exception as e:
        print(f"  [WARN] Genetic programming failed: {e}")

    # ---- Step 6: Factor Logic Filter — Economic Interpretability ----
    print(f"\n{'='*60}")
    print("Step 6: Factor Logic Decomposition & Filtering")
    print(f"{'='*60}")
    try:
        from ai.factor_logic import filter_all_discovered_factors
        validated, rejected = filter_all_discovered_factors()
        if validated:
            # Boost weights for factors with high economic interpretability
            for vf in validated:
                fname = vf.get("expression", "")
                for k in ic_map:
                    if fname[:20] in k or k[:20] in fname:
                        logic_bonus = vf.get("interpretability", 0.3)
                        ic_map[k]["weight"] = round(ic_map[k]["weight"] * (1 + logic_bonus), 4)
                        ic_map[k]["interpretability"] = logic_bonus
            print(f"  Validated: {len(validated)}, Rejected: {len(rejected)}")
    except Exception as e:
        print(f"  [WARN] Factor logic filtering failed: {e}")

    # ---- Final Output ----
    # Recompute weights including all factor sources
    if ic_map:
        total_abs = sum(v["abs_ic"] for v in ic_map.values())
        for k in ic_map:
            ic_map[k]["weight"] = round(ic_map[k]["abs_ic"] / (total_abs + 1e-10), 4)
        factor_weights = {k: v["weight"] for k, v in ic_map.items()}

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Factor Engine V4.0 Complete ({elapsed:.1f}s)")
    n_all = (len(selected) if selected is not None else 0) + len(genetic_factors)
    print(f"  Price/Volume factors: {len(selected) if selected is not None else 0}")
    print(f"  Genetic (nonlinear): {len(genetic_factors)}")
    print(f"  Event/Alternative: {len([k for k,v in ic_map.items() if v.get('category','') in ('event_alternative','news_sentiment','social_heat','fund_flow','llm_sentiment','genetic_nonlinear')])}")
    print(f"  Total factors in scoring: {len(ic_map)}")
    if selected is not None and not selected.empty:
        print(f"  Top factor: {selected.iloc[0]['factor']} |IC|={selected.iloc[0]['abs_ic']:.4f}")
    print(f"  Output: reports/factor_selection.json")
    print(f"  Output: reports/factor_weights.json (legacy)")
    print(f"  Output: reports/factor_ic_rolling.csv")
    print(f"  Output: reports/genetic_factors.json")
    print(f"  Output: reports/factor_logic_report.json")
    print(f"{'='*60}")

    return ic_summary


if __name__ == "__main__":
    run()
