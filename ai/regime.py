"""
A-Insight Pro
Market Regime Detection V1.0

Classifies the current market into one of 6 regimes:
  - BULL_TRENDING:  Rising market with low volatility
  - BULL_VOLATILE:  Rising market with high volatility
  - BEAR_TRENDING:  Falling market with low volatility
  - BEAR_VOLATILE:  Falling market with high volatility
  - SIDEWAYS:       Flat market, low volatility
  - CHOPPY:         Flat market, high volatility (whipsaw risk)

Why this matters:
  - Different alpha factors work in different regimes
  - Momentum works in trending markets, mean-reversion in sideways
  - High vol regimes need tighter risk controls
  - Regime-aware models can switch strategies
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

FEATURE_DIR = "features"
REPORT_DIR = "reports"
OUTPUT_FILE = os.path.join(REPORT_DIR, "market_regime.csv")


def detect_regime(lookback_days=60):
    """
    Detect the current market regime from aggregate stock data.

    Uses two dimensions:
    1. Trend: % change of equal-weighted index over lookback
    2. Volatility: average daily volatility across stocks

    Returns dict with regime classification and metrics.
    """
    if not os.path.exists(FEATURE_DIR):
        return {"regime": "UNKNOWN", "confidence": 0.0}

    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    if len(files) < 50:
        return {"regime": "UNKNOWN", "confidence": 0.0}

    all_returns = []
    all_volatilities = []

    for fname in files:
        code = fname.replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, fname), encoding="utf-8-sig")
            if len(df) < lookback_days:
                continue

            recent = df.tail(lookback_days)
            if "close" in recent.columns:
                # 20-day return (trend proxy)
                if len(recent) >= 20:
                    ret_20d = recent["close"].iloc[-1] / recent["close"].iloc[-20] - 1
                    all_returns.append(ret_20d)

                # 20-day daily return volatility
                if "return_1d" in recent.columns:
                    daily_vol = recent["return_1d"].tail(20).std()
                elif "return" in recent.columns:
                    daily_vol = recent["return"].tail(20).std()
                else:
                    daily_ret = recent["close"].pct_change()
                    daily_vol = daily_ret.tail(20).std()

                all_volatilities.append(daily_vol)

        except Exception:
            continue

    if len(all_returns) < 50:
        return {"regime": "UNKNOWN", "confidence": 0.0}

    # Aggregate metrics
    avg_return = np.median(all_returns)  # Median return across stocks
    avg_vol = np.median(all_volatilities)

    # Regime thresholds
    TREND_UP = 0.03       # 3% over 20 days = uptrend
    TREND_DOWN = -0.03    # -3% = downtrend
    VOL_HIGH = 0.025      # 2.5% daily vol = high vol

    # Classify
    if avg_return > TREND_UP:
        if avg_vol < VOL_HIGH:
            regime = "BULL_TRENDING"
        else:
            regime = "BULL_VOLATILE"
    elif avg_return < TREND_DOWN:
        if avg_vol < VOL_HIGH:
            regime = "BEAR_TRENDING"
        else:
            regime = "BEAR_VOLATILE"
    else:
        if avg_vol < VOL_HIGH:
            regime = "SIDEWAYS"
        else:
            regime = "CHOPPY"

    # Confidence: how far from boundaries
    trend_strength = abs(avg_return) / max(abs(TREND_UP), 0.001)
    vol_strength = avg_vol / max(VOL_HIGH, 0.001)
    confidence = min(1.0, max(0.3, (trend_strength + vol_strength) / 4))

    # Regime-specific recommendations
    recommendations = {
        "BULL_TRENDING": {
            "strategy": "趋势跟随 (Trend Following)",
            "factor_weight": "momentum:1.5, quality:1.0, mean_reversion:0.5",
            "position_pct": 70,
            "stop_loss_pct": -8,
        },
        "BULL_VOLATILE": {
            "strategy": "回调买入 (Buy Dips)",
            "factor_weight": "momentum:0.8, volatility:1.2, quality:1.0",
            "position_pct": 50,
            "stop_loss_pct": -5,
        },
        "BEAR_TRENDING": {
            "strategy": "防御为主 (Defensive)",
            "factor_weight": "quality:1.5, tail_risk:1.5, momentum:0.3",
            "position_pct": 20,
            "stop_loss_pct": -3,
        },
        "BEAR_VOLATILE": {
            "strategy": "现金为王 (Cash is King)",
            "factor_weight": "tail_risk:2.0, quality:1.0, volatility:1.0",
            "position_pct": 10,
            "stop_loss_pct": -2,
        },
        "SIDEWAYS": {
            "strategy": "均值回归 (Mean Reversion)",
            "factor_weight": "mean_reversion:1.5, technical_mr:1.5, momentum:0.5",
            "position_pct": 40,
            "stop_loss_pct": -5,
        },
        "CHOPPY": {
            "strategy": "轻仓短线 (Light & Short)",
            "factor_weight": "volatility:1.5, quality:1.2, tail_risk:1.0",
            "position_pct": 25,
            "stop_loss_pct": -3,
        },
    }

    rec = recommendations.get(regime, recommendations["SIDEWAYS"])

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "regime": regime,
        "confidence": round(confidence, 3),
        "avg_20d_return": round(float(avg_return), 4),
        "avg_daily_volatility": round(float(avg_vol), 4),
        "n_stocks_analyzed": len(all_returns),
        "strategy": rec["strategy"],
        "factor_weight_hint": rec["factor_weight"],
        "suggested_position_pct": rec["position_pct"],
        "suggested_stop_loss_pct": rec["stop_loss_pct"],
    }

    # Save
    os.makedirs(REPORT_DIR, exist_ok=True)
    pd.DataFrame([result]).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    return result


def print_regime():
    """Print regime analysis to console."""
    result = detect_regime()
    if result.get("regime") == "UNKNOWN":
        print("Unable to detect market regime (insufficient data)")
        return result

    print("=" * 50)
    print(f"Market Regime: {result['regime']}")
    print(f"  Confidence:    {result['confidence']:.1%}")
    print(f"  20d Return:    {result['avg_20d_return']:+.2%}")
    print(f"  Daily Vol:     {result['avg_daily_volatility']:.2%}")
    print(f"  Strategy:      {result['strategy']}")
    print(f"  Position:      {result['suggested_position_pct']}%")
    print(f"  Stop Loss:     {result['suggested_stop_loss_pct']}%")
    print(f"  Factor Hint:   {result['factor_weight_hint']}")
    print(f"  Stocks:        {result['n_stocks_analyzed']}")
    print("=" * 50)

    return result


if __name__ == "__main__":
    print_regime()
