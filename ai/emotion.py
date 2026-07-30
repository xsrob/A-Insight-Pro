"""
A-Insight Pro
Market Emotion + Smart Money Flow Model V4.0

Core principle: markets move on two forces —
  1. Sentiment (市场热度): how bullish/bearish the crowd is
  2. Smart Money (主力动向): what institutional players are actually doing

Dimensions (10 total):
  ── Market Heat (6 dims, 50% weight) ──
  1. Breadth           - % stocks above MA20                        (15%)
  2. Strength Depth    - % stocks with >3% gain                     (8%)
  3. Volume Momentum   - Aggregate volume vs 20-day avg             (8%)
  4. Fear Index        - Average volatility (inverted)              (8%)
  5. Trend Alignment   - % stocks MA5 > MA20                        (6%)
  6. Extreme Spread    - New highs to new lows ratio                 (5%)

  ── Smart Money (4 dims, 50% weight) ──
  7. Abnormal Volume   - % stocks with vol > 2x avg (主力活动的痕迹)  (15%)
  8. Accum Signal      - Stealth accumulation vs distribution        (15%)
  9. Inst Momentum     - Consecutive high-volume directional moves   (10%)
  10. Smart Flow Dir   - Volume-weighted net capital flow            (10%)

Output:
  - Composite score (0-100)
  - Smart Money Signal: 主力吸筹 / 主力出货 / 主力观望 / 分歧
  - Institutional Activity: HIGH / MEDIUM / LOW
  - Per-dimension diagnostics
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

FEATURE_DIR = "features"
DATA_DIR = "data"
REPORT_DIR = "reports"
OUTPUT_FILE = os.path.join(REPORT_DIR, "market_emotion.csv")
os.makedirs(REPORT_DIR, exist_ok=True)

# ─── Weights: 50% Market Heat + 50% Smart Money ───
WEIGHTS = {
    # Market Heat
    "breadth":        0.15,
    "strength":       0.08,
    "volume":         0.08,
    "fear":           0.08,
    "trend":          0.06,
    "spread":         0.05,
    # Smart Money
    "abnormal_vol":   0.15,
    "accumulation":   0.15,
    "inst_momentum":  0.10,
    "smart_flow":     0.10,
}


def load_feature_snapshot():
    """Load the latest row from each stock feature file."""
    records = []
    if not os.path.exists(FEATURE_DIR):
        return pd.DataFrame()

    for f in os.listdir(FEATURE_DIR):
        if not f.endswith(".csv"):
            continue
        code = f.replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, f), encoding="utf-8-sig")
            if len(df) < 20:
                continue
            row = df.iloc[-1].to_dict()
            row["code"] = code
            records.append(row)
        except Exception:
            continue

    return pd.DataFrame(records)


def load_recent_price_data(lookback=5):
    """
    Load last `lookback` rows per stock from data/ for smart money analysis.
    Returns a dict: {code: DataFrame with last N rows}
    """
    stock_data = {}
    if not os.path.exists(DATA_DIR):
        return stock_data

    for f in os.listdir(DATA_DIR):
        if not f.endswith(".csv"):
            continue
        code = f.replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f), encoding="utf-8-sig")
            if len(df) < lookback + 2:
                continue
            # Keep only needed columns
            cols = ["date", "open", "high", "low", "close", "volume"]
            available = [c for c in cols if c in df.columns]
            recent = df[available].tail(lookback).copy()
            if len(recent) < lookback:
                continue

            # Compute daily return
            if "close" in recent.columns:
                recent["return_1d"] = recent["close"].pct_change()

            # Compute volume ratio vs trailing average
            if "volume" in recent.columns:
                vol_mean = recent["volume"].iloc[:-1].mean()
                if vol_mean > 0:
                    recent["vol_ratio"] = recent["volume"] / vol_mean
                else:
                    recent["vol_ratio"] = 1.0

            # Position within day's range (close position = (close-low)/(high-low))
            if all(c in recent.columns for c in ["high", "low", "close"]):
                denom = recent["high"] - recent["low"]
                recent["close_position"] = np.where(
                    denom > 0,
                    (recent["close"] - recent["low"]) / denom,
                    0.5
                )
            else:
                recent["close_position"] = 0.5

            stock_data[code] = recent
        except Exception:
            continue

    return stock_data


# ═══════════════════════════════════════════
# Market Heat Dimensions (1-6)
# ═══════════════════════════════════════════

def compute_breadth(snap):
    """% stocks above MA20."""
    total = len(snap)
    if "ma20" in snap.columns and "close" in snap.columns:
        above = (snap["close"] > snap["ma20"]).sum()
        return (above / total) * 100, int(above)
    elif "ma20_ratio" in snap.columns:
        above = (snap["ma20_ratio"] > 0).sum()
        return (above / total) * 100, int(above)
    return 50.0, 0


def compute_strength(snap):
    """% stocks with >3% daily gain."""
    total = len(snap)
    ret_col = None
    for c in ["return_1d", "return"]:
        if c in snap.columns:
            ret_col = c
            break
    if ret_col:
        strong_up = (snap[ret_col] > 0.03).sum()
        strong_down = (snap[ret_col] < -0.03).sum()
        return (strong_up / total) * 100, int(strong_up), int(strong_down)
    return 5.0, 0, 0


def compute_volume_momentum(price_data):
    """Aggregate volume vs 20-day average across all stocks."""
    ratios = []
    for code, df in price_data.items():
        if "vol_ratio" in df.columns and len(df) > 0:
            ratios.append(df["vol_ratio"].iloc[-1])
    if ratios:
        avg_ratio = np.mean(ratios)
        return min(100, max(0, 50 + (avg_ratio - 1) * 50))
    return 50.0


def compute_fear(snap):
    """Average volatility across universe (inverted)."""
    vol_col = None
    for c in ["volatility_20d", "volatility"]:
        if c in snap.columns:
            vol_col = c
            break
    if vol_col:
        avg_vol = snap[vol_col].dropna().mean()
        return max(0, min(100, 100 - avg_vol * 2000))
    return 50.0


def compute_trend(snap):
    """% stocks with MA5 > MA20."""
    total = len(snap)
    if "ma5" in snap.columns and "ma20" in snap.columns:
        aligned = (snap["ma5"] > snap["ma20"]).sum()
        return (aligned / total) * 100, int(aligned)
    return 50.0, 0


def compute_spread(price_data):
    """Ratio of stocks near 60d highs vs 60d lows."""
    near_high = 0
    near_low = 0
    for code, df in price_data.items():
        if "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
            continue
        close = df["close"].iloc[-1]
        high_60 = df["high"].max()
        low_60 = df["low"].min()
        if high_60 <= 0:
            continue
        if close >= high_60 * 0.95:
            near_high += 1
        elif close <= low_60 * 1.05:
            near_low += 1
    total = near_high + near_low
    if total > 0:
        return (near_high / total) * 100, near_high, near_low
    return 50.0, 0, 0


# ═══════════════════════════════════════════
# Smart Money Dimensions (7-10)
# ═══════════════════════════════════════════

def compute_abnormal_volume(price_data):
    """
    % of stocks with abnormally high volume (>2x average).

    Interpretation:
    - 3-8%:  Healthy, selective institutional activity
    - 8-15%: Strong smart money rotation
    - >20%:  Either sector-wide event or panic (check accumulation for context)
    - <2%:   Institutions are inactive / waiting

    Score peaks at 10-12% (active but not panicked).
    """
    total = len(price_data)
    abnormal = 0
    details = []  # top abnormal volume stocks

    for code, df in price_data.items():
        if "vol_ratio" not in df.columns:
            continue
        latest_ratio = df["vol_ratio"].iloc[-1]
        if pd.notna(latest_ratio) and latest_ratio > 2.0:
            abnormal += 1
            ret = df["return_1d"].iloc[-1] if "return_1d" in df.columns else 0
            details.append({
                "code": code,
                "vol_ratio": round(latest_ratio, 2),
                "return": round(float(ret) * 100, 2) if pd.notna(ret) else 0,
            })

    if total == 0:
        return 50.0, 0, []

    pct = (abnormal / total) * 100

    # Optimal range: 5-15% → score peaks here
    if pct < 3:
        score = pct / 3 * 40 + 10   # 0% → 10, 3% → 50
    elif pct <= 15:
        score = 50 + (pct - 3) / 12 * 40  # 3% → 50, 15% → 90
    elif pct <= 30:
        score = 90 - (pct - 15) / 15 * 40  # 15% → 90, 30% → 50
    else:
        score = max(10, 50 - (pct - 30) / 10 * 40)  # >30% → declining

    # Sort details by vol_ratio descending
    details.sort(key=lambda x: x["vol_ratio"], reverse=True)

    return round(score, 2), abnormal, details[:10]


def compute_accumulation(price_data):
    """
    Detect stealth accumulation/distribution patterns.

    Accumulation (主力吸筹):
      - Volume > 1.5x average
      - Daily return magnitude < 2% (not drawing attention)
      - Close in upper half of day's range (close_position > 0.5)
      → Institutions buying quietly without pushing price up too much

    Distribution (主力出货):
      - Volume > 1.5x average
      - Daily return magnitude < 2%
      - Close in lower half of day's range (close_position < 0.5)
      → Institutions selling into strength

    Score = accumulation / (accumulation + distribution) * 100
    """
    accumulating = 0
    distributing = 0
    acc_details = []
    dist_details = []

    for code, df in price_data.items():
        if "vol_ratio" not in df.columns or "return_1d" not in df.columns:
            continue

        latest = df.iloc[-1]
        vol_ratio = latest.get("vol_ratio", 1.0)
        ret = latest.get("return_1d", 0)
        close_pos = latest.get("close_position", 0.5)

        if pd.isna(vol_ratio) or pd.isna(ret):
            continue

        # High volume + muted price movement = stealth activity
        if vol_ratio > 1.5 and abs(ret) < 0.02:
            if close_pos > 0.55:  # Close in upper half → buying pressure
                accumulating += 1
                acc_details.append({
                    "code": code,
                    "vol_ratio": round(float(vol_ratio), 2),
                    "return": round(float(ret) * 100, 2),
                    "close_pos": round(float(close_pos), 2),
                })
            elif close_pos < 0.45:  # Close in lower half → selling pressure
                distributing += 1
                dist_details.append({
                    "code": code,
                    "vol_ratio": round(float(vol_ratio), 2),
                    "return": round(float(ret) * 100, 2),
                    "close_pos": round(float(close_pos), 2),
                })

    total_stealth = accumulating + distributing
    if total_stealth == 0:
        return 50.0, 0, 0, [], []

    # Ratio of accumulation to total stealth activity
    acc_ratio = accumulating / total_stealth

    # Scale: 0.5 (balanced) → 50, 1.0 (all accumulating) → 100, 0 (all distributing) → 0
    score = acc_ratio * 100

    acc_details.sort(key=lambda x: x["vol_ratio"], reverse=True)
    dist_details.sort(key=lambda x: x["vol_ratio"], reverse=True)

    return round(score, 2), accumulating, distributing, acc_details[:5], dist_details[:5]


def compute_institutional_momentum(price_data):
    """
    Detect consecutive high-volume directional movement.

    Pattern: Today AND yesterday BOTH had:
      - Volume > 1.3x average
      - Positive return
      - Close in upper portion of range (close_position > 0.5)

    This signals persistent institutional buying across multiple sessions —
    the strongest smart money signal available in daily data.

    Score: % of stocks showing this pattern, scaled to 0-100.
    Optimal 5-10% of stocks → strong but not overheated.
    """
    total = len(price_data)
    momentum_count = 0
    details = []

    for code, df in price_data.items():
        if len(df) < 2:
            continue
        if "vol_ratio" not in df.columns or "return_1d" not in df.columns:
            continue

        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        t_vol = today.get("vol_ratio", 1.0)
        y_vol = yesterday.get("vol_ratio", 1.0)
        t_ret = today.get("return_1d", 0)
        y_ret = yesterday.get("return_1d", 0)
        t_pos = today.get("close_position", 0.5)
        y_pos = yesterday.get("close_position", 0.5)

        if pd.isna(t_vol) or pd.isna(y_vol):
            continue

        # Both days: elevated volume + positive return + strong close
        if (t_vol > 1.3 and y_vol > 1.3 and
            t_ret > 0 and y_ret > 0 and
            t_pos > 0.5 and y_pos > 0.5):
            momentum_count += 1
            details.append({
                "code": code,
                "vol_today": round(float(t_vol), 2),
                "vol_yest": round(float(y_vol), 2),
                "ret_2d": round((float(t_ret) + float(y_ret)) * 100, 2),
            })

    if total == 0:
        return 50.0, 0, []

    pct = (momentum_count / total) * 100

    # Score peaks at 8% (strong but sustainable institutional buying)
    if pct < 2:
        score = pct / 2 * 30 + 10   # 0% → 10, 2% → 40
    elif pct <= 10:
        score = 40 + (pct - 2) / 8 * 55  # 2% → 40, 10% → 95
    elif pct <= 20:
        score = 95 - (pct - 10) / 10 * 45  # 10% → 95, 20% → 50
    else:
        score = max(10, 50 - (pct - 20) / 10 * 40)

    details.sort(key=lambda x: x["ret_2d"], reverse=True)

    return round(score, 2), momentum_count, details[:10]


def compute_smart_flow(price_data):
    """
    Volume-weighted net capital flow direction.

    For each stock:
      flow = volume * sign(return) * close_position_adjustment

    - Positive return with close near high → strong buying flow
    - Positive return with close near low → weakening (sell into strength)
    - Negative return with close near low → strong selling flow
    - Negative return with close near high → weakening (buy the dip)

    Net flow aggregated across all stocks, normalized to 0-100.
    """
    total_buy_flow = 0.0
    total_sell_flow = 0.0

    for code, df in price_data.items():
        if "volume" not in df.columns or "return_1d" not in df.columns:
            continue

        latest = df.iloc[-1]
        vol = latest.get("volume", 0)
        ret = latest.get("return_1d", 0)
        close_pos = latest.get("close_position", 0.5)

        if pd.isna(vol) or pd.isna(ret) or vol <= 0:
            continue

        # Flow magnitude: volume (log scale to avoid mega-caps dominating)
        flow_magnitude = np.log1p(vol)

        # Flow direction: return sign + close position context
        if ret > 0:
            # Buying flow, weighted by close strength
            direction = 0.5 + close_pos * 0.5  # 0.75-1.0 for strong closes
            total_buy_flow += flow_magnitude * direction
        else:
            # Selling flow, weighted by close weakness
            direction = 0.5 + (1 - close_pos) * 0.5
            total_sell_flow += flow_magnitude * direction

    total_flow = total_buy_flow + total_sell_flow
    if total_flow == 0:
        return 50.0, 0, 0

    # Net flow ratio
    net_ratio = total_buy_flow / total_flow  # 0-1, where >0.5 = net buying

    # Scale to 0-100
    score = net_ratio * 100

    return round(score, 2), round(total_buy_flow, 1), round(total_sell_flow, 1)


# ═══════════════════════════════════════════
# Composite + Signal Interpretation
# ═══════════════════════════════════════════

def interpret_smart_money(abnormal_score, acc_score, momentum_score, flow_score):
    """
    Synthesize the 4 smart money dimensions into a clear signal.

    Returns: (signal_label, activity_level, interpretation)
    """
    smart_avg = (abnormal_score + acc_score + momentum_score + flow_score) / 4

    # Activity level
    if smart_avg >= 70:
        activity = "HIGH"
    elif smart_avg >= 45:
        activity = "MEDIUM"
    else:
        activity = "LOW"

    # Signal determination
    if acc_score >= 65 and momentum_score >= 60 and flow_score >= 55:
        signal = "主力吸筹"  # Accumulation + momentum + inflow
    elif acc_score <= 35 and flow_score <= 40:
        signal = "主力出货"  # Distribution + outflow
    elif smart_avg >= 55:
        # Active but mixed signals
        if abs(acc_score - 50) < 15:
            signal = "主力观望"  # Watching, not committing
        else:
            signal = "分歧"      # Conflicting signals
    elif smart_avg >= 40:
        signal = "主力观望"
    else:
        signal = "主力休息"      # Low activity across all dimensions

    # Detailed interpretation
    if signal == "主力吸筹":
        detail = "检测到持续高量买入 + 吸筹形态，主力正在积极建仓"
    elif signal == "主力出货":
        detail = "检测到高量分布形态 + 资金流出，主力可能在减仓"
    elif signal == "分歧":
        detail = "主力内部方向不一致，部分吸筹部分出货，市场可能处于转折点"
    elif signal == "主力观望":
        detail = "主力活动存在但方向不明确，等待明确信号"
    else:
        detail = "主力资金普遍低活跃，市场缺乏机构方向性引导"

    return signal, activity, detail, round(smart_avg, 1)


def compute_emotion():
    """Main entry point — compute full market emotion + smart money report."""
    print("=" * 60)
    print("Market Emotion + Smart Money Model V4.0")
    print("=" * 60)

    # ── Load Data ──
    snap = load_feature_snapshot()
    price_data = load_recent_price_data(lookback=5)

    if snap.empty and not price_data:
        print("No data available")
        return

    total = len(snap) if not snap.empty else len(price_data)
    print(f"Stocks analyzed: {total}")
    print(f"  Feature snapshots: {len(snap)}")
    print(f"  Price histories:   {len(price_data)}")

    # ═══ Market Heat Dimensions ═══
    breadth, above_ma20 = compute_breadth(snap)
    strength, strong_up, strong_down = compute_strength(snap)
    vol_score = compute_volume_momentum(price_data)
    fear = compute_fear(snap)
    trend, trend_aligned = compute_trend(snap)
    spread, near_high, near_low = compute_spread(price_data)

    # ═══ Smart Money Dimensions ═══
    abnormal_score, abnormal_count, abnormal_top = compute_abnormal_volume(price_data)
    acc_score, acc_count, dist_count, acc_top, dist_top = compute_accumulation(price_data)
    momentum_score, momentum_count, momentum_top = compute_institutional_momentum(price_data)
    flow_score, buy_flow, sell_flow = compute_smart_flow(price_data)

    # ═══ Composite Score ═══
    emotion_score = round(
        breadth    * WEIGHTS["breadth"] +
        strength   * WEIGHTS["strength"] +
        vol_score  * WEIGHTS["volume"] +
        fear       * WEIGHTS["fear"] +
        trend      * WEIGHTS["trend"] +
        spread     * WEIGHTS["spread"] +
        abnormal_score * WEIGHTS["abnormal_vol"] +
        acc_score      * WEIGHTS["accumulation"] +
        momentum_score * WEIGHTS["inst_momentum"] +
        flow_score     * WEIGHTS["smart_flow"],
        2
    )

    # Separate heat vs smart money sub-scores
    heat_score = round(
        breadth    * (WEIGHTS["breadth"] / 0.50) +
        strength   * (WEIGHTS["strength"] / 0.50) +
        vol_score  * (WEIGHTS["volume"] / 0.50) +
        fear       * (WEIGHTS["fear"] / 0.50) +
        trend      * (WEIGHTS["trend"] / 0.50) +
        spread     * (WEIGHTS["spread"] / 0.50),
        2
    )

    smart_score = round(
        abnormal_score * (WEIGHTS["abnormal_vol"] / 0.50) +
        acc_score      * (WEIGHTS["accumulation"] / 0.50) +
        momentum_score * (WEIGHTS["inst_momentum"] / 0.50) +
        flow_score     * (WEIGHTS["smart_flow"] / 0.50),
        2
    )

    # ═══ Smart Money Signal ═══
    signal, activity, signal_detail, smart_avg = interpret_smart_money(
        abnormal_score, acc_score, momentum_score, flow_score
    )

    # ═══ Level & Position ═══
    # Market emotion + smart money influence positioning
    if emotion_score >= 80:
        level = "极度积极"
        position_pct = 80
    elif emotion_score >= 60:
        level = "偏积极"
        position_pct = 60
    elif emotion_score >= 40:
        level = "中性"
        position_pct = 40
    elif emotion_score >= 20:
        level = "偏谨慎"
        position_pct = 20
    else:
        level = "极度恐慌"
        position_pct = 10

    # Smart money can override: strong accumulation → +10% position
    if signal == "主力吸筹":
        position_pct = min(90, position_pct + 10)
    elif signal == "主力出货":
        position_pct = max(5, position_pct - 15)

    # ═══ Output ═══
    result = pd.DataFrame([{
        "date":                datetime.now().strftime("%Y-%m-%d"),
        "market_emotion":      emotion_score,
        "heat_score":          heat_score,       # Market crowd sentiment
        "smart_money_score":   smart_score,       # Institutional activity
        "level":               level,
        "smart_money_signal":  signal,
        "smart_money_activity": activity,
        "suggested_position_pct": position_pct,
        # Market heat breakdown
        "breadth":             round(breadth, 2),
        "strength_depth":      round(strength, 2),
        "volume_momentum":     round(vol_score, 2),
        "fear_index":          round(fear, 2),
        "trend_alignment":     round(trend, 2),
        "extreme_spread":      round(spread, 2),
        # Smart money breakdown
        "abnormal_volume":     abnormal_score,
        "accumulation_signal": acc_score,
        "inst_momentum":       momentum_score,
        "smart_flow":          flow_score,
        # Counters
        "stocks_analyzed":     total,
        "above_ma20_count":    int(above_ma20),
        "strong_up_count":     int(strong_up),
        "strong_down_count":   int(strong_down),
        "near_high":           int(near_high),
        "near_low":            int(near_low),
        "abnormal_vol_count":  abnormal_count,
        "accumulating_count":  acc_count,
        "distributing_count":  dist_count,
        "inst_momentum_count": momentum_count,
    }])

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # ═══ Print Detailed Diagnostics ═══
    print(f"\n{'─' * 60}")
    print(f"  COMPOSITE:  {emotion_score:.0f}  →  {level}")
    print(f"  Position:   {position_pct}%")
    print(f"{'─' * 60}")
    print(f"  Market Heat:  {heat_score:.0f}  |  Smart Money:  {smart_score:.0f}")
    print(f"  Smart Signal: {signal} ({activity})")
    print(f"  → {signal_detail}")
    print(f"{'─' * 60}")
    print(f"  -- Market Heat Dimensions --")
    print(f"  Breadth (MA20):       {breadth:5.1f}  [{above_ma20}/{total} above MA20]")
    print(f"  Strength (>3%):       {strength:5.1f}  [↑{strong_up} ↓{strong_down}]")
    print(f"  Volume Momentum:      {vol_score:5.1f}")
    print(f"  Fear Index (inv):     {fear:5.1f}")
    print(f"  Trend (5>20):         {trend:5.1f}  [{trend_aligned} aligned]")
    print(f"  Spread (Hi/Lo):       {spread:5.1f}  [H:{near_high} L:{near_low}]")
    print(f"  -- Smart Money Dimensions --")
    print(f"  Abnormal Volume:      {abnormal_score:5.1f}  [{abnormal_count} stocks vol>2x]")
    print(f"  Accumulation Signal:  {acc_score:5.1f}  [吸{acc_count} 出{dist_count}]")
    print(f"  Inst Momentum:        {momentum_score:5.1f}  [{momentum_count} consecutive]")
    print(f"  Smart Flow:           {flow_score:5.1f}  [buy:{buy_flow:.0f} sell:{sell_flow:.0f}]")

    # Top smart money stocks
    if abnormal_top:
        print(f"\n  [Abnormal Volume] Top:")
        for s in abnormal_top[:5]:
            print(f"     {s['code']}  vol:{s['vol_ratio']}x  ret:{s['return']:+.1f}%")

    if acc_top:
        print(f"\n  [Accumulation] Top:")
        for s in acc_top[:5]:
            print(f"     {s['code']}  vol:{s['vol_ratio']}x  ret:{s['return']:+.1f}%  pos:{s['close_pos']}")

    if momentum_top:
        print(f"\n  [Inst Momentum 2-day] Top:")
        for s in momentum_top[:5]:
            print(f"     {s['code']}  vol_t:{s['vol_today']}x  vol_y:{s['vol_yest']}x  ret_2d:{s['ret_2d']:+.1f}%")

    print(f"\n  Saved: {OUTPUT_FILE}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    compute_emotion()
