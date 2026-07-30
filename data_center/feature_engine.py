"""
A-Insight Pro
股票特征工程 V2.0
- 45 features: technical + derived + market context
- Walk-forward normalization (no look-ahead bias)
- Configurable feature set via model_config.yaml
"""

import os
import numpy as np
import pandas as pd
from config.settings import DATA_DIR, FEATURE_DIR, MODEL_CFG


def calc_technical_indicators(df):
    """Calculate all technical indicators on a DataFrame."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Moving Averages ---
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    # MA ratios to close
    df["ma5_ratio"] = close / df["ma5"] - 1
    df["ma10_ratio"] = close / df["ma10"] - 1
    df["ma20_ratio"] = close / df["ma20"] - 1
    df["ma60_ratio"] = close / df["ma60"] - 1

    # --- Bollinger Bands (20-day, 2 std) ---
    df["bollinger_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bollinger_upper"] = df["bollinger_mid"] + 2 * bb_std
    df["bollinger_lower"] = df["bollinger_mid"] - 2 * bb_std
    df["bollinger_pct_b"] = (close - df["bollinger_lower"]) / (df["bollinger_upper"] - df["bollinger_lower"] + 1e-10)
    df["bollinger_bandwidth"] = (df["bollinger_upper"] - df["bollinger_lower"]) / (df["bollinger_mid"] + 1e-10)

    # --- Returns ---
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    # --- Volatility ---
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # --- RSI (14) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # --- OBV (On-Balance Volume) ---
    df["obv"] = (volume * np.sign(close.diff().fillna(0))).cumsum()

    # --- MFI (Money Flow Index 14) ---
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price.diff() > 0, 0.0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price.diff() < 0, 0.0).rolling(14).sum()
    mfi_ratio = positive_flow / (negative_flow + 1e-10)
    df["mfi_14"] = 100 - (100 / (1 + mfi_ratio))

    # --- ATR (Average True Range 14) ---
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # --- Volume ---
    df["volume_ratio_20d"] = volume / volume.rolling(20).mean()

    # --- Price position ---
    df["price_position_20d"] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-10)
    df["price_position_60d"] = (close - close.rolling(60).min()) / (close.rolling(60).max() - close.rolling(60).min() + 1e-10)

    # --- Target: future 5-day return ---
    df["future_return"] = close.shift(-5) / close - 1

    # --- V1 Compatibility aliases (for backward compat with predict.py) ---
    df["return"] = df["return_1d"]                    # V1: return = 1d return
    df["volatility"] = df["volatility_20d"]           # V1: volatility = 20d vol
    df["rsi"] = df["rsi_14"]                          # V1: rsi = RSI(14)
    df["volume_change"] = volume.pct_change()         # V1: volume change
    # ma5/ma10/ma20/ma60 already exist in both V1 and V2
    # macd already exists in both V1 and V2

    return df


def add_market_features(df, index_df):
    """Add market context features from index data."""
    if index_df is None or len(index_df) < 60:
        for col in ["csi300_return_1d", "csi300_return_5d", "csi300_volatility_20d"]:
            df[col] = 0.0
        df["market_breadth"] = 0.5
        return df

    index_df = index_df.copy()
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.sort_values("date")
    index_df["csi300_return"] = index_df["close"].pct_change()
    index_df["csi300_return_5d"] = index_df["close"].pct_change(5)
    index_df["csi300_volatility_20d"] = index_df["csi300_return"].rolling(20).std()

    # Merge on date
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(
        index_df[["date", "csi300_return", "csi300_return_5d", "csi300_volatility_20d"]],
        on="date", how="left"
    )
    df.rename(columns={"csi300_return": "csi300_return_1d"}, inplace=True)

    for col in ["csi300_return_1d", "csi300_return_5d", "csi300_volatility_20d"]:
        df[col] = df[col].fillna(method="ffill").fillna(0)

    df["market_breadth"] = 0.5  # Will be updated by emotion module
    return df


def walk_forward_normalize(df, feature_cols, min_window=120):
    """Normalize features using expanding window z-score (no look-ahead bias)."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns or col in ["future_return", "date", "code"]:
            continue
        # Expanding window mean/std
        rolling_mean = df[col].expanding(min_periods=min_window).mean()
        rolling_std = df[col].expanding(min_periods=min_window).std()
        df[f"{col}_norm"] = (df[col] - rolling_mean) / (rolling_std + 1e-10)
    return df


def calc_features(code, index_df=None):
    """Calculate all features for a single stock."""
    file = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(file):
        return False

    df = pd.read_csv(file, encoding="utf-8-sig")
    if len(df) < 120:
        return False

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Calculate all technical indicators
    df = calc_technical_indicators(df)

    # Add market context
    if MODEL_CFG.get("features", {}).get("market_features", {}).get("enabled", True):
        df = add_market_features(df, index_df)

    # ── V2.1: Compute advanced alpha factors ──
    try:
        from ai.advanced_factors import compute_advanced_factors_for_stock
        market_ret = None
        if index_df is not None and "close" in index_df.columns:
            market_ret = index_df["close"].pct_change().values
        adv_df = compute_advanced_factors_for_stock(df, market_ret)
        if adv_df is not None and not adv_df.empty:
            for col in adv_df.columns:
                if col not in df.columns:
                    df[col] = adv_df[col].values
    except ImportError:
        pass
    except Exception:
        pass

    # Normalize numeric feature columns
    feature_cols = [c for c in df.columns if c not in ("date", "future_return", "code")]
    df = walk_forward_normalize(df, feature_cols)

    # Save ALL rows (training filters NaN later, prediction needs latest row)
    out = os.path.join(FEATURE_DIR, f"{code}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # Training requires valid future_return — check if we have enough
    valid_rows = df.dropna(subset=["future_return"])
    if len(valid_rows) < 60:
        return False
    print(f"  {code} 完成, {len(df)} 行, {len(df.columns)} 特征")
    return True


def run(index_df=None):
    """Run feature engineering for all stocks."""
    print("=" * 40)
    print("特征工程 V2.1 启动")
    print("=" * 40)

    os.makedirs(FEATURE_DIR, exist_ok=True)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    codes = [f.replace(".csv", "") for f in files if len(f.replace(".csv", "")) == 6]

    print(f"股票数量: {len(codes)}")

    success = 0
    for i, code in enumerate(codes, 1):
        print(f"[{i}/{len(codes)}] {code}")
        if calc_features(code, index_df):
            success += 1

    print("=" * 40)
    print(f"特征工程完成: {success}/{len(codes)}")
    print("=" * 40)

    # Add sector features (V2.1)
    print("\n添加行业板块特征...")
    try:
        from data_center.sector_features import compute_sector_features
        compute_sector_features()
    except Exception as e:
        print(f"  行业特征添加失败: {e}")

    # Add cross-sectional rank factors (V2.2)
    print("\n计算横截面排名因子...")
    try:
        from ai.advanced_factors import compute_cross_sectional_factors
        # Load all feature files
        stock_dict = {}
        for fname in os.listdir(FEATURE_DIR):
            if not fname.endswith(".csv"):
                continue
            code = fname.replace(".csv", "").zfill(6)
            try:
                df = pd.read_csv(os.path.join(FEATURE_DIR, fname), encoding="utf-8-sig")
                if len(df) >= 60:
                    stock_dict[code] = df
            except Exception:
                continue
        if stock_dict:
            stock_dict = compute_cross_sectional_factors(stock_dict)
            # Save updated DataFrames back
            for code, df in stock_dict.items():
                df.to_csv(os.path.join(FEATURE_DIR, f"{code}.csv"), index=False, encoding="utf-8-sig")
            print(f"  横截面因子已保存到 {len(stock_dict)} 只股票")
    except Exception as e:
        print(f"  横截面因子计算失败: {e}")

    return success


if __name__ == "__main__":
    run()
