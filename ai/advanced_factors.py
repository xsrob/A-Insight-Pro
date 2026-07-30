"""
A-Insight Pro
Advanced Alpha Factors V1.0 — High-Impact Factors with Academic Support

New factor categories (25+ factors):

1. CROSS_SECTIONAL (6):  Relative rankings vs peer universe — captures
   relative value that time-series factors miss entirely.
   - cs_rank_mom_5d, cs_rank_mom_20d, cs_rank_vol_ratio
   - cs_rank_rsi, cs_rank_turnover, cs_rank_volatility

2. MAX_EFFECT (4):  Stocks with extreme positive daily returns subsequently
   underperform. One of the strongest anomalies in global markets, especially
   pronounced in A-shares due to retail investor attention bias.
   - max_ret_5d, max_ret_20d, min_ret_20d, ret_range_20d

3. SHORT_TERM_REVERSAL (4):  1-5 day reversals are strong in A-shares
   driven by retail overreaction and T+1 settlement mechanics.
   - reversal_1d, reversal_5d, reversal_vol_adj, ret_skew_5d

4. IDIOSYNCRATIC (5):  Residual risk/return after removing market factor.
   Idio vol (low vol anomaly), idio momentum (firm-specific drift).
   - idio_vol_20d, idio_mom_20d, beta_60d, r_squared, market_corr

5. LIQUIDITY (5):  Liquidity premium — less liquid stocks earn higher
   returns as compensation. Amihud measure, turnover, dollar volume.
   - amihud_illiq, turnover_5d, turnover_std_20d, turnover_trend,
     dollar_vol_growth

6. PATH_DEPENDENT (4):  How price got here matters beyond where it is.
   Path choppiness, consecutive moves, up-day ratio.
   - path_volatility, consecutive_ret, up_days_ratio, streak_signal

7. GAP_EFFECTS (3):  Overnight gaps contain distinct information from
   intraday moves. Gap-fill behavior is a real phenomenon.
   - overnight_gap, intraday_ret, gap_ma_divergence

8. LEAD_LAG (3):  Sector-relative dynamics. Leaders predict followers.
   - sector_relative_5d, sector_lead_5d, peer_corr_20d

Total: 34 new factors across 8 categories.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

FEATURE_DIR = "features"
DATA_DIR = "data"


# ============================================================
# Per-Stock Factor Computation (can run on single stock)
# ============================================================

def compute_max_effect_factors(df):
    """
    MAX effect: Stocks with extreme positive daily returns underperform.

    Bali, Cakici, Whitelaw (2011): MAX factor predicts negative returns.
    Stronger in markets with high retail participation (like A-shares).
    """
    ret = df["close"].pct_change()
    factors = {}

    # Maximum daily return in window
    if len(ret) > 5:
        factors["max_ret_5d"] = ret.rolling(5).max()
    if len(ret) > 20:
        factors["max_ret_20d"] = ret.rolling(20).max()
    if len(ret) > 20:
        factors["min_ret_20d"] = ret.rolling(20).min()
        factors["ret_range_20d"] = factors["max_ret_20d"] - factors["min_ret_20d"]

    return factors


def compute_reversal_factors(df):
    """
    Short-term reversal: 1-5 day price reversals.

    Jegadeesh (1990), Lehmann (1990): Short-term reversals exist
    across markets. In A-shares, T+1 settlement creates mechanical
    reversal pressure.

    reversal_1d: Negative of today's return (buy losers)
    reversal_vol_adj: Reversal scaled by recent volatility
    ret_skew_5d: Skewness of daily returns (negative skew → crash risk)
    """
    ret = df["close"].pct_change()
    factors = {}

    if len(ret) > 1:
        factors["reversal_1d"] = -ret  # Short-term reversal signal

    if len(ret) > 5:
        factors["reversal_5d"] = -df["close"].pct_change(5)
        # Skewness of daily returns over 5 days
        factors["ret_skew_5d"] = ret.rolling(5).skew()

    # Volatility-adjusted reversal (stronger signal for high-vol stocks)
    if len(ret) > 10:
        vol_10d = ret.rolling(10).std()
        factors["reversal_vol_adj"] = -ret / (vol_10d + 1e-10)

    return factors


def compute_idiosyncratic_factors(df, market_ret=None):
    """
    Idiosyncratic risk & return: residual after removing market factor.

    Ang, Hodrick, Xing, Zhang (2006): Idiosyncratic volatility puzzle
    — high idio vol → low future returns.

    beta_60d: Rolling market beta (CAPM)
    idio_vol_20d: Std of residuals after removing beta*market
    idio_mom_20d: Residual momentum (firm-specific drift, not market)
    r_squared: How much of stock return is explained by market
    """
    ret = df["close"].pct_change()
    factors = {}

    # If no market return provided, use flat market as fallback
    if market_ret is None or len(market_ret) != len(ret):
        market_ret = pd.Series(0.0, index=ret.index)

    if len(ret) > 60:
        # Rolling beta: cov(stock, market) / var(market)
        cov = ret.rolling(60).cov(market_ret)
        var_mkt = market_ret.rolling(60).var()
        factors["beta_60d"] = cov / (var_mkt + 1e-10)

        # R-squared from beta regression
        stock_var = ret.rolling(60).var()
        factors["r_squared"] = (factors["beta_60d"] ** 2 * var_mkt) / (stock_var + 1e-10)
        factors["market_corr"] = ret.rolling(60).corr(market_ret)

    if len(ret) > 20 and "beta_60d" in factors:
        # Idiosyncratic return: actual - beta * market
        expected_ret = factors["beta_60d"].shift(1) * market_ret
        residual = ret - expected_ret

        # Idio volatility
        factors["idio_vol_20d"] = residual.rolling(20).std()

        # Idio momentum (20-day cumulative residual return)
        factors["idio_mom_20d"] = residual.rolling(20).sum()

    return factors


def compute_liquidity_factors(df):
    """
    Liquidity factors: Amihud illiquidity, turnover.

    Amihud (2002): Illiquidity measure = |return| / dollar_volume.
    Higher illiquidity → higher expected return (liquidity premium).

    Turnover: High turnover predicts low returns (attention-driven
    overpricing, especially in A-shares).
    """
    close = df["close"]
    volume = df["volume"]
    ret = close.pct_change()
    factors = {}

    if len(df) > 20:
        # Amihud illiquidity: |return| / (volume * price) averaged over window
        # Scale by 10^8 because A-share dollar volumes are very large (100M+ yuan)
        dollar_vol = volume * close
        daily_illiq = abs(ret) / (dollar_vol + 1e-10) * 1e8
        factors["amihud_illiq"] = daily_illiq.rolling(20).mean()

        # For cross-sectional ranking, log-transform (distribution is very skewed)
        factors["amihud_illiq_log"] = np.log(factors["amihud_illiq"] + 1e-15)

        # Turnover proxy (volume relative to 20-day average)
        vol_ma20 = volume.rolling(20).mean()
        factors["turnover_5d"] = volume.rolling(5).mean() / (vol_ma20 + 1e-10)

        # Turnover volatility (change in trading activity)
        turnover = volume / (vol_ma20 + 1e-10)
        factors["turnover_std_20d"] = turnover.rolling(20).std()

        # Turnover trend: is trading activity accelerating or decelerating?
        factors["turnover_trend"] = (volume.rolling(5).mean() - volume.rolling(10).mean()) / (vol_ma20 + 1e-10)

        # Dollar volume growth
        factors["dollar_vol_growth"] = dollar_vol.pct_change(20)

    return factors


def compute_path_dependent_factors(df):
    """
    Path-dependent factors: how price arrived at current level.

    Path volatility (choppiness): ratio of sum(abs(daily_returns)) to abs(total_return).
    High = choppy path (indecision), Low = smooth trend.

    Consecutive returns: streak of same-direction days.
    Up days ratio: % of positive days in window.
    """
    ret = df["close"].pct_change()
    factors = {}

    if len(ret) > 10:
        # Path volatility (efficiency ratio / choppiness index)
        abs_sum = ret.rolling(10).apply(lambda x: np.sum(np.abs(x)), raw=True)
        total_ret = df["close"].pct_change(10)
        raw_path_vol = abs_sum / (np.abs(total_ret) + 1e-10)
        # Clip extreme values: total_ret near zero creates huge ratios
        # Typical range: 1.0 (smooth trend) to ~3.0 (very choppy)
        factors["path_volatility"] = np.clip(raw_path_vol, 0.5, 10.0)

    if len(ret) > 5:
        # Consecutive same-direction days (current streak)
        signs = np.sign(ret.dropna().values)
        if len(signs) > 0:
            current_sign = signs[-1]
            streak = 0
            for s in reversed(signs):
                if s == current_sign:
                    streak += 1
                else:
                    break
            # Positive = consecutive up, negative = consecutive down
            factors["consecutive_ret"] = float(streak * current_sign)

        # Up days ratio over 20 days
        factors["up_days_ratio"] = (ret > 0).rolling(20).mean()

        # Streak signal: combines streak length and direction
        if "consecutive_ret" in factors:
            streak_val = factors["consecutive_ret"]
            # After 3+ up days, mean-reversion pressure builds
            # After 3+ down days, bounce probability increases
            if streak_val >= 4:
                factors["streak_signal"] = -0.5  # Overextended up
            elif streak_val <= -4:
                factors["streak_signal"] = 0.5   # Oversold bounce
            elif streak_val >= 2:
                factors["streak_signal"] = -0.2
            elif streak_val <= -2:
                factors["streak_signal"] = 0.2
            else:
                factors["streak_signal"] = 0.0

    return factors


def compute_gap_factors(df):
    """
    Overnight gap & intraday return decomposition.

    gap = (open - prev_close) / prev_close  → overnight information
    intraday = (close - open) / open         → intraday price discovery

    Research shows overnight and intraday returns have different
    predictive content. Gaps often partially fill.
    """
    factors = {}
    if "open" not in df.columns or len(df) < 5:
        return factors

    close = df["close"]
    open_p = df["open"]

    # Overnight gap
    factors["overnight_gap"] = (open_p - close.shift(1)) / (close.shift(1) + 1e-10)

    # Intraday return
    factors["intraday_ret"] = (close - open_p) / (open_p + 1e-10)

    # Gap vs MA: is price gapping away from MA20?
    if "ma20" in df.columns:
        factors["gap_ma_divergence"] = (open_p - df["ma20"]) / (df["ma20"] + 1e-10) - \
                                        (close.shift(1) - df["ma20"].shift(1)) / (df["ma20"].shift(1) + 1e-10)

    # Gap reversal probability (simplified): if gap > 2%, expect partial fill
    if len(df) > 10:
        gaps = factors["overnight_gap"]
        # Rolling probability of gap fill (same-day reversal after gap)
        gap_filled = (np.sign(gaps) != np.sign(factors.get("intraday_ret", gaps)))
        factors["gap_fill_prob"] = gap_filled.rolling(20).mean()

    return factors


def compute_lead_lag_factors(df, sector_ret=None):
    """
    Sector-relative dynamics.

    sector_relative_5d: Stock return minus sector return — stock-specific alpha
    sector_lead_5d: Did stock move BEFORE its sector? (lead indicator)
    peer_corr_20d: Correlation with sector peers — diversification value
    """
    ret = df["close"].pct_change()
    factors = {}

    if sector_ret is not None and len(ret) > 5:
        # Relative strength vs sector
        stock_ret_5d = df["close"].pct_change(5)
        factors["sector_relative_5d"] = stock_ret_5d - sector_ret

        # Lead indicator: stock 5d return vs sector 5d return
        # Positive = stock leading sector up, Negative = stock lagging
        if len(ret) > 10:
            stock_lead_5d = ret.shift(5).rolling(5).sum()
            sector_lead_5d_series = sector_ret.shift(5).rolling(5).sum() if hasattr(sector_ret, 'shift') else 0
            factors["sector_lead_5d"] = stock_lead_5d - sector_lead_5d_series

    if sector_ret is not None and len(ret) > 20:
        factors["peer_corr_20d"] = ret.rolling(20).corr(sector_ret)

    return factors


# ============================================================
# Cross-Sectional Factors (requires all stocks simultaneously)
# ============================================================

def compute_cross_sectional_factors(stock_data_dict):
    """
    Compute cross-sectional (rank-based) factors across all stocks.

    For each date, ranks stocks on key metrics and converts to
    percentile scores [0, 1]. This captures RELATIVE value:
    - A stock with 5% return might be weak if peers returned 10%
    - A stock with 3% return might be strong if peers returned -2%

    Args:
        stock_data_dict: {code: DataFrame} with feature data

    Returns:
        Updated stock_data_dict with cs_rank_* columns added.
    """
    print("  Computing cross-sectional factors...")

    # Collect all (date, code) pairs with key metrics
    metrics = ["return_5d", "return_20d", "vol_ratio", "rsi", "turnover", "volatility"]
    metric_sources = {
        "return_5d": lambda df: df["close"].pct_change(5) if "close" in df.columns else pd.Series(dtype=float),
        "return_20d": lambda df: df["close"].pct_change(20) if "close" in df.columns else pd.Series(dtype=float),
        "vol_ratio": lambda df: df["volume"] / df["volume"].rolling(20).mean() if "volume" in df.columns else pd.Series(dtype=float),
        "rsi": lambda df: df["rsi_14"] if "rsi_14" in df.columns else (df["rsi"] if "rsi" in df.columns else pd.Series(dtype=float)),
        "turnover": lambda df: df["volume"] / df["volume"].rolling(20).mean() if "volume" in df.columns else pd.Series(dtype=float),
        "volatility": lambda df: df["close"].pct_change().rolling(20).std() if "close" in df.columns else pd.Series(dtype=float),
    }

    # First pass: compute each metric for each stock
    stock_metrics = {}
    all_dates = set()

    for code, df in stock_data_dict.items():
        if len(df) < 60:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        if "date" not in df.columns:
            continue

        stock_m = pd.DataFrame({"date": df["date"]})
        for metric_name, compute_fn in metric_sources.items():
            try:
                stock_m[metric_name] = compute_fn(df).values
            except Exception:
                stock_m[metric_name] = np.nan

        stock_metrics[code] = stock_m
        all_dates.update(df["date"].dropna().values)

    sorted_dates = sorted(all_dates)
    print(f"    Dates with data: {len(sorted_dates)}")

    # Second pass: for each date, rank stocks cross-sectionally
    # Optimization: process in batches by date
    n_stocks = len(stock_metrics)
    rank_columns = {}

    for metric_name in metrics:
        rank_columns[metric_name] = {}

    processed_dates = 0
    for date in sorted_dates:
        date_metrics = {}
        for code, sm in stock_metrics.items():
            date_mask = sm["date"] == date
            if not date_mask.any():
                continue
            row = sm[date_mask].iloc[0]
            vals = {}
            for m in metrics:
                v = row.get(m, np.nan)
                if not pd.isna(v):
                    vals[m] = v
            if vals:
                date_metrics[code] = vals

        if len(date_metrics) < 30:  # Need minimum stocks for meaningful ranks
            continue

        # Rank each metric across stocks
        for metric_name in metrics:
            values = {}
            for code, mv in date_metrics.items():
                if metric_name in mv:
                    values[code] = mv[metric_name]

            if len(values) < 30:
                continue

            # Sort and assign percentile ranks [0, 1]
            sorted_codes = sorted(values, key=values.get)
            n = len(sorted_codes)
            for rank_idx, code in enumerate(sorted_codes):
                percentile = rank_idx / (n - 1) if n > 1 else 0.5
                if code not in rank_columns[metric_name]:
                    rank_columns[metric_name][code] = {}
                rank_columns[metric_name][code][date] = percentile

        processed_dates += 1

    print(f"    Processed {processed_dates} dates")

    # Third pass: merge rank columns back into stock DataFrames
    merged_count = 0
    for code in list(stock_data_dict.keys()):
        df = stock_data_dict[code].copy()
        df = df.sort_values("date").reset_index(drop=True)

        for metric_name in metrics:
            col_name = f"cs_rank_{metric_name}"
            df[col_name] = np.nan

            if code in rank_columns[metric_name]:
                rank_map = rank_columns[metric_name][code]
                for idx, row in df.iterrows():
                    date_val = row.get("date")
                    if date_val in rank_map:
                        df.at[idx, col_name] = rank_map[date_val]

            # Forward-fill missing ranks (stock not in top N on some dates)
            df[col_name] = df[col_name].fillna(0.5)

        stock_data_dict[code] = df
        merged_count += 1

    print(f"    Cross-sectional factors merged to {merged_count} stocks")
    print(f"    Factors: {[f'cs_rank_{m}' for m in metrics]}")

    return stock_data_dict


# ============================================================
# New Factor Categories for factor_engine.py
# ============================================================

ADVANCED_FACTOR_CATEGORIES = {
    "cross_sectional": [
        "cs_rank_return_5d", "cs_rank_return_20d", "cs_rank_vol_ratio",
        "cs_rank_rsi", "cs_rank_turnover", "cs_rank_volatility",
    ],
    "max_effect": [
        "max_ret_5d", "max_ret_20d", "min_ret_20d", "ret_range_20d",
    ],
    "short_term_reversal": [
        "reversal_1d", "reversal_5d", "reversal_vol_adj", "ret_skew_5d",
    ],
    "idiosyncratic": [
        "idio_vol_20d", "idio_mom_20d", "beta_60d", "r_squared", "market_corr",
    ],
    "liquidity": [
        "amihud_illiq", "amihud_illiq_log", "turnover_5d",
        "turnover_std_20d", "turnover_trend", "dollar_vol_growth",
    ],
    "path_dependent": [
        "path_volatility", "consecutive_ret", "up_days_ratio", "streak_signal",
    ],
    "gap_effects": [
        "overnight_gap", "intraday_ret", "gap_ma_divergence", "gap_fill_prob",
    ],
    "lead_lag": [
        "sector_relative_5d", "sector_lead_5d", "peer_corr_20d",
    ],
}


def get_all_advanced_factor_names():
    """Return flat list of all advanced factor names."""
    names = []
    for cat_factors in ADVANCED_FACTOR_CATEGORIES.values():
        names.extend(cat_factors)
    return names


def compute_advanced_factors_for_stock(df, market_ret=None, sector_ret=None):
    """
    Compute all advanced per-stock factors for a single stock DataFrame.

    Args:
        df: DataFrame with OHLCV + technical indicators
        market_ret: Series of market (CSI300) daily returns, aligned by index
        sector_ret: Series of sector daily returns, aligned by index

    Returns:
        DataFrame with advanced factor columns (index-aligned with df)
    """
    if len(df) < 60:
        return pd.DataFrame(index=df.index)

    df = df.sort_values("date").reset_index(drop=True)

    all_factors = {}

    # Compute all categories
    all_factors.update(compute_max_effect_factors(df))
    all_factors.update(compute_reversal_factors(df))
    all_factors.update(compute_idiosyncratic_factors(df, market_ret))
    all_factors.update(compute_liquidity_factors(df))
    all_factors.update(compute_path_dependent_factors(df))
    all_factors.update(compute_gap_factors(df))
    all_factors.update(compute_lead_lag_factors(df, sector_ret))

    result = pd.DataFrame(all_factors, index=df.index)
    return result
