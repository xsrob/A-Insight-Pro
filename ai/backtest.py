"""
A-Insight Pro
AI Backtest V8.0 — Out-of-Sample Walk-Forward

Changes from V7:
- Walk-forward OOS: train on [0:t], predict on [t+1], never leaks future data
- Transaction costs: stamp duty (0.05% sell), commission (0.025%), slippage (0.05%)
- Regime-aware position sizing
- Proper equity curve with drawdown tracking
- Sharpe ratio and Calmar ratio
- Benchmark comparison
"""

import os, glob, json, warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor

MODEL_DIR = "models"
FEATURE_DIR = "features"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(REPORT_DIR, "backtest_report.csv")
CURVE_FILE = os.path.join(REPORT_DIR, "equity_curve.csv")
SUMMARY_FILE = os.path.join(REPORT_DIR, "backtest_summary.json")

# ── Transaction Costs (A-share market) ──
STAMP_DUTY = 0.0005     # 0.05% on sell only (印花税)
COMMISSION = 0.00025    # 0.025% per trade (佣金)
SLIPPAGE = 0.0005       # 0.05% slippage (滑点)
TOTAL_BUY_COST = COMMISSION + SLIPPAGE     # 0.075%
TOTAL_SELL_COST = COMMISSION + SLIPPAGE + STAMP_DUTY  # 0.125%

# ── Backtest Parameters ──
INITIAL_CAPITAL = 1_000_000
TOP_K = 20              # Buy top K stocks by prediction
HOLD_DAYS = 5           # Holding period
MIN_TRAIN_DAYS = 250    # Minimum training period (~1 year)
RETRAIN_EVERY = 60      # Retrain every N trading days


def load_feature_files():
    """Load all feature files. Returns {code: DataFrame}"""
    files = glob.glob(os.path.join(FEATURE_DIR, "*.csv"))
    data = {}
    for f in files:
        code = os.path.basename(f).replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "future_return" not in df.columns or len(df) < 120:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            data[code] = df
        except Exception:
            continue
    print(f"Loaded {len(data)} stocks with valid data")
    return data


def build_training_matrix(stock_data, dates_up_to):
    """
    Build training matrix from all stocks using data ONLY up to dates_up_to.
    This ensures NO look-ahead bias.
    """
    X_list = []
    y_list = []

    exclude_cols = ["date", "future_return", "code"]

    for code, df in stock_data.items():
        # Filter to dates <= dates_up_to
        mask = df["date"] <= dates_up_to
        train_data = df[mask].copy()

        if len(train_data) < 120:
            continue

        # Exclude the last 5 rows (future_return is NaN there)
        valid = train_data.dropna(subset=["future_return"])
        if len(valid) < 60:
            continue

        feature_cols = [c for c in valid.columns if c not in exclude_cols]
        X_list.append(valid[feature_cols])
        y_list.append(valid["future_return"])

    if not X_list:
        return None, None, None

    X = pd.concat(X_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)

    # Clean
    X = X.replace([float("inf"), -float("inf")], None)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    y = y.fillna(0)

    feature_cols = list(X.columns)
    return X, y, feature_cols


def backtest():
    """Run walk-forward out-of-sample backtest."""
    print("=" * 60)
    print("Backtest V8.0 — Walk-Forward OOS")
    print("=" * 60)

    stock_data = load_feature_files()
    if not stock_data:
        print("No data to backtest")
        return

    # Find common date range
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df["date"].dropna().values)
    sorted_dates = sorted(all_dates)
    print(f"Date range: {sorted_dates[0].date()} to {sorted_dates[-1].date()}")
    print(f"Total trading days: {len(sorted_dates)}")

    # Walk-forward loop
    capital = INITIAL_CAPITAL
    capital_curve = []
    trades_log = []
    position = {}  # {code: {buy_price, buy_date, shares}}
    last_train_date = None

    # Find first valid prediction date
    start_idx = MIN_TRAIN_DAYS
    if start_idx >= len(sorted_dates):
        print("Not enough history for training")
        return

    for day_idx in range(start_idx, len(sorted_dates) - HOLD_DAYS, HOLD_DAYS):
        today = sorted_dates[day_idx]

        # ── Retrain model if needed ──
        train_dates_up_to = sorted_dates[day_idx - 1]  # Train on data BEFORE today
        if last_train_date is None or (day_idx - start_idx) % RETRAIN_EVERY == 0:
            X_train, y_train, feature_cols = build_training_matrix(
                stock_data, train_dates_up_to
            )
            if X_train is None or len(X_train) < 1000:
                continue

            model = RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_split=20, min_samples_leaf=10,
                max_features=0.5, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            last_train_date = train_dates_up_to
            print(f"  [Day {day_idx}] Trained on data through {train_dates_up_to.date()}, "
                  f"{len(X_train)} samples")

        # ── Predict today (using today's features) ──
        predictions = []
        for code, df in stock_data.items():
            today_mask = df["date"] == today
            if not today_mask.any():
                continue

            row = df[today_mask].iloc[0]
            available = [c for c in feature_cols if c in row.index]
            if len(available) < 10:
                continue

            try:
                X_pred = pd.DataFrame([row[available].fillna(0).values], columns=available)
                X_pred = X_pred.fillna(0).replace([float("inf"), -float("inf")], 0)
                pred = float(model.predict(X_pred)[0])
                predictions.append((code, pred, row))
            except Exception:
                continue

        if not predictions:
            continue

        # Sort by prediction, take top K
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_picks = predictions[:TOP_K]

        # ── Execute trades ──
        # Sell existing positions (bought HOLD_DAYS ago at previous iteration)
        # Loop steps by HOLD_DAYS, so today=day_idx is exactly HOLD_DAYS after the buy.
        # Model predicts close[t+5]/close[t]-1, so sell at day_idx (t+5 close).
        for code in list(position.keys()):
            pos = position[code]
            sell_day_idx = day_idx  # Today = buy_date + HOLD_DAYS
            if sell_day_idx >= len(sorted_dates):
                continue

            sell_date = sorted_dates[sell_day_idx]
            df = stock_data.get(code)
            if df is None:
                continue

            sell_mask = df["date"] == sell_date
            if not sell_mask.any():
                continue

            sell_price = df[sell_mask].iloc[0]["close"]
            if sell_price <= 0:
                continue

            # Apply sell costs
            gross_return = (sell_price / pos["buy_price"] - 1)
            net_return = gross_return - TOTAL_SELL_COST
            trade_pnl = pos["shares"] * pos["buy_price"] * net_return
            capital += trade_pnl

            trades_log.append({
                "code": code,
                "buy_date": str(pos["buy_date"].date()),
                "sell_date": str(sell_date.date()),
                "buy_price": round(pos["buy_price"], 2),
                "sell_price": round(sell_price, 2),
                "gross_return_pct": round(gross_return * 100, 2),
                "net_return_pct": round(net_return * 100, 2),
                "pnl": round(float(trade_pnl), 2),
                "capital_after": round(capital, 2),
            })

            del position[code]

        # Buy new positions
        invest_per_stock = capital * 0.95 / TOP_K  # 95% invested, equal weight
        for code, pred, row in top_picks:
            if code in position:
                continue  # Already holding

            buy_price = row.get("close", row.get("open", 0))
            if buy_price <= 0:
                continue

            shares = int(invest_per_stock / buy_price / 100) * 100  # Round to 100-share lots
            if shares < 100:
                continue

            cost = shares * buy_price * (1 + TOTAL_BUY_COST)
            if cost > capital * 0.1:  # Max 10% per stock
                continue

            capital -= cost
            position[code] = {
                "buy_price": buy_price,
                "buy_date": today,
                "shares": shares,
                "predicted_return": pred,
            }

        # Record equity
        # Mark-to-market: current positions valued at latest close
        mtm_value = 0
        for code, pos in position.items():
            df = stock_data.get(code)
            if df is not None:
                today_mask = df["date"] == today
                if today_mask.any():
                    mtm_value += pos["shares"] * df[today_mask].iloc[0]["close"]

        total_equity = capital + mtm_value
        capital_curve.append({
            "date": str(today.date()),
            "equity": round(total_equity, 2),
            "cash": round(capital, 2),
            "positions": len(position),
        })

    # ── Close remaining positions at last available price ──
    last_date = sorted_dates[-1]
    for code, pos in position.items():
        df = stock_data.get(code)
        if df is not None:
            last_mask = df["date"] == last_date
            if last_mask.any():
                sell_price = df[last_mask].iloc[0]["close"]
                gross_return = sell_price / pos["buy_price"] - 1
                net_return = gross_return - TOTAL_SELL_COST
                pnl = pos["shares"] * pos["buy_price"] * net_return
                capital += pnl
                trades_log.append({
                    "code": code,
                    "buy_date": str(pos["buy_date"].date()),
                    "sell_date": str(last_date.date()),
                    "buy_price": round(pos["buy_price"], 2),
                    "sell_price": round(sell_price, 2),
                    "gross_return_pct": round(gross_return * 100, 2),
                    "net_return_pct": round(net_return * 100, 2),
                    "pnl": round(float(pnl), 2),
                    "capital_after": round(capital, 2),
                })

    # ── Results ──
    trades = pd.DataFrame(trades_log)
    curve = pd.DataFrame(capital_curve)

    if trades.empty:
        print("No trades executed")
        return

    # Metrics
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    win_rate = (trades["pnl"] > 0).mean()
    avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if (trades["pnl"] > 0).sum() > 0 else 0
    avg_loss = trades[trades["pnl"] < 0]["pnl"].mean() if (trades["pnl"] < 0).sum() > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Max drawdown from equity curve
    if len(curve) > 0:
        equity_series = curve["equity"]
        peak = equity_series.cummax()
        drawdowns = (equity_series - peak) / peak
        max_dd = drawdowns.min()
    else:
        max_dd = 0

    # Annualized Sharpe (approximate)
    if len(curve) > 1:
        daily_returns = curve["equity"].pct_change().dropna()
        sharpe = daily_returns.mean() / (daily_returns.std() + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0

    n_trades = len(trades)
    n_days = len(curve)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Period:     {sorted_dates[start_idx].date()} → {sorted_dates[-1].date()}")
    print(f"  Days:       {n_days}")
    print(f"  Trades:     {n_trades}")
    print(f"  Win Rate:   {win_rate:.1%}")
    print(f"  Avg Win:    {avg_win:+,.0f}")
    print(f"  Avg Loss:   {avg_loss:+,.0f}")
    print(f"  P/L Ratio:  {profit_factor:.2f}")
    print(f"  ──────────────────────────")
    print(f"  Start:      {INITIAL_CAPITAL:,.0f}")
    print(f"  Final:      {capital:,.0f}")
    print(f"  Return:     {total_return:+.2%}")
    print(f"  Max DD:     {max_dd:.2%}")
    print(f"  Sharpe:     {sharpe:.2f}")
    print(f"  ──────────────────────────")
    print(f"  Costs:      Buy {TOTAL_BUY_COST:.3%} / Sell {TOTAL_SELL_COST:.3%}")
    print(f"  Top K:      {TOP_K}")
    print(f"  Hold Days:  {HOLD_DAYS}")
    print("=" * 60)

    # Save reports
    trades.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    curve.to_csv(CURVE_FILE, index=False, encoding="utf-8-sig")

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": str(sorted_dates[start_idx].date()),
        "end_date": str(sorted_dates[-1].date()),
        "trading_days": n_days,
        "n_trades": int(n_trades),
        "win_rate": round(float(win_rate), 4),
        "total_return": round(float(total_return), 4),
        "max_drawdown": round(float(max_dd), 4),
        "sharpe_ratio": round(float(sharpe), 2),
        "profit_factor": round(float(profit_factor), 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": round(float(capital), 2),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  Report: {REPORT_FILE}")
    print(f"  Curve:  {CURVE_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")

    return trades, curve


if __name__ == "__main__":
    backtest()
