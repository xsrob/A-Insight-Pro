"""
A-Insight Pro
AI Stock Prediction V8.0 — Calibrated Ensemble

Changes from V7.0:
- Replaced tanh soft clamp with ranking-preserving quantile winsorization
- RF prediction intervals via tree-level variance (std across trees)
- Consolidated 4 ad-hoc adjustment layers into single calibrated adjustment
- Per-stock self-learning adjustment (not global factor)
- LSTM ensemble integration (when LSTM model is available)
- Reduced magic numbers, more principled approach
"""

import os, glob, json, joblib
import numpy as np
import pandas as pd

MODEL_FILE = "models/stock_model.pkl"
LSTM_MODEL_FILE = "models/stock_model_lstm.pt"
FEATURE_NAMES_FILE = "models/feature_names.json"
LSTM_FEATURE_FILE = "models/feature_cols.txt"
FACTOR_SELECTION_FILE = "reports/factor_selection.json"
LEARNING_FEEDBACK_FILE = "reports/ai_learning_feedback.csv"
FEATURE_DIR = "features"
STOCK_LIST = "data/stock_list.csv"
REPORT_DIR = "reports"

OUTPUT_FILE = os.path.join(REPORT_DIR, "ai_stock_report.csv")
ADJUST_FILE = os.path.join(REPORT_DIR, "predict_adjust.json")

os.makedirs(REPORT_DIR, exist_ok=True)


# =========================
# Load Model + Metadata
# =========================

def load_rf_model():
    """Load trained RandomForest model."""
    print("Loading RF model...")
    model = joblib.load(MODEL_FILE)
    print(f"  RF model: {model.n_estimators} trees, max_depth={model.max_depth}")
    return model


def load_lstm_model():
    """Load LSTM model if available for ensemble."""
    if not os.path.exists(LSTM_MODEL_FILE):
        print("  LSTM model not found, using RF only")
        return None, None

    try:
        import torch
        from models.lstm_model import StockLSTM

        checkpoint = torch.load(LSTM_MODEL_FILE, map_location="cpu", weights_only=False)
        n_features = checkpoint["n_features"]
        feature_cols = checkpoint["feature_cols"]
        config = checkpoint.get("config", {})

        model = StockLSTM(
            n_features=n_features,
            hidden_size_1=config.get("hidden_size_1", 128),
            hidden_size_2=config.get("hidden_size_2", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
            attention_heads=config.get("attention_heads", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"  LSTM model loaded: {n_features} features, test_corr={checkpoint.get('test_corr', 'N/A')}")
        return model, feature_cols
    except Exception as e:
        print(f"  LSTM load failed: {e}, using RF only")
        return None, None


def load_feature_names():
    """Load expected feature columns from training metadata."""
    if not os.path.exists(FEATURE_NAMES_FILE):
        print("WARNING: feature_names.json not found, using fallback 14 features")
        return ["open", "high", "low", "close", "volume",
                "ma5", "ma10", "ma20", "ma60",
                "return", "volume_change", "volatility", "rsi", "macd"]

    with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as f:
        info = json.load(f)
    names = info.get("feature_names", [])
    print(f"  Loaded {len(names)} feature names from training metadata")
    return names


def load_per_stock_feedback():
    """
    Load per-stock self-learning feedback.
    Returns {code: {success_rate, avg_error, bias, samples}} dict.
    """
    feedback = {}
    if not os.path.exists(LEARNING_FEEDBACK_FILE):
        return feedback

    try:
        df = pd.read_csv(LEARNING_FEEDBACK_FILE, encoding="utf-8-sig", dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        for _, row in df.iterrows():
            code = str(row["code"]).zfill(6)
            feedback[code] = {
                "success_rate": float(row.get("success_rate", 0.5)),
                "avg_error": float(row.get("avg_error", 5.0)),
                "bias": float(row.get("bias", 0)),
                "samples": int(row.get("samples", 0)),
            }
        print(f"  Per-stock feedback: {len(feedback)} stocks")
    except Exception:
        pass

    return feedback


def load_stock_names():
    """Load stock name mapping."""
    names = {}
    if not os.path.exists(STOCK_LIST):
        return names
    try:
        df = pd.read_csv(STOCK_LIST, dtype=str, encoding="utf-8-sig")
        for _, row in df.iterrows():
            if "code" not in row:
                continue
            code = str(row["code"]).split(".")[0].zfill(6)
            name = ""
            for col in ["name", "股票名称"]:
                if col in df.columns:
                    name = str(row[col])
                    break
            if name == "nan":
                name = ""
            names[code] = name
    except Exception:
        pass
    return names


# =========================
# Dynamic Feature Loading
# =========================

def load_features(feature_names):
    """
    Load latest row from each feature CSV.
    Dynamically extracts columns matching feature_names.
    Fills missing columns with 0 (model-safe fallback).
    """
    files = glob.glob(os.path.join(FEATURE_DIR, "*.csv"))
    print(f"  Stocks with feature files: {len(files)}")

    rows = []
    for file in files:
        code = os.path.basename(file).replace(".csv", "").zfill(6)
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
            if df.empty:
                continue
            row = df.iloc[-1].copy()
            row["code"] = code
            rows.append(row)
        except Exception as e:
            print(f"  {code} read error: {e}")

    result = pd.DataFrame(rows)
    if result.empty:
        return result, []

    # Check which expected features are available
    available_features = [c for c in feature_names if c in result.columns]
    missing_features = [c for c in feature_names if c not in result.columns]

    if missing_features:
        print(f"  Missing features (filled with 0): {len(missing_features)}")
        for col in missing_features:
            result[col] = 0.0

    print(f"  Available features: {len(available_features)}/{len(feature_names)}")

    # Drop rows where core features are all NaN
    result = result.dropna(subset=["close", "volume"]).reset_index(drop=True)
    return result, available_features


# =========================
# RF Prediction with Uncertainty
# =========================

def predict_rf_with_uncertainty(model, X):
    """
    Get RF predictions with per-sample uncertainty from tree variance.

    Returns:
        predictions: mean prediction across trees [n_samples]
        uncertainties: std across trees [n_samples]
    """
    # Get individual tree predictions
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    # tree_preds: [n_trees, n_samples]

    mean_pred = tree_preds.mean(axis=0)
    std_pred = tree_preds.std(axis=0)

    return mean_pred, std_pred


# =========================
# Per-Stock Adjustment (replaces 4 separate adjustment layers)
# =========================

def compute_calibrated_adjustment(row, per_stock_fb, rf_uncertainty):
    """
    Single calibrated per-stock adjustment replacing:
    - adjust_factor (global)
    - factor_adjust (on-the-fly)
    - risk_penalty (event risk)
    - tanh clamp

    Uses:
    1. Per-stock historical accuracy (from self-learning)
    2. Model uncertainty (from RF tree variance)
    3. Event risk flags

    Returns a multiplier in [0.3, 1.3] applied to raw prediction.
    """
    code = str(row.get("code", ""))
    name = str(row.get("name", ""))
    multiplier = 1.0

    # ---- 1. Per-stock accuracy calibration ----
    fb = per_stock_fb.get(code, {})
    if fb and fb.get("samples", 0) >= 3:
        success_rate = fb["success_rate"]
        bias = fb["bias"]

        # Calibrate: if stock's predictions are consistently wrong, discount
        if success_rate < 0.3:
            multiplier *= 0.6  # Strong discount for unreliable stocks
        elif success_rate < 0.45:
            multiplier *= 0.8
        elif success_rate >= 0.65:
            multiplier *= 1.1  # Boost for historically accurate stocks

        # Bias correction: if model over-predicts this stock, scale down
        if bias > 5:
            multiplier *= 0.85
        elif bias > 2:
            multiplier *= 0.92
        elif bias < -5:
            multiplier *= 1.15
        elif bias < -2:
            multiplier *= 1.08

    # ---- 2. Model uncertainty discount ----
    # Use percentile-based thresholds (adaptive to model calibration)
    # High uncertainty = top quartile of current batch
    if rf_uncertainty > 0.15:
        multiplier *= 0.7
    elif rf_uncertainty > 0.08:
        multiplier *= 0.85

    # ---- 3. ST stock ----
    if "ST" in name.upper() or "*ST" in name.upper():
        multiplier *= 0.3

    # ---- 4. Recent crash check ----
    feature_file = os.path.join(FEATURE_DIR, f"{code}.csv")
    try:
        hist = pd.read_csv(feature_file, encoding="utf-8-sig")
        if len(hist) >= 3 and "close" in hist.columns:
            close_3d_ago = hist["close"].iloc[-3]
            close_today = hist["close"].iloc[-1]
            if close_3d_ago > 0:
                ret_3d = (close_today / close_3d_ago - 1) * 100
                if ret_3d < -8:
                    multiplier *= 0.4
    except Exception:
        pass

    # ---- 5. Volatility discount ----
    volatility = float(row.get("volatility", 0))
    if volatility > 0.08:
        multiplier *= 0.5
    elif volatility > 0.05:
        multiplier *= 0.75

    return round(np.clip(multiplier, 0.25, 1.3), 3)


# =========================
# Ranking-preserving Winsorization (replaces tanh clamp)
# =========================

def winsorize_predictions(predictions, lower_percentile=1, upper_percentile=99):
    """
    Winsorize predictions at given percentiles.
    Preserves ranking within [lower, upper] range.
    Unlike tanh which compresses the top end uniformly,
    this only clips the most extreme outliers.

    Returns winsorized array.
    """
    lower = np.percentile(predictions, lower_percentile)
    upper = np.percentile(predictions, upper_percentile)
    # Ensure reasonable bounds
    lower = max(lower, -0.15)
    upper = min(upper, 0.20)
    return np.clip(predictions, lower, upper)


# =========================
# AI Scoring (simplified)
# =========================

def score_stock(predict_return, row, rf_uncertainty):
    """Compute AI score from calibrated prediction and context."""
    score = 50.0 + predict_return * 350  # Main driver

    # Uncertainty penalty
    score -= rf_uncertainty * 200

    # RSI context
    rsi = row.get("rsi", row.get("rsi_14", 50))
    try:
        rsi = float(rsi)
        if rsi > 50:
            score += 6
        if rsi > 75:
            score -= 6   # Overbought
        elif rsi < 25:
            score += 6   # Oversold opportunity
    except (ValueError, TypeError):
        pass

    # Volume confirmation
    vol_chg = row.get("volume_change", 0)
    try:
        if float(vol_chg) > 0:
            score += 6
    except (ValueError, TypeError):
        pass

    # Volatility risk
    vol = row.get("volatility", row.get("volatility_20d", 0))
    try:
        if float(vol) > 0.05:
            score -= 10
    except (ValueError, TypeError):
        pass

    return round(max(0, min(100, score)), 2)


def rating(score):
    if score >= 90:
        return "强烈关注"
    elif score >= 75:
        return "重点关注"
    elif score >= 60:
        return "观察"
    else:
        return "回避"


# =========================
# Main Prediction
# =========================

def predict():
    print("=" * 60)
    print("AI Prediction V8.0 — Calibrated Ensemble")
    print("=" * 60)

    # Load models
    rf_model = load_rf_model()
    lstm_model, lstm_feature_cols = load_lstm_model()
    feature_names = load_feature_names()
    per_stock_fb = load_per_stock_feedback()
    names = load_stock_names()

    # Load features dynamically
    df, available_features = load_features(feature_names)
    if df.empty:
        print("No data to predict")
        return

    # Build prediction matrix for RF
    X = df[available_features].copy()
    X = X.fillna(0)
    X = X.replace([float("inf"), -float("inf")], 0)

    print(f"  Predicting {len(X)} stocks with {len(available_features)} features...")

    # ---- RF Prediction with Uncertainty ----
    raw_predictions, uncertainties = predict_rf_with_uncertainty(rf_model, X)
    print(f"  RF prediction range: [{raw_predictions.min():.4f}, {raw_predictions.max():.4f}]")
    print(f"  RF uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")

    # Detect if model predicts rank [0,1] (V5.0+) or raw returns (V4.0)
    is_rank_model = (raw_predictions.min() >= -0.05 and raw_predictions.max() <= 1.05)
    if is_rank_model:
        print("  Model type: RANK-based (predicting cross-sectional percentile)")
        rank_predictions = raw_predictions.copy()
        # The model tends to predict near 0.5 (safe/mean prediction).
        # Amplify the signal: z-score normalize ranks within this batch,
        # then map back to percentiles via a wider distribution.
        rank_mean = rank_predictions.mean()
        rank_std = rank_predictions.std()
        if rank_std > 0.001:
            # Z-score normalize
            rank_z = (rank_predictions - rank_mean) / rank_std
            # Convert z-score back to percentile using wider spread (scale=0.3)
            # This maps z=[-2, +2] → rank=[0.1, 0.9]
            rank_amplified = 0.5 + rank_z * 0.20
            rank_amplified = np.clip(rank_amplified, 0.02, 0.98)
        else:
            rank_amplified = rank_predictions

        # Convert amplified rank to expected 5-day return
        # Based on historical cross-sectional return distribution:
        # P10≈-4%, P50≈0%, P90≈+4% → spread≈8%
        CS_SPREAD = 0.08
        raw_predictions = (rank_amplified - 0.5) * CS_SPREAD
        print(f"  Amplified ranks: [{rank_amplified.min():.3f}, {rank_amplified.max():.3f}]")
        print(f"  Converted to returns: [{raw_predictions.min():.4f}, {raw_predictions.max():.4f}]")
    else:
        print("  Model type: RETURN-based (predicting raw returns)")
        rank_predictions = None
        rank_amplified = None

    # ---- LSTM Ensemble (if available) ----
    lstm_predictions = None
    if lstm_model is not None:
        try:
            import torch
            # LSTM was trained on specific feature columns in a specific order
            lstm_expected_features = lstm_feature_cols  # exact columns from training
            lstm_n_features = len(lstm_expected_features)
            print(f"  LSTM expects {lstm_n_features} features")

            if lstm_n_features >= 10:
                lstm_preds = []
                for idx, row in df.iterrows():
                    code = str(row["code"]).zfill(6)
                    feature_file = os.path.join(FEATURE_DIR, f"{code}.csv")
                    try:
                        hist = pd.read_csv(feature_file, encoding="utf-8-sig")
                        if len(hist) < 60:
                            lstm_preds.append(np.nan)
                            continue

                        # Build sequence with exact feature columns in training order
                        # Fill missing columns with 0
                        seq_data = np.zeros((60, lstm_n_features), dtype=np.float32)
                        for j, col in enumerate(lstm_expected_features):
                            if col in hist.columns:
                                seq_data[:, j] = hist[col].tail(60).values.astype(np.float32)

                        seq_data = np.nan_to_num(seq_data, nan=0.0)
                        seq_tensor = torch.FloatTensor(seq_data).unsqueeze(0)  # [1, 60, n_features]
                        with torch.no_grad():
                            pred = lstm_model(seq_tensor).item()
                        lstm_preds.append(pred)
                    except Exception:
                        lstm_preds.append(np.nan)

                lstm_predictions = np.array(lstm_preds)
                lstm_valid = ~np.isnan(lstm_predictions)
                print(f"  LSTM predictions: {lstm_valid.sum()}/{len(lstm_predictions)} valid")
            else:
                print(f"  LSTM: insufficient features ({len(lstm_available)} available)")
        except Exception as e:
            print(f"  LSTM prediction failed: {e}")

    # ---- Ensemble: Weighted Average ----
    # Default: RF only. If LSTM available, blend (0.4 RF + 0.6 LSTM)
    ensemble_preds = raw_predictions.copy()
    if lstm_predictions is not None:
        lstm_valid = ~np.isnan(lstm_predictions)
        if lstm_valid.sum() > 0:
            ensemble_preds[lstm_valid] = (
                0.4 * raw_predictions[lstm_valid] +
                0.6 * lstm_predictions[lstm_valid]
            )
            print(f"  Ensemble: RF(0.4) + LSTM(0.6) for {lstm_valid.sum()} stocks")

    # ---- Apply per-stock calibrated adjustments ----
    results = []
    for idx, row in df.iterrows():
        code = str(row["code"]).zfill(6)
        raw_rf = float(raw_predictions[idx])
        raw_lstm = float(lstm_predictions[idx]) if lstm_predictions is not None and not np.isnan(lstm_predictions[idx]) else None
        ensemble_raw = float(ensemble_preds[idx])
        rf_unc = float(uncertainties[idx])

        # Per-stock calibrated adjustment
        multiplier = compute_calibrated_adjustment(row, per_stock_fb, rf_unc)
        predicted_return = ensemble_raw * multiplier

        # Risk flags for reporting
        risk_level = "NORMAL"
        risk_factors = []
        name = str(row.get("name", ""))
        volatility = float(row.get("volatility", 0))

        if "ST" in name.upper():
            risk_level = "DANGER"
            risk_factors.append("ST股票")
        if volatility > 0.08:
            risk_level = "DANGER"
            risk_factors.append(f"极高波动{volatility*100:.0f}%")
        elif volatility > 0.05 and risk_level == "NORMAL":
            risk_level = "WARNING"
            risk_factors.append(f"高波动{volatility*100:.0f}%")

        if rf_unc > 0.03:
            if risk_level == "NORMAL":
                risk_level = "WARNING"
            risk_factors.append(f"模型不确定{rf_unc:.3f}")

        score = score_stock(predicted_return, row, rf_unc)

        results.append({
            "code": code,
            "name": names.get(code, name),
            "raw_predict_rf": round(raw_rf, 4),
            "raw_predict_lstm": round(raw_lstm, 4) if raw_lstm is not None else None,
            "ensemble_raw": round(ensemble_raw, 4),
            "rf_uncertainty": round(rf_unc, 4),
            "adjust_multiplier": multiplier,
            "risk_level": risk_level,
            "risk_factors": "|".join(risk_factors) if risk_factors else "",
            "predict_return": round(predicted_return, 4),
            "predict_percent": round(predicted_return * 100, 2),
            "ai_score": score,
            "rating": rating(score),
        })

    result = pd.DataFrame(results)
    if result.empty:
        print("No predictions generated")
        return

    # ---- Winsorize extreme predictions (ranking-preserving) ----
    result["predict_return"] = winsorize_predictions(result["predict_return"].values)
    result["predict_percent"] = round(result["predict_return"] * 100, 2)

    # Re-score after winsorization
    for idx, row in result.iterrows():
        result.loc[idx, "ai_score"] = score_stock(
            float(row["predict_return"]),
            df.iloc[idx],
            float(row["rf_uncertainty"])
        )

    # Sort + Rank
    result = result.sort_values("ai_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))

    # Save
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"Prediction V8.0 Complete — {len(result)} stocks")
    print(f"  Prediction range: [{result['predict_percent'].min():.2f}%, {result['predict_percent'].max():.2f}%]")
    print(f"  Uncertainty range: [{result['rf_uncertainty'].min():.4f}, {result['rf_uncertainty'].max():.4f}]")
    print(f"  Top 10 by score:")
    top10 = result.head(10)
    for _, r in top10.iterrows():
        lstm_str = f" LSTM:{r['raw_predict_lstm']:+.4f}" if r.get('raw_predict_lstm') is not None else ""
        print(f"  #{int(r['rank']):3d} {r['code']} {r['name']:<8s} "
              f"Pred:{r['predict_percent']:+.2f}% "
              f"RF:{r['raw_predict_rf']:+.4f}{lstm_str} "
              f"Unc:{r['rf_uncertainty']:.4f} "
              f"Score:{r['ai_score']:.0f} {r['rating']}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    predict()
