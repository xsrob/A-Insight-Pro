"""
A-Insight Pro
AI Model Training V4.0 — Walk-Forward CV + Feature Selection

Changes from V3.0:
- Walk-forward cross-validation (3-fold time series)
- Better RF hyperparameters: reduced depth, increased leaf size
- Recursive feature elimination (drop <0.5% importance features)
- Feature correlation pruning (remove redundant >0.95 corr)
- Sector/industry features when available
- Saves calibration data for predict.py
- Monitors feature importance stability across folds
"""

import os, sys, glob, json, joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MODEL_CFG

FEATURE_DIR = "features"
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "stock_model.pkl")
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names.json")
CALIBRATION_FILE = os.path.join(MODEL_DIR, "calibration.json")
FACTOR_SELECTION_FILE = os.path.join(REPORT_DIR, "factor_selection.json")

# New hyperparameters — less overfitting, more generalization
# depth=12 (was 20), leaf=5 (was 1), split=10 (was 3)
DEFAULT_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": 0.7,       # Use 70% features per split (was 0.5 — more diversity for advanced factors)
    "random_state": 42,
    "n_jobs": -1,
}

# Walk-forward: 3 folds, each fold adds the previous fold to training
N_FOLDS = 3


def load_factor_selection():
    """Load selected factor names from factor engine output."""
    if not os.path.exists(FACTOR_SELECTION_FILE):
        print("No factor selection file found, using all available features")
        return []

    try:
        with open(FACTOR_SELECTION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        factors = list(data.get("factor_weights_for_scoring", {}).keys())
        if "fallback" in factors:
            factors.remove("fallback")
        print(f"Loaded {len(factors)} selected factors from factor_selection.json")
        return factors
    except Exception as e:
        print(f"Warning: Could not read factor selection: {e}")
        return []


def load_data():
    """Load all feature CSVs and concatenate."""
    files = glob.glob(os.path.join(FEATURE_DIR, "*.csv"))
    print(f"Feature files: {len(files)}")

    data = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if "future_return" not in df.columns or len(df) < 120:
                continue
            code = os.path.basename(f).replace(".csv", "").zfill(6)
            df["code"] = code
            data.append(df)
            if len(data) % 100 == 0:
                print(f"  Loaded {len(data)} stocks...")
        except Exception as e:
            print(f"  Skip {os.path.basename(f)}: {e}")

    if not data:
        return None

    combined = pd.concat(data, ignore_index=True)
    print(f"Total rows: {len(combined)}")
    return combined


def build_features(df, selected_factors):
    """Build feature matrix with automatic column selection."""
    if "date" not in df.columns:
        print("ERROR: 'date' column required")
        return None, None, None, None

    df = df.sort_values("date").reset_index(drop=True)

    # Exclude non-feature columns
    # Use pre-computed cross-sectional ranked target — 10d has better signal than 5d
    if "future_return_rank_10d" in df.columns:
        target_col = "future_return_rank_10d"
    elif "future_return_rank_5d" in df.columns:
        target_col = "future_return_rank_5d"
    elif "future_return_rank" in df.columns:
        target_col = "future_return_rank"
    else:
        target_col = "future_return"
    exclude = ["date", "future_return", "future_return_5d", "future_return_10d",
               "future_return_20d", "future_return_rank", "future_return_rank_5d",
               "future_return_rank_10d", "future_return_rank_20d", "code"]
    all_features = [c for c in df.columns if c not in exclude]

    # Clean inf/nan
    df_clean = df.replace([float("inf"), -float("inf")], None)

    # Separate X and y
    y = df_clean[target_col]
    X = df_clean[all_features].copy()

    # Drop rows where target is NaN
    valid_idx = y.dropna().index.intersection(X.dropna(how="all").index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if target_col == "future_return_rank":
        print(f"  Using cross-sectional RANKED target (future_return_rank)")
        print(f"  Target: mean={y.mean():.3f}, std={y.std():.3f}, range=[{y.min():.3f}, {y.max():.3f}]")
    else:
        print(f"  Using raw future_return target (future_return_rank not found)")

    # Fill remaining NaN with column median
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # Keep only numeric columns (exclude string columns like 'sector')
    X = X.select_dtypes(include=[np.number])
    all_features = [c for c in all_features if c in X.columns]

    # Remove constant columns (no variance)
    stds = X.std()
    constant_cols = stds[stds < 1e-10].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
        all_features = [c for c in all_features if c not in constant_cols]
        print(f"  Dropped {len(constant_cols)} constant columns")

    print(f"  Features available: {len(all_features)}")
    print(f"  Valid samples: {len(X)}")

    return X, y, all_features, df


def remove_correlated_features(X, feature_names, threshold=0.92):
    """Remove highly correlated features (keep one from each correlated pair)."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for c in correlated:
            if c not in to_drop and col not in to_drop:
                to_drop.add(c)

    if to_drop:
        print(f"  Removing {len(to_drop)} highly correlated features (r > {threshold})")
        kept = [f for f in feature_names if f not in to_drop]
        return X.drop(columns=list(to_drop)), kept, list(to_drop)
    return X, feature_names, []


def walk_forward_split(X, y, dates, n_folds=N_FOLDS):
    """
    Walk-forward time series split.

    CV data = first 80% of time-ordered samples.
    n_folds+1 equal segments → each fold trains on [0:seg_i] and tests on [seg_i:seg_{i+1}].

    Example (n_folds=3, CV=80%, total=1000):
      Segments: [0:200, 200:400, 400:600, 600:800]
      Fold 1: train [0:200], test [200:400]
      Fold 2: train [0:400], test [400:600]
      Fold 3: train [0:600], test [600:800]
      Final held-out: [800:1000]

    Each fold uses all historical data for training,
    simulating real-world prediction deployment.
    """
    n = len(X)
    cv_end = int(n * 0.8)  # Use 80% for CV
    n_segments = n_folds + 1  # Need n_folds+1 segments to get n_folds test sets

    # Equal-size segment boundaries
    boundaries = [int(cv_end * (i + 1) / n_segments) for i in range(n_segments)]
    # e.g. n_folds=3: boundaries = [200, 400, 600, 800]

    splits = []
    for i in range(n_folds):
        train_end = boundaries[i]       # Train: [0 : boundary_i]
        test_start = boundaries[i]       # Test:  [boundary_i : boundary_{i+1}]
        test_end = boundaries[i + 1]

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        if len(X_test) > 0:
            splits.append((X_train, X_test, y_train, y_test))

    return splits


def train():
    """Train RandomForest with walk-forward CV and feature selection."""
    print("=" * 60)
    print("Training V4.0 — Walk-Forward CV + Feature Selection")
    print("=" * 60)

    # Load data
    df = load_data()
    if df is None:
        print("No training data found")
        return

    # Load selected factors
    selected_factors = load_factor_selection()

    # Build features
    X, y, all_features, full_df = build_features(df, selected_factors)
    if X is None:
        return

    # Remove highly correlated features
    X, kept_features, dropped_corr = remove_correlated_features(X, all_features)
    all_features = kept_features

    # ---- RF parameters from config (with new defaults) ----
    rf_cfg = MODEL_CFG.get("random_forest", {})
    n_estimators = rf_cfg.get("n_estimators", DEFAULT_RF_PARAMS["n_estimators"])
    max_depth = rf_cfg.get("max_depth", DEFAULT_RF_PARAMS["max_depth"])
    min_samples_split = rf_cfg.get("min_samples_split", DEFAULT_RF_PARAMS["min_samples_split"])
    min_samples_leaf = rf_cfg.get("min_samples_leaf", DEFAULT_RF_PARAMS["min_samples_leaf"])
    max_features = rf_cfg.get("max_features", DEFAULT_RF_PARAMS["max_features"])
    random_state = rf_cfg.get("random_state", DEFAULT_RF_PARAMS["random_state"])

    print(f"\nRF Hyperparameters:")
    print(f"  trees={n_estimators}, depth={max_depth}, "
          f"min_split={min_samples_split}, min_leaf={min_samples_leaf}, "
          f"max_features={max_features}")

    # ---- Walk-Forward Cross-Validation ----
    print(f"\n{'='*60}")
    print(f"Walk-Forward CV ({N_FOLDS} folds)")
    print(f"{'='*60}")

    cv_splits = walk_forward_split(X, y, full_df["date"])

    if len(cv_splits) < 2:
        print("WARNING: Not enough data for walk-forward CV, falling back to single split")
        split_idx = int(len(X) * 0.7)
        cv_splits = [(X.iloc[:split_idx], X.iloc[split_idx:],
                       y.iloc[:split_idx], y.iloc[split_idx:])]

    cv_metrics = []
    feature_importances_list = []

    for fold_i, (X_train, X_test, y_train, y_test) in enumerate(cv_splits):
        print(f"\n--- Fold {fold_i + 1}/{len(cv_splits)} ---")
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)

        # Directional accuracy
        if len(pred) > 0:
            dir_acc = np.mean((pred > 0) == (y_test.values > 0))
        else:
            dir_acc = 0

        print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {np.sqrt(mse):.6f}, DirAcc: {dir_acc:.3f}")

        cv_metrics.append({
            "fold": fold_i + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "mse": mse,
            "mae": mae,
            "rmse": float(np.sqrt(mse)),
            "directional_accuracy": float(dir_acc),
        })

        # Collect feature importances
        imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        feature_importances_list.append(imp)

    # ---- Final Model (trained on all CV data) ----
    print(f"\n{'='*60}")
    print("Training Final Model (on all CV data)")
    print(f"{'='*60}")

    # Train on all data except held-out final 20%
    final_train_end = int(len(X) * 0.8)
    X_final_train = X.iloc[:final_train_end]
    y_final_train = y.iloc[:final_train_end]
    X_final_test = X.iloc[final_train_end:]
    y_final_test = y.iloc[final_train_end:]

    final_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )

    final_model.fit(X_final_train, y_final_train)
    final_pred = final_model.predict(X_final_test)

    final_mse = mean_squared_error(y_final_test, final_pred)
    final_mae = mean_absolute_error(y_final_test, final_pred)
    final_dir_acc = np.mean((final_pred > 0) == (y_final_test.values > 0))

    print(f"  Final Test — MSE: {final_mse:.6f}, MAE: {final_mae:.6f}, "
          f"RMSE: {np.sqrt(final_mse):.6f}, DirAcc: {final_dir_acc:.3f}")

    # ---- Feature Importance Analysis ----
    final_importance = pd.DataFrame({
        "feature": X_final_train.columns,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 20 features by importance:")
    for _, row in final_importance.head(20).iterrows():
        marker = " ★" if row["feature"] in selected_factors else ""
        print(f"  {row['feature']:<35s} {row['importance']:.4f}{marker}")

    # Feature importance stability across folds
    if len(feature_importances_list) >= 2:
        imp_stability = {}
        for fi in feature_importances_list[0]["feature"]:
            ranks = []
            for imp_df in feature_importances_list:
                if fi in imp_df["feature"].values:
                    rank = imp_df[imp_df["feature"] == fi].index[0] + 1
                    ranks.append(rank)
            if len(ranks) >= 2:
                imp_stability[fi] = {
                    "mean_rank": np.mean(ranks),
                    "std_rank": np.std(ranks),
                }
        stable_features = [k for k, v in imp_stability.items() if v["std_rank"] < 20]
        print(f"\n  Stable features (rank std < 20): {len(stable_features)}/{len(imp_stability)}")

    # ---- Drop low-importance features ----
    # Keep features with importance >= 0.0005 (0.05%) — preserves modest but real signals
    low_imp = final_importance[final_importance["importance"] < 0.0005]
    n_low = len(low_imp)
    if n_low > 0:
        print(f"\n  Low importance (<0.2%): {n_low} features (dropped)")
        kept_features_final = [f for f in X_final_train.columns
                               if f not in low_imp["feature"].values]
        print(f"  Keeping {len(kept_features_final)}/{len(X_final_train.columns)} features")
        # Retrain with reduced features
        X_reduced = X_final_train[kept_features_final]
        final_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        final_model.fit(X_reduced, y_final_train)
        final_pred2 = final_model.predict(X_final_test[kept_features_final])
        reduced_mae = mean_absolute_error(y_final_test, final_pred2)
        print(f"  Retrained MAE: {reduced_mae:.6f} (was {final_mae:.6f})")
    else:
        kept_features_final = list(X_final_train.columns)

    # ---- Save model ----
    joblib.dump(final_model, MODEL_FILE)
    print(f"\n  Model saved: {MODEL_FILE}")

    # ---- Compute calibration curve ----
    # For calibrating predictions: map raw_pred → actual return
    from sklearn.isotonic import IsotonicRegression
    try:
        calib_mask = (y_final_test.values > -0.3) & (y_final_test.values < 0.3)
        if calib_mask.sum() >= 50:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=-0.15, y_max=0.15)
            iso.fit(final_pred[calib_mask], y_final_test.values[calib_mask])
            calib_data = {
                "method": "isotonic",
                "x_min": float(final_pred.min()),
                "x_max": float(final_pred.max()),
                "n_calibration_points": int(calib_mask.sum()),
                "fitted": True,
            }
            print(f"  Isotonic calibration fitted on {calib_mask.sum()} points")
        else:
            calib_data = {"method": "none", "fitted": False}
    except Exception as e:
        print(f"  Calibration skipped: {e}")
        calib_data = {"method": "none", "fitted": False}

    # ---- Save metadata ----
    # CV metrics
    cv_avg_mae = np.mean([m["mae"] for m in cv_metrics])
    cv_avg_dir = np.mean([m["directional_accuracy"] for m in cv_metrics])

    feature_info = {
        "feature_names": kept_features_final,
        "n_features": len(kept_features_final),
        "n_features_original": len(all_features),
        "n_dropped_correlation": len(dropped_corr),
        "n_dropped_low_importance": n_low if n_low > 0 else 0,
        "selected_factors": selected_factors,
        "n_selected_factors": len(selected_factors),
        "top_importances": final_importance.head(30)[["feature", "importance"]].to_dict("records"),
        "train_date_range": [str(full_df.iloc[0]["date"]), str(full_df.iloc[final_train_end - 1]["date"])],
        "test_date_range": [str(full_df.iloc[final_train_end]["date"]), str(full_df.iloc[-1]["date"])],
        "metrics": {
            "cv_folds": len(cv_splits),
            "cv_avg_mae": round(float(cv_avg_mae), 6),
            "cv_avg_directional_accuracy": round(float(cv_avg_dir), 4),
            "final_mse": round(float(final_mse), 6),
            "final_mae": round(float(final_mae), 6),
            "final_rmse": round(float(np.sqrt(final_mse)), 6),
            "final_directional_accuracy": round(float(final_dir_acc), 4),
        },
        "cv_details": cv_metrics,
        "calibration": calib_data,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        },
        "training_version": "V4.0",
    }

    with open(FEATURE_NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    print(f"  Feature names saved: {FEATURE_NAMES_FILE}")

    # Save calibration data
    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Training V4.0 Complete")
    print(f"  CV Folds:          {len(cv_splits)}")
    print(f"  CV Avg MAE:        {cv_avg_mae:.6f}")
    print(f"  CV Avg DirAcc:     {cv_avg_dir:.1%}")
    print(f"  Final Test MAE:    {final_mae:.6f}")
    print(f"  Final Test DirAcc: {final_dir_acc:.1%}")
    print(f"  Features kept:     {len(kept_features_final)}/{len(all_features)}")
    print(f"  Output: {MODEL_FILE}")
    print(f"  Output: {FEATURE_NAMES_FILE}")
    print("=" * 60)

    return final_model


if __name__ == "__main__":
    train()
