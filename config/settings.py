"""
A-Insight Pro
系统配置文件 V2.0
Supports YAML-based configuration
"""

import os
import yaml

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURE_DIR = os.path.join(PROJECT_ROOT, "features")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure directories exist
for d in [DATA_DIR, FEATURE_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# Project info
PROJECT_NAME = "A-Insight Pro"
VERSION = "2.0.0"
DATA_SOURCE = "akshare"
DATABASE_NAME = os.path.join(PROJECT_ROOT, "market_data.db")

# ==========================================
# Load YAML configs
# ==========================================

def _load_yaml(filename):
    """Load YAML config file with fallback to defaults."""
    path = os.path.join(CONFIG_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}

SCORING_CFG = _load_yaml("scoring_config.yaml")
MODEL_CFG = _load_yaml("model_config.yaml")

# --- Convenience accessors ---
def get_weights():
    return SCORING_CFG.get("weights", {
        "predict_percent": 0.35, "win_rate": 0.20,
        "avg_return": 0.15, "total_return": 0.15, "max_drawdown": 0.15
    })

def get_level_thresholds():
    return SCORING_CFG.get("levels", {"A_plus": 85, "A": 70, "B": 55})

def get_backtest_cfg():
    return SCORING_CFG.get("backtest", {
        "top_k": 20, "initial_capital": 1_000_000, "rebalance_freq_days": 5,
        "transaction_cost": 0.001, "slippage": 0.0005
    })

def get_lstm_cfg():
    return MODEL_CFG.get("lstm", {
        "hidden_size_1": 128, "hidden_size_2": 64, "num_layers": 2,
        "bidirectional": True, "dropout": 0.3, "attention_heads": 4,
        "seq_len": 60, "pred_horizon": 5, "batch_size": 64,
        "learning_rate": 0.001, "epochs": 100, "early_stopping_patience": 15
    })

def get_rf_cfg():
    return MODEL_CFG.get("random_forest", {
        "n_estimators": 200, "max_depth": 12, "random_state": 42, "n_jobs": -1
    })

def get_ensemble_cfg():
    return MODEL_CFG.get("ensemble", {
        "method": "weighted_average", "lstm_weight": 0.6, "rf_weight": 0.4
    })
