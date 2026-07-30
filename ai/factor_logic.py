"""
A-Insight Pro
Factor Logic Decomposition & Economic Interpretability V1.0

Problem: GP/ML can discover thousands of mathematical expressions that
correlate with returns, but most are statistical noise without real economic
meaning. This module filters for factors with genuine economic logic.

Method:
  1. Decompose factor expression into constituent parts
  2. Classify factor type (momentum, value, volatility, quality, etc.)
  3. Map to known economic mechanisms (risk premium, behavioral bias, etc.)
  4. Score interpretability on 0-1 scale
  5. Filter: only factors with interpretability > 0.3 survive

Economic logic library:
  - Momentum: short-term persistence (behavioral: under-reaction)
  - Mean-Reversion: over-reaction correction, liquidity provision
  - Volatility: risk compensation, leverage effect
  - Value: price-to-fundamental convergence
  - Quality: flight-to-quality premium
  - Sentiment: noise trader risk
"""

import os, json, re, math
import numpy as np
import pandas as pd
from datetime import datetime

REPORT_DIR = "reports"

# ============================================================
# Economic Logic Rules Engine
# ============================================================

ECONOMIC_MECHANISMS = {
    "momentum_persistence": {
        "pattern": r"(return|mom|pct_change|close\.diff)",
        "description": "价格趋势延续：行为金融中的反应不足效应",
        "logic_score": 0.85,
        "risk_premium": "behavioral",
        "example": "mom_20d, return_10d",
    },
    "mean_reversion": {
        "pattern": r"(ma\d*_ratio|ma\d*_distance|bollinger|price_position|ma_cross)",
        "description": "价格偏离均值后回归：流动性提供者补偿",
        "logic_score": 0.80,
        "risk_premium": "liquidity",
        "example": "ma20_distance, bb_position",
    },
    "volatility_risk": {
        "pattern": r"(volatility|vol_|std|variance|atr)",
        "description": "波动率作为风险度量：高波动→高预期收益或低预期收益(杠杆效应)",
        "logic_score": 0.75,
        "risk_premium": "volatility",
        "example": "vol_20d, atr_ratio",
    },
    "volume_information": {
        "pattern": r"(volume|vol_ratio|amount|turnover|obv|mfi)",
        "description": "成交量蕴含信息：放量→分歧或共识信号",
        "logic_score": 0.70,
        "risk_premium": "information_asymmetry",
        "example": "vol_ratio, amount_ratio",
    },
    "quality_premium": {
        "pattern": r"(sharpe|consistency|calmar|up_down)",
        "description": "质量溢价：高夏普/稳定收益股票获得风险补偿",
        "logic_score": 0.65,
        "risk_premium": "quality",
        "example": "sharpe_proxy, return_consistency",
    },
    "tail_risk": {
        "pattern": r"(skew|kurt|tail|drawdown|max.*dd)",
        "description": "尾部风险定价：极端事件的风险补偿",
        "logic_score": 0.78,
        "risk_premium": "tail_risk",
        "example": "skewness_20d, tail_ratio",
    },
    "sentiment_flow": {
        "pattern": r"(sentiment|hot|search|news|social|tweet|follow)",
        "description": "投资者情绪→短期价格压力→反转",
        "logic_score": 0.60,
        "risk_premium": "noise_trader",
        "example": "news_sentiment_mean, hot_heat_value",
    },
    "capital_flow": {
        "pattern": r"(fund_flow|north_bound|margin|main_net)",
        "description": "资金流向→供需失衡→价格变动",
        "logic_score": 0.72,
        "risk_premium": "demand_pressure",
        "example": "fund_flow_main_net, north_bound_daily",
    },
    "cross_sectional": {
        "pattern": r"(rel_|relative|cross_section|sector|industry)",
        "description": "横截面相对价值：行业内相对定价错误",
        "logic_score": 0.68,
        "risk_premium": "relative_value",
        "example": "rel_mom_20d",
    },
}


def classify_factor_expression(expr_str):
    """
    Classify a factor expression by its economic mechanism.
    Returns list of matching mechanisms with scores.
    """
    matches = []
    expr_lower = expr_str.lower()

    for mech_name, mech_info in ECONOMIC_MECHANISMS.items():
        if re.search(mech_info["pattern"], expr_lower):
            matches.append({
                "mechanism": mech_name,
                "description": mech_info["description"],
                "logic_score": mech_info["logic_score"],
                "risk_premium": mech_info["risk_premium"],
            })

    return matches


def compute_interpretability(expr_str, components=None):
    """
    Score a factor expression for economic interpretability (0-1).

    Criteria:
      - Maps to known economic mechanism: +0.4 to +0.85
      - Expression simplicity (Occam's razor): +0.1 to +0.2
      - Logical direction (sign makes sense): +0.1
      - Domain-specific (not generic noise): +0.05
    """
    score = 0.0
    reasons = []

    # 1. Economic mechanism match (max 0.6)
    matches = classify_factor_expression(expr_str)
    if matches:
        best_score = max(m["logic_score"] for m in matches)
        score += best_score * 0.6
        reasons.append(f"Mechanism: {matches[0]['mechanism']} ({matches[0]['description']})")

    # 2. Simplicity bonus (fewer operations = more interpretable)
    ops = expr_str.count("+") + expr_str.count("-") + expr_str.count("*") + expr_str.count("/")
    if ops <= 2:
        score += 0.20
        reasons.append("Simple expression (+0.20)")
    elif ops <= 5:
        score += 0.12
        reasons.append("Moderate complexity (+0.12)")
    else:
        score += 0.03
        reasons.append("Complex expression (+0.03)")

    # 3. Component count penalty
    if components:
        n_comp = len(set(components))
        if n_comp <= 2:
            score += 0.10
            reasons.append("Few base components (+0.10)")
        elif n_comp <= 4:
            score += 0.05

    # 4. Known pattern bonus
    known_patterns = [
        (r"mom.*\d+d", "Momentum pattern recognized"),
        (r"vol.*\d+d", "Volatility pattern recognized"),
        (r"ma.*ratio", "Mean-reversion pattern recognized"),
        (r"rsi|macd|bollinger", "Technical indicator pattern recognized"),
        (r"sentiment|news|social", "Sentiment factor recognized"),
    ]
    for pattern, reason in known_patterns:
        if re.search(pattern, expr_str.lower()):
            score += 0.03
            break

    return min(score, 1.0), reasons


def decompose_expression(expr_str):
    """
    Decompose a GP-generated expression into its economic building blocks.

    Returns:
      - base_features: list of primitive features used
      - operations: list of mathematical operations
      - structure: description of the expression structure
      - component_count: number of distinct base features
    """
    # Extract feature names (patterns like X0, X1, or named features)
    features = re.findall(r'[a-z_][a-z0-9_]*\d*[a-z]*', expr_str.lower())
    # Filter out math operations
    ops_keywords = {'add', 'sub', 'mul', 'div', 'sqrt', 'abs', 'log', 'sin', 'cos',
                    'neg', 'sign', 'exp', 'pow', 'tan', 'max', 'min', 'clip'}
    base_features = [f for f in features if f not in ops_keywords and len(f) > 2]

    # Count operations
    operations = []
    for op in ['+', '-', '*', '/', 'sqrt', 'abs', 'log', 'sin', 'cos', '^', '**']:
        count = expr_str.count(op)
        if count > 0:
            operations.append(f"{op}({count})")

    structure = f"{len(base_features)} base features, {len(operations)} operation types"

    return {
        "base_features": list(set(base_features)),
        "operations": operations,
        "structure": structure,
        "component_count": len(set(base_features)),
    }


def filter_factors_by_logic(factors, min_interpretability=0.3):
    """
    Filter a list of discovered factors by economic interpretability.

    Args:
        factors: list of dicts with 'expression' key
        min_interpretability: minimum score to keep (0-1)

    Returns:
        validated: factors that pass the logic filter
        rejected: factors that fail
    """
    validated = []
    rejected = []

    for f in factors:
        expr = f.get("expression", str(f))
        components = f.get("components", None)
        interpretability, reasons = compute_interpretability(expr, components)

        result = {
            **f,
            "interpretability": round(interpretability, 3),
            "economic_reasons": reasons,
            "decomposition": decompose_expression(expr),
            "mechanisms": [m["mechanism"] for m in classify_factor_expression(expr)],
        }

        if interpretability >= min_interpretability:
            validated.append(result)
        else:
            rejected.append(result)

    # Sort validated by interpretability * |IC|
    validated.sort(
        key=lambda r: r["interpretability"] * abs(r.get("abs_ic", r.get("ic", 0))),
        reverse=True
    )

    return validated, rejected


def analyze_factor_batch(factors):
    """Analyze a batch of factors and print report."""
    print("=" * 60)
    print("Factor Logic Decomposition & Filtering")
    print("=" * 60)

    validated, rejected = filter_factors_by_logic(factors)

    print(f"\n  Total factors: {len(factors)}")
    print(f"  Validated (logic >0.3): {len(validated)}")
    print(f"  Rejected (noise): {len(rejected)}")

    if validated:
        print(f"\n  Top Validated Factors:")
        print(f"  {'Expr':<45s} {'|IC|':>7s} {'Logic':>6s} {'Mechanism':<25s}")
        print(f"  {'-'*83}")
        for f in validated[:10]:
            ic = abs(f.get("abs_ic", f.get("ic", 0)))
            logic = f.get("interpretability", 0)
            mech = f.get("mechanisms", ["unknown"])[0] if f.get("mechanisms") else "unknown"
            expr_short = f["expression"][:42]
            print(f"  {expr_short:<45s} {ic:7.4f} {logic:6.3f} {mech:<25s}")

    if rejected:
        print(f"\n  Top Rejected (noise):")
        for f in rejected[:3]:
            print(f"    {f['expression'][:60]}... (logic={f.get('interpretability',0):.3f})")

    # Save report
    report = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(factors),
        "validated": len(validated),
        "rejected": len(rejected),
        "validated_factors": validated,
        "rejected_samples": [r["expression"] for r in rejected[:5]],
    }

    path = os.path.join(REPORT_DIR, "factor_logic_report.json")
    with open(path, "w", encoding="utf-8") as f:
        clean = json.loads(json.dumps(report, default=str))
        json.dump(clean, f, ensure_ascii=False, indent=2)

    print(f"\n  Report: {path}")
    return validated, rejected


# ============================================================
# Integration: Filter genetic + event factors
# ============================================================

def filter_all_discovered_factors():
    """
    Load all discovered factors (genetic, event, price/volume)
    and filter them through economic logic. Keep only interpretable ones.
    """
    all_factors = []

    # 1. Genetic factors
    gf_path = os.path.join(REPORT_DIR, "genetic_factors.json")
    if os.path.exists(gf_path):
        with open(gf_path, "r", encoding="utf-8") as f:
            gf = json.load(f)
        for fact in gf.get("top_factors", []):
            all_factors.append({
                **fact,
                "source": "genetic_programming",
            })

    # 2. Factor selection (IC-tested)
    fs_path = os.path.join(REPORT_DIR, "factor_selection.json")
    if os.path.exists(fs_path):
        with open(fs_path, "r", encoding="utf-8") as f:
            fs = json.load(f)
        for fact in fs.get("top_factors", []):
            all_factors.append({
                "expression": fact.get("factor", ""),
                "ic": fact.get("ic_mean", 0),
                "abs_ic": fact.get("abs_ic", 0),
                "source": "ic_analysis",
                "components": [fact.get("factor", "")],
            })

    if not all_factors:
        print("No discovered factors found. Run factor_engine and genetic_factors first.")
        return [], []

    return analyze_factor_batch(all_factors)


if __name__ == "__main__":
    # Demo: classify some example expressions
    test_expressions = [
        "mom_20d * volatility_5d + rsi_14",
        "sqrt(abs(return_1d)) * sign(volume_ratio_20d)",
        "ma20_distance / (vol_20d + 0.01)",
        "sin(X3) * log(abs(X7 + X2)) + X5",  # Pure noise
        "bollinger_pct_b * price_position_20d",
        "random_noise + meaningless_mix * 42",
    ]

    print("Factor Logic Classification Demo:")
    for expr in test_expressions:
        score, reasons = compute_interpretability(expr)
        matches = classify_factor_expression(expr)
        mech = matches[0]["mechanism"] if matches else "none"
        print(f"  [{score:.2f}|{mech:<22s}] {expr}")

    # Load and filter all discovered factors
    print("\n")
    filter_all_discovered_factors()
