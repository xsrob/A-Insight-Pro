"""
A-Insight Pro - AI Scoring V9.0 - Calibrated + Regime-Aware

Changes from V8.0:
- Regime-aware position sizing from market_regime.csv
- Uses RF uncertainty from predict.py for risk adjustment
- Simplified factor confidence computation
- Better market state integration
"""

import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import REPORT_DIR, get_level_thresholds, SCORING_CFG
os.makedirs(REPORT_DIR, exist_ok=True)

PREDICT_FILE = os.path.join(REPORT_DIR, 'ai_stock_report.csv')
LEARNING_FILE = os.path.join(REPORT_DIR, 'ai_learning_feedback.csv')
EMOTION_FILE = os.path.join(REPORT_DIR, 'market_emotion.csv')
REGIME_FILE = os.path.join(REPORT_DIR, 'market_regime.csv')
FACTOR_SELECTION_FILE = os.path.join(REPORT_DIR, 'factor_selection.json')
OUTPUT_FILE = os.path.join(REPORT_DIR, 'final_stock_rank.csv')

# Load factor weights
FACTOR_WT = {}
FACTOR_IC_MAP = {}
if os.path.exists(FACTOR_SELECTION_FILE):
    try:
        with open(FACTOR_SELECTION_FILE, 'r', encoding='utf-8') as f:
            fdata = json.load(f)
        FACTOR_WT = fdata.get('factor_weights_for_scoring', {})
        FACTOR_IC_MAP = fdata.get('factor_ic_map', {})
        if FACTOR_WT.get('fallback') == 1.0:
            FACTOR_WT = {}
    except Exception:
        pass


def load_csv(path):
    if os.path.exists(path):
        try: return pd.read_csv(path, encoding='utf-8-sig')
        except: pass
    return pd.DataFrame()


def get_market_state():
    """Get market emotion + smart money state."""
    df = load_csv(EMOTION_FILE)
    if df.empty: return 50, 'unknown', 20, 'unknown', 'LOW'
    last = df.iloc[-1]
    return (
        float(last.get('market_emotion', 50)),
        str(last.get('level', 'unknown')),
        int(last.get('suggested_position_pct', 20)),
        str(last.get('smart_money_signal', 'unknown')),
        str(last.get('smart_money_activity', 'LOW')),
    )


def get_market_regime():
    """Get market regime for position sizing."""
    df = load_csv(REGIME_FILE)
    if df.empty: return 'UNKNOWN', 30
    last = df.iloc[-1]
    regime = str(last.get('regime', 'UNKNOWN'))
    pos_pct = int(last.get('suggested_position_pct', 30))
    return regime, pos_pct


def get_level(s):
    t = get_level_thresholds()
    if s >= t.get('A_plus', 85): return 'A+'
    elif s >= t.get('A', 70): return 'A'
    elif s >= t.get('B', 55): return 'B'
    return 'C'


def get_signal(score, market_score):
    t = get_level_thresholds()
    offset = 5 if market_score < 30 else (-5 if market_score > 60 else 0)
    adj = score - offset
    r = SCORING_CFG.get('ratings', {})
    if adj >= t.get('A_plus', 85): return r.get('A_plus', 'STRONG_BUY')
    elif adj >= t.get('A', 70): return r.get('A', 'BUY')
    elif adj >= t.get('B', 55): return r.get('B', 'WATCH')
    return r.get('C', 'AVOID')


def get_position(score, market_score, regime_pos_pct):
    """Position sizing: regime-aware with score scaling."""
    t = get_level_thresholds()
    if score >= t.get('A_plus', 85): sw = 1.0
    elif score >= t.get('A', 70): sw = 0.7
    elif score >= t.get('B', 55): sw = 0.4
    else: sw = 0

    # Base position from regime, scaled by score weight
    return round(sw * regime_pos_pct / 100 * 10, 1)


def get_risk(score, rf_uncertainty=0):
    """Risk parameters based on score and model uncertainty."""
    t = get_level_thresholds()
    if score >= t.get('A', 70): sl, mx = -8.0, 10.0
    elif score >= t.get('B', 55): sl, mx = -5.0, 5.0
    else: sl, mx = -3.0, 2.0

    # Tighten stops when model is highly uncertain (>75th percentile)
    if rf_uncertainty > 0.20:
        sl = max(sl, -3.0)
        mx = min(mx, 5.0)
    elif rf_uncertainty > 0.12:
        sl = max(sl, -5.0)
        mx = min(mx, 7.0)

    return round(sl, 1), round(mx, 1)


def scoring():
    print('=' * 60)
    print('AI Scoring V9.0 — Calibrated + Regime-Aware')
    print('=' * 60)

    pred = load_csv(PREDICT_FILE)
    learning = load_csv(LEARNING_FILE)
    if pred.empty: print('No prediction data'); return

    for df in [pred, learning]:
        if not df.empty and 'code' in df.columns:
            df['code'] = df['code'].astype(str).str.zfill(6)

    if 'predict_percent' not in pred.columns:
        pred['predict_percent'] = pred.get('predict_return', pd.Series(0)) * 100

    # ── Market state ──
    mkt_score, mkt_level, mkt_pos, smart_signal, smart_activity = get_market_state()
    regime, regime_pos = get_market_regime()
    smart_tag = f' | 主力: {smart_signal}[{smart_activity}]' if smart_signal != 'unknown' else ''
    print(f'Market: {mkt_level} ({mkt_score:.0f}) | Regime: {regime} | Position: {regime_pos}%{smart_tag}')

    # ── Base Score ──
    # For rank-based models, predictions are compressed ([-3%, +3%] vs [-10%, +15%]).
    # Use wider spread to differentiate stocks in the narrow prediction range.
    # Rank model: ~2% pred → score change of 2*8=16 points. Spread=8.
    # Return model: ~10% pred → score change of 10*3.5=35 points. Spread=3.5.
    pred_range = pred['predict_percent'].max() - pred['predict_percent'].min()
    if pred_range < 6.0:
        BASE_SCORE_SPREAD = 8.0  # Rank model — wider spread for narrow predictions
        print(f'  Rank model detected (range={pred_range:.1f}%), using spread={BASE_SCORE_SPREAD}')
    else:
        BASE_SCORE_SPREAD = 3.5  # Return model

    # Factor confidence (simplified, relies on feature columns from predict.py)
    pred['factor_confidence'] = 0.0

    # Use RF uncertainty if available (V8.0 predict)
    has_uncertainty = 'rf_uncertainty' in pred.columns

    # Compute per-stock factor confidence from available features
    import glob as _glob
    _feat_dir = 'features'
    _stock_rows = {}
    for _fname in _glob.glob(os.path.join(_feat_dir, '*.csv')):
        _code = os.path.basename(_fname).replace('.csv', '').zfill(6)
        try:
            _df = pd.read_csv(_fname, encoding='utf-8-sig')
            if len(_df) > 0:
                _stock_rows[_code] = _df.iloc[-1]
        except Exception:
            pass

    for idx, row in pred.iterrows():
        code = str(row.get('code', '')).zfill(6)
        stock_row = _stock_rows.get(code)
        if stock_row is None:
            continue

        signals = []
        try:
            close_v = float(stock_row.get('close', 0))
            ma20_v = float(stock_row.get('ma20', close_v))
            if close_v > 0 and ma20_v > 0:
                trend = (close_v - ma20_v) / ma20_v * 100
                signals.append(np.clip(trend, -15, 15) * 0.06)

            ma5_v = float(stock_row.get('ma5', 0))
            ma10_v = float(stock_row.get('ma10', 0))
            if ma5_v > 0 and ma10_v > 0:
                aligned = 1.0 if ma5_v > ma10_v else -1.0
                signals.append(aligned * 0.5)

            rsi_v = float(stock_row.get('rsi', 50))
            if rsi_v > 70:
                signals.append(-1.0)
            elif rsi_v < 30:
                signals.append(1.0)
            elif rsi_v > 55:
                signals.append(0.5)
            elif rsi_v < 45:
                signals.append(-0.5)

            vol_chg_v = float(stock_row.get('volume_change', 0))
            signals.append(np.clip(vol_chg_v, -0.8, 0.8) * 0.8)

            vol_v = float(stock_row.get('volatility', 0.01))
            if vol_v > 0.05:
                signals.append(-0.8)

            macd_v = float(stock_row.get('macd', 0))
            if close_v > 0:
                macd_norm = macd_v / close_v * 100
                signals.append(np.clip(macd_norm, -3, 3) * 0.3)

            # Sector relative strength if available
            if 'sector_relative_strength' in stock_row.index:
                srs = float(stock_row.get('sector_relative_strength', 0))
                signals.append(np.clip(srs, -0.1, 0.1) * 3.0)
        except (ValueError, TypeError, KeyError):
            pass

        if signals:
            conf = sum(signals)
            pred.loc[idx, 'factor_confidence'] = round(np.clip(conf, -3, 3), 2)

    pred['base_score'] = (50 + pred['predict_percent'] * BASE_SCORE_SPREAD).clip(0, 100)

    # ── Learning penalty ──
    lc = SCORING_CFG.get('learning', {})
    pred['learning_penalty'] = 0.0
    if not learning.empty and 'success_rate' in learning.columns:
        lmap = {}
        for code, grp in learning.groupby('code'):
            lmap[code] = {
                'sr': grp['success_rate'].mean() if 'success_rate' in grp.columns else 0.5,
                'ae': grp['avg_error'].mean() if 'avg_error' in grp.columns else 0
            }
        for idx, row in pred.iterrows():
            info = lmap.get(row['code'])
            if info is None: continue
            penalty = 0
            if info['sr'] < lc.get('low_success_threshold', 0.3): penalty += lc.get('penalty_magnitude', -8)
            if info['ae'] < lc.get('overestimate_threshold', -5): penalty += lc.get('penalty_magnitude', -8)
            pred.loc[idx, 'learning_penalty'] = max(lc.get('min_penalty', -10),
                                                     min(lc.get('max_penalty', 5), penalty))

    # ── Uncertainty penalty (adaptive thresholds) ──
    pred['uncertainty_penalty'] = 0.0
    if has_uncertainty:
        unc_values = pred['rf_uncertainty'].dropna()
        if len(unc_values) > 0:
            p50 = unc_values.quantile(0.50)
            p75 = unc_values.quantile(0.75)
            for idx, row in pred.iterrows():
                unc = float(row.get('rf_uncertainty', 0))
                if unc > p75:
                    pred.loc[idx, 'uncertainty_penalty'] = -6   # Top 25% most uncertain
                elif unc > p50:
                    pred.loc[idx, 'uncertainty_penalty'] = -3   # Top half

    # ── Market adjust ──
    pred['market_adjust'] = round((mkt_score - 50) * 0.1, 1)

    # ── Final score ──
    pred['AI_SCORE'] = (pred['base_score'] + pred['learning_penalty'] +
                         pred['market_adjust'] + pred['factor_confidence'] +
                         pred['uncertainty_penalty']).clip(0, 100).round(2)

    # ── Signals + Position + Risk ──
    pred['LEVEL'] = pred['AI_SCORE'].apply(get_level)
    pred['SIGNAL'] = pred.apply(lambda r: get_signal(r['AI_SCORE'], mkt_score), axis=1)
    pred['POSITION_PCT'] = pred.apply(lambda r: get_position(r['AI_SCORE'], mkt_score, regime_pos), axis=1)

    # Risk with uncertainty consideration
    for idx, row in pred.iterrows():
        unc = float(row.get('rf_uncertainty', 0)) if has_uncertainty else 0
        sl, mx = get_risk(row['AI_SCORE'], unc)
        pred.loc[idx, 'STOP_LOSS'] = sl
        pred.loc[idx, 'MAX_SINGLE'] = mx

    # ── Filter ST stocks ──
    if 'name' in pred.columns:
        pred['name'] = pred['name'].fillna('')
        st_mask = pred['name'].str.contains('ST', na=False)
        n_st = st_mask.sum()
        if n_st > 0:
            print(f'  Filtered {n_st} ST stocks')
            pred.loc[st_mask, 'AI_SCORE'] = (pred.loc[st_mask, 'AI_SCORE'] * 0.5).clip(0, 100)
            pred.loc[st_mask, 'LEVEL'] = 'C'
            pred.loc[st_mask, 'SIGNAL'] = 'AVOID'
            pred.loc[st_mask, 'POSITION_PCT'] = 0.0

    # ── Sort & Rank ──
    pred = pred.sort_values(['AI_SCORE', 'predict_percent'], ascending=[False, False]).reset_index(drop=True)
    if 'rank' in pred.columns: pred = pred.drop(columns=['rank'])
    pred.insert(0, 'rank', range(1, len(pred) + 1))

    # ── Get stock names ──
    if 'name' not in pred.columns or pred['name'].isna().all():
        try:
            from ai.stock_info import fetch_stock_info
            info = fetch_stock_info()
            pred['name'] = pred['code'].map(lambda c: info.get(str(c).zfill(6), {}).get('name', ''))
        except Exception:
            pred['name'] = ''

    # ── Output ──
    out_cols = ['rank', 'code', 'name', 'predict_percent', 'base_score', 'factor_confidence',
                'rf_uncertainty', 'uncertainty_penalty',
                'AI_SCORE', 'LEVEL', 'SIGNAL',
                'POSITION_PCT', 'STOP_LOSS', 'MAX_SINGLE',
                'learning_penalty', 'market_adjust']
    available = [c for c in out_cols if c in pred.columns]
    result = pred[available].copy()
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    # Simplified output for dashboard
    dashboard_cols = ['rank', 'code', 'predict_percent', 'AI_SCORE', 'LEVEL', 'SIGNAL']
    dash_available = [c for c in dashboard_cols if c in pred.columns]
    pred[dash_available].to_csv(
        os.path.join(REPORT_DIR, 'ai_prediction.csv'), index=False, encoding='utf-8-sig')

    print(f'Output: {OUTPUT_FILE} ({len(result)} stocks)')
    print(f'  Regime: {regime} (position target: {regime_pos}%)')
    if FACTOR_WT:
        print(f'  Factor weights active: {len(FACTOR_WT)} factors')
    if has_uncertainty:
        p50 = result['rf_uncertainty'].quantile(0.50) if len(result) > 0 else 0.1
        high_unc = (result['rf_uncertainty'] > p50).sum()
        print(f'  Above-median uncertainty (>{p50:.3f}): {high_unc} stocks')
    for lvl in ['A+', 'A', 'B', 'C']:
        n = (result['LEVEL'] == lvl).sum()
        sig = result[result['LEVEL'] == lvl]['SIGNAL'].iloc[0] if n > 0 else '-'
        print(f'  {lvl}: {n} stocks | Signal: {sig}')
    print(f'Top 5:')
    for _, r in result.head(5).iterrows():
        fc = r.get('factor_confidence', 0)
        unc = r.get('rf_uncertainty', 0)
        print(f'  #{int(r["rank"])} {r["code"]} {r.get("name","")} | '
              f'Pred:{r["predict_percent"]:+.2f}% | '
              f'Score:{r["AI_SCORE"]:.0f} | {r["SIGNAL"]} | '
              f'FConf:{fc:+.1f} | Unc:{unc:.3f} | '
              f'Stop:{r["STOP_LOSS"]}% | Pos:{r["POSITION_PCT"]}%')
    print('=' * 60)
    return result


if __name__ == '__main__':
    scoring()
