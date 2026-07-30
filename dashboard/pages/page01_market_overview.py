"""Market Overview V4.0 — Market Heat + Smart Money Dashboard"""
import streamlit as st, pandas as pd, numpy as np, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT = os.path.join(ROOT, 'reports')
sys.path.insert(0, ROOT)

@st.cache_data(ttl=300)
def load_data():
    data = {}
    for name in ['market_emotion.csv','final_stock_rank.csv','backtest_summary.csv','factor_ic_analysis.csv']:
        p = os.path.join(REPORT, name)
        if os.path.exists(p):
            try: data[name.split('.')[0]] = pd.read_csv(p, encoding='utf-8-sig')
            except: pass
    return data

def show():
    st.title("📊 Market Overview")
    data = load_data()
    emotion = data.get('market_emotion')
    rank = data.get('final_stock_rank')
    summary = data.get('backtest_summary')

    if emotion is not None and not emotion.empty:
        last = emotion.iloc[-1]
        mkt_score = float(last.get('market_emotion', 50))
        heat_score = float(last.get('heat_score', mkt_score))
        smart_score = float(last.get('smart_money_score', 50))
        smart_signal = str(last.get('smart_money_signal', ''))
        smart_activity = str(last.get('smart_money_activity', ''))

        # ── Top KPI Row ──
        st.subheader("Sentiment & Smart Money")

        # Signal color coding
        signal_colors = {
            "主力吸筹": "🟢", "主力出货": "🔴", "主力观望": "🟡",
            "分歧": "🟠", "主力休息": "⚪", "": "⚪",
        }
        signal_icon = signal_colors.get(smart_signal, "⚪")

        cols = st.columns(5)
        cols[0].metric(
            "Market Emotion",
            f"{mkt_score:.0f}",
            delta=f"{last.get('level', '')}",
        )
        cols[1].metric(
            "Smart Money",
            f"{smart_score:.0f}",
            delta=f"{signal_icon} {smart_signal}" if smart_signal else None,
        )
        cols[2].metric(
            "Position",
            f"{int(last.get('suggested_position_pct', 20))}%",
            delta=f"Activity: {smart_activity}" if smart_activity else None,
        )
        cols[3].metric(
            "Accumulating",
            f"{int(last.get('accumulating_count', 0))}",
            delta=f"Distributing: {int(last.get('distributing_count', 0))}",
            delta_color="inverse",
        )
        cols[4].metric(
            "Inst Momentum",
            f"{int(last.get('inst_momentum_count', 0))}",
            delta=f"Abnormal Vol: {int(last.get('abnormal_vol_count', 0))}",
        )

        # ── Two-Panel: Market Heat vs Smart Money ──
        c1, c2 = st.columns(2)

        with c1:
            st.caption("🔥 Market Heat Dimensions")
            heat_dims = ['breadth','strength_depth','volume_momentum','fear_index','trend_alignment','extreme_spread']
            heat_labels = ['Breadth','Strength','Volume','Fear(inv)','Trend','Spread']
            heat_vals = [float(last.get(d, 0)) for d in heat_dims]
            heat_chart = pd.DataFrame({"Dimension": heat_labels, "Score": heat_vals}).set_index("Dimension")
            st.bar_chart(heat_chart, horizontal=True, height=200)

        with c2:
            st.caption("💰 Smart Money Dimensions")
            smart_dims = ['abnormal_volume','accumulation_signal','inst_momentum','smart_flow']
            smart_labels = ['Abnormal Vol','Accumulation','Inst Momentum','Smart Flow']
            smart_vals = [float(last.get(d, 0)) for d in smart_dims]
            smart_chart = pd.DataFrame({"Dimension": smart_labels, "Score": smart_vals}).set_index("Dimension")
            st.bar_chart(smart_chart, horizontal=True, height=200)

        # ── Smart Money Gauge Row ──
        st.subheader(f"{signal_icon} Smart Money Signal: {smart_signal}")
        activity_bar = {
            "HIGH": ("🟢 High — 机构资金高度活跃，方向性信号可靠度高", 1.0),
            "MEDIUM": ("🟡 Medium — 机构活动适中，信号可作为参考", 0.6),
            "LOW": ("⚪ Low — 机构资金低活跃，市场缺乏主力引导", 0.3),
        }.get(smart_activity, ("", 0.0))
        st.info(activity_bar[0])

        # ── Dimension detail table ──
        with st.expander("📋 All 10 Dimensions Detail"):
            dim_data = {
                "Dimension": [
                    "Breadth (%>MA20)", "Strength Depth (%>3%)", "Volume Momentum",
                    "Fear Index (inv)", "Trend Align (5>20)", "Extreme Spread (H/L)",
                    "Abnormal Volume (2x)", "Accumulation Signal", "Inst Momentum (consec)",
                    "Smart Flow Direction",
                ],
                "Type": ["Heat"]*6 + ["Smart Money"]*4,
                "Score": [
                    float(last.get('breadth', 0)),
                    float(last.get('strength_depth', 0)),
                    float(last.get('volume_momentum', 0)),
                    float(last.get('fear_index', 0)),
                    float(last.get('trend_alignment', 0)),
                    float(last.get('extreme_spread', 0)),
                    float(last.get('abnormal_volume', 0)),
                    float(last.get('accumulation_signal', 0)),
                    float(last.get('inst_momentum', 0)),
                    float(last.get('smart_flow', 0)),
                ],
                "Raw Count": [
                    f"{int(last.get('above_ma20_count', 0))} stocks",
                    f"↑{int(last.get('strong_up_count', 0))} ↓{int(last.get('strong_down_count', 0))}",
                    "-",
                    "-",
                    "-",
                    f"H:{int(last.get('near_high', 0))} L:{int(last.get('near_low', 0))}" if 'near_high' in last else "-",
                    f"{int(last.get('abnormal_vol_count', 0))} stocks",
                    f"吸{int(last.get('accumulating_count', 0))} 出{int(last.get('distributing_count', 0))}",
                    f"{int(last.get('inst_momentum_count', 0))} stocks",
                    "-",
                ],
            }
            st.dataframe(pd.DataFrame(dim_data), use_container_width=True, hide_index=True)

    else:
        st.warning("No emotion data. Run `python main.py --emotion` first.")

    # ── AI Ranking ──
    if rank is not None and not rank.empty:
        st.subheader("🎯 AI Ranking Summary")
        c1, c2, c3, c4 = st.columns(4)
        for lvl, c in zip(['A+', 'A', 'B', 'C'], [c1, c2, c3, c4]):
            n = (rank['LEVEL'] == lvl).sum() if 'LEVEL' in rank.columns else 0
            c.metric(f"Level {lvl}", n)

        top = rank.head(10)
        cols_show = [c for c in ['code', 'name', 'predict_percent', 'AI_SCORE', 'LEVEL', 'SIGNAL', 'POSITION_PCT', 'STOP_LOSS'] if c in top.columns]
        st.dataframe(top[cols_show], use_container_width=True, hide_index=True)
