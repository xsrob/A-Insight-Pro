"""Backtest Analysis - Enhanced"""
import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT = os.path.join(ROOT, 'reports')

@st.cache_data(ttl=600)
def load_data():
    data = {}
    for name in ['backtest_summary.csv','backtest_report.csv','equity_curve.csv']:
        p = os.path.join(REPORT, name)
        if os.path.exists(p):
            try: data[name.split('.')[0]] = pd.read_csv(p, encoding='utf-8-sig')
            except: pass
    return data

def show():
    st.title("Backtest Analysis")
    data = load_data()
    summary = data.get('backtest_summary')
    bt = data.get('backtest_report')
    equity = data.get('equity_curve')

    if summary is not None and not summary.empty:
        s = summary.iloc[0]
        cols = st.columns(8)
        cols[0].metric("Total Ret", f"{s.get('total_return_pct',0):.1f}%")
        cols[1].metric("Sharpe", f"{s.get('sharpe_ratio',0):.2f}")
        cols[2].metric("Max DD", f"{s.get('max_drawdown_pct',0):.1f}%")
        cols[3].metric("Win Rate", f"{s.get('win_rate_pct',0):.1f}%")
        cols[4].metric("Sortino", f"{s.get('sortino_ratio',0):.2f}")
        cols[5].metric("Calmar", f"{s.get('calmar_ratio',0):.2f}")
        cols[6].metric("Ann.Ret", f"{s.get('annualized_return_pct',0):.1f}%")
        cols[7].metric("Trades", f"{int(s.get('total_trades',0))}")

    if equity is not None and not equity.empty and 'capital' in equity.columns:
        st.subheader("Equity Curve")
        v = equity['capital'].values
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=v/v[0]-1, mode='lines', name='Portfolio', line=dict(color='#1e3799',width=2), fill='tozeroy', fillcolor='rgba(30,55,153,0.1)'))
        fig.update_layout(height=350, yaxis_tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drawdown")
        peak = np.maximum.accumulate(v); dd = (v-peak)/peak
        fig2 = go.Figure(go.Scatter(y=dd, mode='lines', fill='tozeroy', fillcolor='rgba(231,76,60,0.15)', line=dict(color='#e74c3c',width=1.5)))
        fig2.update_layout(height=200, yaxis_tickformat='.1%')
        st.plotly_chart(fig2, use_container_width=True)

    if bt is not None and not bt.empty:
        st.subheader("Per-Stock Performance")
        sc = [c for c in ['code','trade_count','win_rate','avg_return','total_return','max_drawdown'] if c in bt.columns]
        st.dataframe(bt[sc].head(30), use_container_width=True, hide_index=True)
