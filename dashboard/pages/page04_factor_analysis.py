"""Factor Analysis - Enhanced with IC data"""
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT = os.path.join(ROOT, 'reports')

@st.cache_data(ttl=600)
def load_data():
    data = {}
    for name in ['factor_ic_analysis.csv','final_stock_rank.csv','factor_weights.json']:
        p = os.path.join(REPORT, name)
        if os.path.exists(p):
            try: data[name.split('.')[0]] = pd.read_csv(p, encoding='utf-8-sig')
            except: pass
    return data

def show():
    st.title("Factor Analysis")
    data = load_data()
    ic = data.get('factor_ic_analysis')
    rank = data.get('final_stock_rank')

    if ic is not None and not ic.empty:
        st.subheader("Factor IC Analysis (Information Coefficient)")
        st.caption("|IC| > 0.02 = meaningful, > 0.05 = strong predictor. Hit Rate = % stocks where factor works.")
        ic_df = ic.sort_values('abs_ic_mean', ascending=False)
        fig = px.bar(ic_df.head(15), x='factor', y='abs_ic_mean', color='ic_mean',
                     color_continuous_scale=['#e74c3c','#fff','#27ae60'], title='Factor Predictive Power (|IC|)')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ic_df.head(15)[['factor','ic_mean','ic_ir','hit_rate','n_stocks']], use_container_width=True, hide_index=True)

    if rank is not None and not rank.empty:
        st.subheader("AI_SCORE Distribution by LEVEL")
        if 'LEVEL' in rank.columns:
            fig2 = px.box(rank, x='LEVEL', y='AI_SCORE', color='LEVEL', category_orders={'LEVEL':['A+','A','B','C']})
            st.plotly_chart(fig2, use_container_width=True)
