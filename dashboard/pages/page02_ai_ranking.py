"""AI Ranking Explorer - Enhanced"""
import streamlit as st, pandas as pd, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT = os.path.join(ROOT, 'reports')

@st.cache_data(ttl=300)
def load_rank():
    p = os.path.join(REPORT, 'final_stock_rank.csv')
    if os.path.exists(p):
        try: return pd.read_csv(p, encoding='utf-8-sig')
        except: pass
    return pd.DataFrame()

def show():
    st.title("AI Ranking Explorer")
    rank = load_rank()
    if rank.empty: st.warning("No data"); return

    c1,c2,c3,c4 = st.columns(4)
    with c1: levels = st.multiselect("Level", ['A+','A','B','C'], default=['A+','A','B'])
    with c2: min_score = st.slider("Min Score", 0, 100, 40)
    with c3: min_pred = st.slider("Min Predict %", -10.0, 20.0, -5.0, 0.5)
    with c4: sort_by = st.selectbox("Sort by", ['AI_SCORE','predict_percent','POSITION_PCT'])

    df = rank.copy()
    if levels and 'LEVEL' in df.columns: df = df[df['LEVEL'].isin(levels)]
    if 'AI_SCORE' in df.columns: df = df[df['AI_SCORE'] >= min_score]
    if 'predict_percent' in df.columns: df = df[df['predict_percent'] >= min_pred]
    if sort_by in df.columns: df = df.sort_values(sort_by, ascending=False)

    st.metric("Results", len(df))
    cols_show = [c for c in ['code','name','predict_percent','AI_SCORE','LEVEL','SIGNAL','POSITION_PCT','STOP_LOSS','MAX_SINGLE'] if c in df.columns]
    st.dataframe(df[cols_show], use_container_width=True, hide_index=True, height=600)
    st.download_button("Download CSV", df[cols_show].to_csv(index=False), "ai_ranking_filtered.csv", "text/csv")
