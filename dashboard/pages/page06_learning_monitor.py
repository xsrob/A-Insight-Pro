"""Learning & Bias Monitor"""
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT = os.path.join(ROOT, 'reports')

@st.cache_data(ttl=300)
def load_data():
    data = {}
    for name in ['ai_learning_feedback.csv','simulate_review.csv']:
        p = os.path.join(REPORT, name)
        if os.path.exists(p):
            try: data[name.split('.')[0]] = pd.read_csv(p, encoding='utf-8-sig')
            except: pass
    return data

def show():
    st.title("Learning & Bias Monitor")
    data = load_data()
    learning = data.get('ai_learning_feedback')
    review = data.get('simulate_review')

    if learning is not None and not learning.empty:
        c1,c2,c3 = st.columns(3)
        c1.metric("Stocks Tracked", len(learning))
        if 'success_rate' in learning.columns:
            c2.metric("Avg Success", f"{learning['success_rate'].mean():.1%}")
            c3.metric("Low Success (<30%)", (learning['success_rate']<0.3).sum())

        st.subheader("Success Rate Distribution")
        if 'success_rate' in learning.columns:
            fig = px.histogram(learning, x='success_rate', nbins=20, color_discrete_sequence=['#1e3799'])
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Learning Feedback Detail")
        sc = [c for c in ['code','samples','success_rate','avg_error','final_adjust','risk_level'] if c in learning.columns]
        st.dataframe(learning[sc].head(30), use_container_width=True, hide_index=True)

    if review is not None and not review.empty:
        st.subheader("Recent Review Results")
        if 'result' in review.columns:
            success = (review['result']=='成功').sum()
            total = len(review)
            st.metric("Recent Accuracy", f"{success/total:.1%}" if total>0 else "N/A", delta=f"{success}/{total}")
        sc2 = [c for c in ['date','code','predict_percent','actual_return','result'] if c in review.columns]
        st.dataframe(review[sc2].tail(20), use_container_width=True, hide_index=True)
