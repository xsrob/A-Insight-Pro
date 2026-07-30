"""LSTM Prediction Monitor"""
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT = os.path.join(ROOT, 'reports')
MODELS = os.path.join(ROOT, 'models')

def show():
    st.title("Prediction Monitor")
    lstm_ok = os.path.exists(os.path.join(MODELS,'stock_model_lstm.pt'))
    rf_ok = os.path.exists(os.path.join(MODELS,'stock_model.pkl'))
    c1,c2,c3 = st.columns(3)
    c1.metric("LSTM", "Ready" if lstm_ok else "Not Trained")
    c2.metric("RandomForest", "Ready" if rf_ok else "Not Trained")
    c3.metric("Ensemble", "LSTM 60% + RF 40%")

    pred_path = os.path.join(REPORT, 'final_stock_rank.csv')
    if os.path.exists(pred_path):
        pred = pd.read_csv(pred_path, encoding='utf-8-sig')
        st.subheader("Prediction Distribution")
        if 'predict_percent' in pred.columns:
            fig = px.histogram(pred, x='predict_percent', nbins=50, color_discrete_sequence=['#1e3799'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Predictions")
        cols = [c for c in ['code','name','predict_percent','AI_SCORE','LEVEL','SIGNAL'] if c in pred.columns]
        st.dataframe(pred[cols].head(20), use_container_width=True, hide_index=True)

    # Model health
    adjust_path = os.path.join(REPORT, 'predict_adjust.json')
    if os.path.exists(adjust_path):
        import json
        with open(adjust_path, 'r') as f: adj = json.load(f)
        st.subheader("Model Calibration")
        c1,c2,c3 = st.columns(3)
        c1.metric("Predict Factor", f"{adj.get('predict_factor',1.0):.2f}")
        c2.metric("Avg Error", f"{adj.get('avg_error',0):.2f}%")
        c3.metric("Success Rate", f"{adj.get('avg_success_rate',0):.1%}")
