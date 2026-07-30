"""A-Insight Pro Dashboard V2.0"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title='A-Insight Pro', layout='wide')
st.sidebar.title('A-Insight Pro')
page = st.sidebar.radio('Navigation', [
    'Market Overview', 'AI Ranking', 'Backtest Analysis',
    'Factor Analysis', 'LSTM Predictions', 'Learning Monitor'
], label_visibility='collapsed')
st.sidebar.markdown('---')
st.sidebar.caption('v2.0 | AKShare')

PAGES_MAP = {
    'Market Overview': 'pages.page01_market_overview',
    'AI Ranking': 'pages.page02_ai_ranking',
    'Backtest Analysis': 'pages.page03_backtest',
    'Factor Analysis': 'pages.page04_factor_analysis',
    'LSTM Predictions': 'pages.page05_lstm_predict',
    'Learning Monitor': 'pages.page06_learning_monitor',
}
mod = PAGES_MAP.get(page)
if mod:
    try:
        exec(f'from {mod} import show; show()')
    except Exception as e:
        st.error(f'Page load error: {e}')
