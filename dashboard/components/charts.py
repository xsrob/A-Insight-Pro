"""Reusable Plotly chart components."""
import plotly.graph_objects as go, plotly.express as px, numpy as np

COLORS={"primary":"#1e3799","positive":"#27ae60","negative":"#e74c3c","warning":"#e67e22","neutral":"#7f8c8d"}

def equity_curve(values,benchmark=None):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=np.array(values)/values[0]-1,mode='lines',name='Portfolio',line=dict(color=COLORS['primary'],width=2),fill='tozeroy',fillcolor='rgba(30,55,153,0.1)'))
    if benchmark is not None:
        fig.add_trace(go.Scatter(y=np.array(benchmark)/benchmark[0]-1,mode='lines',name='Benchmark',line=dict(color=COLORS['neutral'],width=1.5,dash='dash')))
    fig.update_layout(height=350,yaxis_tickformat='.1%',margin=dict(l=20,r=20,t=20,b=20))
    return fig

def drawdown_chart(values):
    peak=np.maximum.accumulate(values); dd=(np.array(values)-peak)/peak
    fig=go.Figure(go.Scatter(y=dd,mode='lines',fill='tozeroy',fillcolor='rgba(231,76,60,0.15)',line=dict(color=COLORS['negative'],width=1.5)))
    fig.update_layout(height=200,yaxis_tickformat='.1%',margin=dict(l=20,r=20,t=20,b=20))
    return fig

def correlation_heatmap(df):
    fig=px.imshow(df.corr(),text_auto='.2f',color_continuous_scale='RdBu_r',zmin=-1,zmax=1)
    fig.update_layout(height=400,margin=dict(l=20,r=20,t=20,b=20))
    return fig
