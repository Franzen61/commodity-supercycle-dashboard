import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")

st.title("Commodity Supercycle Regime Model")

# -------------------------------
# DATA DOWNLOAD
# -------------------------------

@st.cache_data
def load_data():

    tickers = {
        "Copper": "HG=F",
        "Gold": "GC=F",
        "DXY": "DX-Y.NYB",
        "GSCI": "^SPGSCI",
        "10Y": "^TNX",
        "CPI": "CPIAUCSL"
    }

    data = pd.DataFrame()

    for name, ticker in tickers.items():

        df = yf.download(ticker, start="2000-01-01", progress=False)

        # CPI non ha Adj Close
        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        else:
            series = df["Close"]

        data[name] = series

    # CPI è mensile → trasformiamo in giornaliero
    data = data.resample("D").ffill()

    # eliminiamo solo righe completamente vuote
    data = data.dropna(how="all")

    return data

df = load_data()

# -------------------------------
# INDICATORS
# -------------------------------

df["Copper_Gold"] = df["Copper"] / df["Gold"]
df["Real_Yield"] = df["10Y"] - df["CPI"].pct_change(252)*100
df["Momentum_6M"] = df["GSCI"].pct_change(126)

window = 756  # 3 years

for col in ["Copper_Gold", "Real_Yield", "DXY", "Momentum_6M"]:
    df[f"{col}_z"] = (df[col] - df[col].rolling(window).mean()) / df[col].rolling(window).std()

df = df.dropna()

# -------------------------------
# REGIME SCORE
# -------------------------------

df["Score"] = (
    (df["Real_Yield_z"] < 0).astype(int) +
    (df["Copper_Gold_z"] > 0).astype(int) +
    (df["DXY_z"] < 0).astype(int) +
    (df["Momentum_6M_z"] > 0).astype(int)
)

df["Prob"] = 1 / (1 + np.exp(-1.5*(df["Score"]-2)))

latest = df.iloc[-1]

# -------------------------------
# METRICS
# -------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Regime Score", int(latest["Score"]))
col2.metric("Supercycle Probability", f"{latest['Prob']*100:.1f}%")

if latest["Score"] <= 1:
    regime_label = "Bear Regime"
elif latest["Score"] == 2:
    regime_label = "Transition"
else:
    regime_label = "Supercycle"

col3.metric("Current Regime", regime_label)

# -------------------------------
# PROBABILITY CHART
# -------------------------------

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Prob"],
    name="Supercycle Probability"
))
fig.update_layout(title="Supercycle Probability Over Time")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Z-SCORE CHART
# -------------------------------

fig2 = go.Figure()
for col in ["Real_Yield_z", "Copper_Gold_z", "DXY_z", "Momentum_6M_z"]:
    fig2.add_trace(go.Scatter(x=df.index, y=df[col], name=col))

fig2.update_layout(title="Macro Z-Scores")
st.plotly_chart(fig2, use_container_width=True)
