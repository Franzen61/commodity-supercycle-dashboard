
"""
COMMODITY SUPERCYCLE DASHBOARD
Versione 1.1 - Macro Cycle Model
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Commodity Supercycle Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Commodity Supercycle Dashboard")
st.markdown("Analisi macro ciclica e regime commodities")

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Configurazione")

    years_back = st.slider("Anni di storico", 5, 25, 15)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)

    commodity_choice = st.selectbox(
        "Indice Commodity",
        ["DBC", "GSG"]
    )

    ticker_map = {
        "DBC": "DBC",
        "GSG": "GSG"
    }

    main_commodity = ticker_map[commodity_choice]
    rolling_window = st.slider("Rolling correlation", 30, 400, 252)

# -----------------------------------------------------------------------------
# DATA LOAD
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data(start, end, commodity_ticker):

    tickers = {
        "Commodity": commodity_ticker,
        "Copper": "HG=F",
        "Gold": "GC=F",
        "Oil": "CL=F",
        "DXY": "DX-Y.NYB",
        "US_10Y": "^TNX",
        "US_2Y": "^FVX",
        "SPX": "^GSPC",
        "VIX": "^VIX",
        "TIP": "TIP"
    }

    data = pd.DataFrame()

    for name, ticker in tickers.items():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )
        if not df.empty:
            data[name] = df["Close"]

    return data.ffill(limit=5)

# -----------------------------------------------------------------------------
# INDICATORS
# -----------------------------------------------------------------------------

def calculate_indicators(df):

    ind = pd.DataFrame(index=df.index)

    if "Copper" in df and "Gold" in df:
        ind["Copper_Gold"] = df["Copper"] / df["Gold"]

    if "US_10Y" in df and "US_2Y" in df:
        ind["Yield_Curve"] = df["US_10Y"] - df["US_2Y"]

    if "US_10Y" in df and "TIP" in df:
        ind["Real_Yield_Proxy"] = df["US_10Y"] / df["TIP"]

    if "Commodity" in df and "SPX" in df:
        ind["Commodity_vs_SPX"] = df["Commodity"] / df["SPX"]

    if "Commodity" in df:
        ind["Momentum_6M"] = df["Commodity"].pct_change(126) * 100

    return ind

# -----------------------------------------------------------------------------
# REGIME LOGIC
# -----------------------------------------------------------------------------

def identify_regime(data, ind):

    if len(data) < 126:
        return "N/A"

    momentum = ind["Momentum_6M"].iloc[-1]
    real_yield = ind["Real_Yield_Proxy"].pct_change(63).iloc[-1]
    copper_gold = ind["Copper_Gold"].pct_change(63).iloc[-1]

    if real_yield < 0 and copper_gold > 0:
        return "🚀 Expansion"

    if momentum > 15 and real_yield < 0:
        return "🔥 Commodity Boom"

    if real_yield > 0 and data["DXY"].pct_change(63).iloc[-1] > 0:
        return "📉 Contraction"

    return "⚖️ Consolidation"

# -----------------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------------

data = load_data(start_date, end_date, main_commodity)
indicators = calculate_indicators(data)

# -----------------------------------------------------------------------------
# KPI
# -----------------------------------------------------------------------------

st.header("Situazione Attuale")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Regime", identify_regime(data, indicators))

with col2:
    if "Commodity" in data:
        st.metric("Commodity", f"{data['Commodity'].iloc[-1]:.2f}")

with col3:
    if "Yield_Curve" in indicators:
        st.metric("Yield Curve", f"{indicators['Yield_Curve'].iloc[-1]:.2f}")

# -----------------------------------------------------------------------------
# CHART
# -----------------------------------------------------------------------------

st.subheader("Performance Normalizzata")

norm = data.dropna()
norm = (norm / norm.iloc[0]) * 100

fig = go.Figure()
for col in norm.columns:
    fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col))

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# CORRELATIONS
# -----------------------------------------------------------------------------

st.subheader("Correlation Matrix")
corr = data.dropna().corr()

fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1)
st.plotly_chart(fig, use_container_width=True)
```
