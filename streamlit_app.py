import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Commodity Supercycle Regime Model")

# -------------------------------
# DATA DOWNLOAD
# -------------------------------
@st.cache_data(ttl=3600)
def load_data():
    """
    Scarica dati storici.
    NOTA: CPI non è disponibile su Yahoo Finance - usiamo un proxy
    """
    tickers = {
        "Copper": "HG=F",
        "Gold": "GC=F",
        "DXY": "DX-Y.NYB",
        "GSCI": "^SPGSCI",
        "US_10Y": "^TNX",
        "TIP": "TIP"  # TIPS ETF come proxy per real yield
    }
    
    data = pd.DataFrame()
    
    with st.spinner("Downloading data from Yahoo Finance..."):
        for name, ticker in tickers.items():
            try:
                df = yf.download(ticker, start="2000-01-01", end=datetime.now().strftime('%Y-%m-%d'), progress=False)
                
                if df.empty:
                    st.warning(f"⚠️ No data for {name} ({ticker})")
                    continue
                
                # Usa Close se disponibile
                if "Close" in df.columns:
                    data[name] = df["Close"]
                elif "Adj Close" in df.columns:
                    data[name] = df["Adj Close"]
                    
            except Exception as e:
                st.warning(f"⚠️ Error downloading {name}: {str(e)}")
                continue
    
    # Forward fill per gap piccoli
    data = data.fillna(method='ffill', limit=5)
    
    # Rimuovi righe con troppi NaN
    data = data.dropna(thresh=len(data.columns)*0.6)
    
    return data

# Carica dati
try:
    df = load_data()
    
    if df.empty:
        st.error("❌ Unable to download data. Please try again later.")
        st.stop()
    
    st.success(f"✅ Data loaded: {len(df)} days")
    
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# -------------------------------
# INDICATORS
# -------------------------------

# Copper/Gold Ratio
if "Copper" in df.columns and "Gold" in df.columns:
    df["Copper_Gold"] = df["Copper"] / df["Gold"]
else:
    st.warning("⚠️ Copper or Gold data missing")
    df["Copper_Gold"] = 0

# Real Yield Proxy
# Invece di usare CPI (non disponibile su YF), usiamo un approccio semplificato
if "US_10Y" in df.columns and "TIP" in df.columns:
    # TIP riflette già l'inflation protection
    # Real Yield approssimato = 10Y nominal - inflation expectation
    # Usiamo il cambio % di TIP come proxy dell'inflation expectation
    inflation_proxy = df["TIP"].pct_change(252) * 100  # YoY change
    df["Real_Yield"] = df["US_10Y"] - inflation_proxy.rolling(21).mean()
elif "US_10Y" in df.columns:
    # Fallback: assumiamo 2% inflation target
    df["Real_Yield"] = df["US_10Y"] - 2.0
else:
    st.warning("⚠️ 10Y Treasury data missing")
    df["Real_Yield"] = 0

# Momentum 6 mesi
if "GSCI" in df.columns:
    df["Momentum_6M"] = df["GSCI"].pct_change(126) * 100  # 6 mesi in percentuale
else:
    st.warning("⚠️ GSCI data missing")
    df["Momentum_6M"] = 0

# Rimuovi NaN generati dai calcoli
df = df.dropna(subset=["Copper_Gold", "Real_Yield", "Momentum_6M"])

if df.empty:
    st.error("❌ Not enough data after calculations")
    st.stop()

# Z-scores
window = 756  # 3 anni

indicators_to_normalize = []
if "Copper_Gold" in df.columns:
    indicators_to_normalize.append("Copper_Gold")
if "Real_Yield" in df.columns:
    indicators_to_normalize.append("Real_Yield")
if "DXY" in df.columns:
    indicators_to_normalize.append("DXY")
if "Momentum_6M" in df.columns:
    indicators_to_normalize.append("Momentum_6M")

for col in indicators_to_normalize:
    mean = df[col].rolling(window).mean()
    std = df[col].rolling(window).std()
    df[f"{col}_z"] = (df[col] - mean) / std

# Rimuovi NaN da rolling windows
df = df.dropna()

if df.empty:
    st.error("❌ Not enough data after z-score calculation")
    st.stop()

# -------------------------------
# REGIME SCORE
# -------------------------------

# Calcola score solo se abbiamo tutti gli indicatori
score_components = []

if "Real_Yield_z" in df.columns:
    score_components.append((df["Real_Yield_z"] < 0).astype(int))
    
if "Copper_Gold_z" in df.columns:
    score_components.append((df["Copper_Gold_z"] > 0).astype(int))
    
if "DXY_z" in df.columns:
    score_components.append((df["DXY_z"] < 0).astype(int))
    
if "Momentum_6M_z" in df.columns:
    score_components.append((df["Momentum_6M_z"] > 0).astype(int))

if len(score_components) > 0:
    df["Score"] = sum(score_components)
    # Normalizza score in base al numero di componenti disponibili
    max_score = len(score_components)
    df["Score_Normalized"] = df["Score"] / max_score * 4  # Scala a 0-4
else:
    st.error("❌ Unable to calculate regime score")
    st.stop()

# Probabilità superciclo (sigmoid function)
df["Prob"] = 1 / (1 + np.exp(-1.5 * (df["Score_Normalized"] - 2)))

# Latest values
latest = df.iloc[-1]

# -------------------------------
# METRICS
# -------------------------------

st.markdown("---")
st.subheader("📈 Current Market Regime")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Regime Score", f"{latest['Score']}/{max_score}")

with col2:
    prob_value = latest['Prob'] * 100
    st.metric("Supercycle Probability", f"{prob_value:.1f}%")

with col3:
    if latest["Score_Normalized"] <= 1.5:
        regime_label = "🐻 Bear Regime"
        regime_color = "red"
    elif latest["Score_Normalized"] <= 2.5:
        regime_label = "⚖️ Transition"
        regime_color = "orange"
    else:
        regime_label = "🚀 Supercycle"
        regime_color = "green"
    
    st.metric("Current Regime", regime_label)

with col4:
    # Trend (confronto con 1 mese fa)
    if len(df) > 21:
        prob_change = (latest["Prob"] - df.iloc[-21]["Prob"]) * 100
        st.metric("1M Change", f"{prob_change:+.1f}pp")

st.markdown("---")

# -------------------------------
# PROBABILITY CHART
# -------------------------------

st.subheader("🎯 Supercycle Probability Over Time")

fig = go.Figure()

# Probability line
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Prob"] * 100,
    name="Supercycle Probability",
    line=dict(color='blue', width=2),
    fill='tozeroy',
    fillcolor='rgba(0,100,255,0.2)'
))

# Threshold lines
fig.add_hline(y=50, line_dash="dash", line_color="gray", 
              annotation_text="50% Threshold")
fig.add_hline(y=75, line_dash="dash", line_color="green", 
              annotation_text="High Probability")
fig.add_hline(y=25, line_dash="dash", line_color="red", 
              annotation_text="Low Probability")

fig.update_layout(
    title="Supercycle Probability Evolution",
    yaxis_title="Probability (%)",
    xaxis_title="Date",
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Z-SCORE CHART
# -------------------------------

st.subheader("📊 Macro Indicators (Z-Scores)")

fig2 = go.Figure()

z_score_cols = [col for col in df.columns if col.endswith('_z')]

colors = ['red', 'green', 'blue', 'orange', 'purple']

for idx, col in enumerate(z_score_cols):
    fig2.add_trace(go.Scatter(
        x=df.index, 
        y=df[col], 
        name=col.replace('_z', ''),
        line=dict(color=colors[idx % len(colors)])
    ))

# Zero line
fig2.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

# +/- 1 sigma zones
fig2.add_hrect(y0=-1, y1=1, fillcolor="gray", opacity=0.1, 
               annotation_text="±1σ", annotation_position="right")

fig2.update_layout(
    title="Normalized Macro Indicators (3-Year Rolling Z-Scores)",
    yaxis_title="Z-Score",
    xaxis_title="Date",
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# DATA TABLE
# -------------------------------

with st.expander("📋 View Latest Data"):
    st.dataframe(
        df[[col for col in df.columns if not col.endswith('_z')]].tail(20).style.format("{:.2f}"),
        use_container_width=True
    )

# -------------------------------
# METHODOLOGY
# -------------------------------

with st.expander("ℹ️ Methodology"):
    st.markdown("""
    ### Regime Score Calculation
    
    The regime score is based on 4 key macro indicators:
    
    1. **Real Yield** (Z-score < 0) → Negative real yields favor commodities
    2. **Copper/Gold Ratio** (Z-score > 0) → Rising ratio indicates economic strength
    3. **Dollar Index** (Z-score < 0) → Weak dollar supports commodity prices
    4. **6-Month Momentum** (Z-score > 0) → Positive momentum signals trend strength
    
    Each condition = +1 point → Score range: 0-4
    
    ### Probability Function
    
    Supercycle Probability = 1 / (1 + e^(-1.5 × (Score - 2)))
    
    - **Score 0-1**: Bear regime (Low probability)
    - **Score 2**: Transition (50% probability)
    - **Score 3-4**: Supercycle (High probability)
    
    ### Data Sources
    
    - Yahoo Finance (daily data)
    - 3-year rolling window for z-score normalization
    - Forward-fill for missing data (max 5 days)
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 Commodity Supercycle Dashboard | Data from Yahoo Finance | Updates every hour</p>
</div>
""", unsafe_allow_html=True)
