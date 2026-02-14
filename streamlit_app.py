import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(layout="wide")
st.title("📊 Commodity Supercycle Regime Model")

# -------------------------------
# CONFIGURAZIONE
# -------------------------------

# Sidebar per parametri
with st.sidebar:
    st.header("⚙️ Settings")
    
    data_frequency = st.selectbox(
        "Data Frequency",
        ["Weekly", "Monthly"],
        index=1  # Default to Monthly
    )
    
    years_back = st.slider("Years of History", 10, 25, 20)
    
    st.markdown("---")
    
    # Z-score window
    st.subheader("📊 Z-Score Settings")
    zscore_years = st.selectbox(
        "Z-Score Rolling Window",
        [3, 5, 7],
        index=1,  # Default 5 years
        help="Longer window = smoother signals, better for long-term cycles"
    )
    
    # Smoothing
    st.subheader("🎨 Probability Smoothing")
    enable_smoothing = st.checkbox("Enable Smoothing", value=True)
    
    if enable_smoothing:
        if data_frequency == "Monthly":
            smooth_periods = st.slider("Smoothing Window (months)", 3, 12, 6)
        else:
            smooth_periods = st.slider("Smoothing Window (weeks)", 4, 24, 12)
    else:
        smooth_periods = 1
    
    st.markdown("---")
    
    # Parametro Real Yield
    st.subheader("💰 Real Yield Calculation")
    inflation_assumption = st.number_input(
        "Assumed Breakeven Inflation (%)",
        min_value=0.0,
        max_value=5.0,
        value=2.3,
        step=0.1,
        help="Current 10Y breakeven inflation from FRED is ~2.3%. Adjust if needed."
    )
    
    st.info(f"💡 {data_frequency} data with {zscore_years}Y z-score window")

# -------------------------------
# DATA DOWNLOAD
# -------------------------------
@st.cache_data(ttl=3600)
def load_data(years, frequency, inflation_rate, zscore_years):
    """
    Scarica dati storici e resample a frequenza scelta
    """
    start_date = (datetime.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    
    tickers = {
        "Copper": "HG=F",
        "Gold": "GC=F",
        "Oil": "CL=F",  # WTI Crude Oil Futures
        "DXY": "DX-Y.NYB",
        "GSCI": "^SPGSCI",
        "US_10Y": "^TNX",
    }
    
    data = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, ticker) in enumerate(tickers.items()):
        try:
            status_text.text(f"Downloading {name}...")
            df = yf.download(ticker, start=start_date, end=datetime.now().strftime('%Y-%m-%d'), progress=False)
            
            if df.empty:
                st.warning(f"⚠️ No data for {name}")
                continue
            
            if "Close" in df.columns:
                data[name] = df["Close"]
            elif "Adj Close" in df.columns:
                data[name] = df["Adj Close"]
                
            progress_bar.progress((idx + 1) / len(tickers))
            
        except Exception as e:
            st.warning(f"⚠️ Error downloading {name}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Resample to weekly or monthly
    if frequency == "Weekly":
        data = data.resample('W').last()
    else:  # Monthly
        data = data.resample('M').last()
    
    # Forward fill piccoli gap
    data = data.fillna(method='ffill', limit=2)
    
    # Rimuovi righe con troppi missing
    data = data.dropna(thresh=len(data.columns)*0.6)
    
    # Calcola Real Yield usando l'assunzione di inflazione
    if "US_10Y" in data.columns:
        data["Real_Yield"] = data["US_10Y"] - inflation_rate
    
    return data

# Carica dati
try:
    df = load_data(years_back, data_frequency, inflation_assumption, zscore_years)
    
    if df.empty:
        st.error("❌ Unable to download data. Please try again later.")
        st.stop()
    
    st.success(f"✅ Data loaded: {len(df)} {data_frequency.lower()} periods ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# -------------------------------
# INDICATORS
# -------------------------------

# Copper/Gold Ratio
if "Copper" in df.columns and "Gold" in df.columns:
    df["Copper_Gold"] = df["Copper"] / df["Gold"]
    st.sidebar.success(f"✅ Copper/Gold calculated: {df['Copper_Gold'].iloc[-1]:.4f}")
else:
    if "Copper" not in df.columns:
        st.sidebar.warning("⚠️ Copper data missing")
    if "Gold" not in df.columns:
        st.sidebar.warning("⚠️ Gold data missing")
    df["Copper_Gold"] = np.nan

# Oil/Gold Ratio
if "Oil" in df.columns and "Gold" in df.columns:
    df["Oil_Gold"] = df["Oil"] / df["Gold"]
    st.sidebar.success(f"✅ Oil/Gold calculated: {df['Oil_Gold'].iloc[-1]:.4f}")
else:
    if "Oil" not in df.columns:
        st.sidebar.warning("⚠️ Oil data missing")
    df["Oil_Gold"] = np.nan

# Momentum
if "GSCI" in df.columns:
    # Per weekly: 26 settimane = 6 mesi
    # Per monthly: 6 mesi
    momentum_periods = 26 if data_frequency == "Weekly" else 6
    df["Momentum_6M"] = df["GSCI"].pct_change(momentum_periods) * 100
    st.sidebar.success(f"✅ Momentum calculated: {df['Momentum_6M'].iloc[-1]:.2f}%")
else:
    st.sidebar.warning("⚠️ GSCI data missing")
    df["Momentum_6M"] = np.nan

# Rimuovi NaN
df = df.dropna(subset=["Copper_Gold", "Oil_Gold", "Real_Yield", "Momentum_6M"])

if df.empty:
    st.error("❌ Not enough data after calculations")
    st.stop()

# Z-scores (configurabile: 3, 5 o 7 anni)
if data_frequency == "Weekly":
    window = zscore_years * 52  # anni * settimane
else:
    window = zscore_years * 12  # anni * mesi

st.sidebar.info(f"📊 Z-score window: {window} {data_frequency.lower()} periods ({zscore_years} years)")

indicators = ["Copper_Gold", "Oil_Gold", "Real_Yield", "DXY", "Momentum_6M"]

for col in indicators:
    if col in df.columns:
        mean = df[col].rolling(window).mean()
        std = df[col].rolling(window).std()
        df[f"{col}_z"] = (df[col] - mean) / std

# Rimuovi NaN da rolling
df = df.dropna()

if df.empty:
    st.error("❌ Not enough data after z-score calculation")
    st.stop()

# -------------------------------
# REGIME SCORE
# -------------------------------

score_components = []
available_indicators = []

if "Real_Yield_z" in df.columns:
    score_components.append((df["Real_Yield_z"] < 0).astype(int))
    available_indicators.append("Real Yield < 0")
    
if "Copper_Gold_z" in df.columns:
    score_components.append((df["Copper_Gold_z"] > 0).astype(int))
    available_indicators.append("Copper/Gold > 0")

if "Oil_Gold_z" in df.columns:
    score_components.append((df["Oil_Gold_z"] > 0).astype(int))
    available_indicators.append("Oil/Gold > 0")
    
if "DXY_z" in df.columns:
    score_components.append((df["DXY_z"] < 0).astype(int))
    available_indicators.append("Dollar < 0")
    
if "Momentum_6M_z" in df.columns:
    score_components.append((df["Momentum_6M_z"] > 0).astype(int))
    available_indicators.append("GSCI Momentum > 0")

if len(score_components) > 0:
    df["Score"] = sum(score_components)
    max_score = len(score_components)
    df["Score_Normalized"] = df["Score"] / max_score * 4
else:
    st.error("❌ Unable to calculate regime score")
    st.stop()

# Probability
df["Prob"] = 1 / (1 + np.exp(-1.5 * (df["Score_Normalized"] - 2)))

# Smoothed Probability (media mobile)
if enable_smoothing and smooth_periods > 1:
    df["Prob_Smooth"] = df["Prob"].rolling(smooth_periods, min_periods=1).mean()
else:
    df["Prob_Smooth"] = df["Prob"]

latest = df.iloc[-1]

# -------------------------------
# METRICS
# -------------------------------

st.markdown("---")
st.subheader("📈 Current Market Regime")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Regime Score", f"{int(latest['Score'])}/{max_score}")

with col2:
    prob_value = latest['Prob_Smooth'] * 100
    if enable_smoothing:
        prob_delta = (latest['Prob_Smooth'] - latest['Prob']) * 100
        st.metric("Supercycle Probability", f"{prob_value:.1f}%", 
                 f"{prob_delta:+.1f}pp (smoothed)" if abs(prob_delta) > 0.1 else None)

with col3:
    if latest["Score_Normalized"] <= 1.5:
        regime_label = "🐻 Bear"
    elif latest["Score_Normalized"] <= 2.5:
        regime_label = "⚖️ Transition"
    else:
        regime_label = "🚀 Supercycle"
    st.metric("Regime", regime_label)

with col4:
    # Real Yield attuale
    st.metric("Real Yield", f"{latest['Real_Yield']:.2f}%")

with col5:
    # Copper/Gold
    if "Copper_Gold" in df.columns:
        st.metric("Cu/Au Ratio", f"{latest['Copper_Gold']:.4f}")

with col6:
    # Oil/Gold
    if "Oil_Gold" in df.columns:
        st.metric("Oil/Au Ratio", f"{latest['Oil_Gold']:.4f}")

# Indicatori attivi
st.markdown(f"**Active Signals ({int(latest['Score'])}/{max_score}):** " + " | ".join([
    f"✅ {ind}" if score_components[i].iloc[-1] == 1 else f"❌ {ind}" 
    for i, ind in enumerate(available_indicators)
]))

st.markdown("---")

# -------------------------------
# TABS PER GRAFICI
# -------------------------------

tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability", "📈 Z-Scores", "💹 Raw Data", "ℹ️ Methodology"])

with tab1:
    st.subheader("🎯 Supercycle Probability Over Time")
    
    fig = go.Figure()
    
    # Raw Probability (sottile, semi-trasparente)
    if enable_smoothing and smooth_periods > 1:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Prob"] * 100,
            name="Raw Probability",
            line=dict(color='lightblue', width=1, dash='dot'),
            opacity=0.5
        ))
    
    # Smoothed Probability (principale)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Prob_Smooth"] * 100,
        name="Smoothed Probability" if enable_smoothing else "Probability",
        line=dict(color='blue', width=3),
        fill='tozeroy',
        fillcolor='rgba(0,100,255,0.2)'
    ))
    
    # Threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutral (50%)")
    fig.add_hline(y=75, line_dash="dot", line_color="green", annotation_text="Strong Signal (75%)")
    fig.add_hline(y=25, line_dash="dot", line_color="red", annotation_text="Weak Signal (25%)")
    
    # Regime zones
    fig.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.05, annotation_text="Bear Zone")
    fig.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.05, annotation_text="Bull Zone")
    
    fig.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="Date",
        hovermode='x unified',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Historical stats (usando smoothed)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bear_pct = (df["Prob_Smooth"] < 0.25).sum() / len(df) * 100
        st.metric("Time in Bear Regime", f"{bear_pct:.1f}%")
    
    with col2:
        transition_pct = ((df["Prob_Smooth"] >= 0.25) & (df["Prob_Smooth"] <= 0.75)).sum() / len(df) * 100
        st.metric("Time in Transition", f"{transition_pct:.1f}%")
    
    with col3:
        bull_pct = (df["Prob_Smooth"] > 0.75).sum() / len(df) * 100
        st.metric("Time in Supercycle", f"{bull_pct:.1f}%")

with tab2:
    st.subheader("📊 Macro Indicators (Z-Scores)")
    
    fig2 = go.Figure()
    
    z_cols = [col for col in df.columns if col.endswith('_z')]
    colors = ['red', 'green', 'blue', 'orange']
    
    for idx, col in enumerate(z_cols):
        fig2.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            name=col.replace('_z', '').replace('_', ' '),
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
    
    # Reference lines
    fig2.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    fig2.add_hrect(y0=-1, y1=1, fillcolor="gray", opacity=0.1, annotation_text="±1σ")
    fig2.add_hrect(y0=-2, y1=-1, fillcolor="red", opacity=0.05)
    fig2.add_hrect(y0=1, y1=2, fillcolor="green", opacity=0.05)
    
    fig2.update_layout(
        yaxis_title="Z-Score (Standard Deviations)",
        xaxis_title="Date",
        hovermode='x unified',
        height=600
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("💹 Raw Indicators")
    
    # Plot each indicator in separate subplots per migliore leggibilità
    raw_cols = ["Real_Yield", "Copper_Gold", "Oil_Gold", "DXY", "Momentum_6M"]
    available_raw = [col for col in raw_cols if col in df.columns]
    
    if len(available_raw) > 0:
        fig3 = make_subplots(
            rows=len(available_raw), 
            cols=1,
            subplot_titles=[col.replace('_', ' ').replace('Au', 'Gold') for col in available_raw],
            vertical_spacing=0.08
        )
        
        for idx, col in enumerate(available_raw):
            fig3.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col.replace('_', ' '),
                    line=dict(width=2),
                    showlegend=False
                ),
                row=idx+1,
                col=1
            )
            
            # Aggiungi linea zero per Momentum e Real Yield
            if col in ["Momentum_6M", "Real_Yield"]:
                fig3.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="gray",
                    row=idx+1,
                    col=1
                )
        
        fig3.update_layout(
            height=300 * len(available_raw),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Data table
    st.subheader("📋 Latest Data")
    display_cols = ["Real_Yield", "Copper_Gold", "Oil_Gold", "DXY", "Momentum_6M", "Score", "Prob"]
    available_display = [col for col in display_cols if col in df.columns]
    
    if len(available_display) > 0:
        st.dataframe(
            df[available_display].tail(20).style.format("{:.4f}"),
            use_container_width=True
        )

with tab4:
    st.markdown(f"""
    ### 📊 Methodology
    
    #### Data Frequency
    - **Current Setting**: {data_frequency}
    - Long-term cycle analysis benefits from lower frequency data to reduce noise
    
    #### Real Yield Calculation
    
    **Current Method:**
    ```
    Real Yield = US 10Y Treasury Yield - Assumed Breakeven Inflation
    ```
    
    - **US 10Y**: ^TNX from Yahoo Finance
    - **Breakeven Inflation**: {inflation_assumption}% (adjustable in sidebar)
    - **Note**: Official FRED DFII10 provides actual TIPS yield (~1.8% currently)
    
    ⚠️ **Limitation**: This is an approximation. For precise Real Yield, use FRED data (DFII10 or T10YIE).
    
    #### Regime Score Components
    
    Each indicator contributes +1 to score when condition is met:
    
    1. **Real Yield** (Z-score < 0) 
       - Negative real yields = supportive for commodities
       
    2. **Copper/Gold Ratio** (Z-score > 0)
       - Rising ratio = industrial demand strength signal
       
    3. **Oil/Gold Ratio** (Z-score > 0)
       - Rising ratio = energy demand and economic activity
       
    4. **Dollar Index** (Z-score < 0)
       - Weak dollar = positive for commodity prices
       
    5. **GSCI Momentum 6M** (Z-score > 0)
       - Positive momentum = broad commodity trend strength
    
    **Total Score**: 0-5
    
    **Why Oil/Gold?**
    - Oil = proxy for global economic activity and energy demand
    - Complements Copper/Gold (industrial metals) with energy component
    - GSCI is 50-60% energy-weighted, so Oil separate provides clarity
    
    #### Z-Score Normalization
    
    - **Window**: Configurable ({zscore_years} years = {window} {data_frequency.lower()} periods)
    - **Formula**: (Value - Rolling Mean) / Rolling StdDev
    - **Purpose**: Normalize different indicators to comparable scale
    - **Longer windows** (5-7 years) = smoother signals, better for identifying long-term supercycles
    
    #### Probability Smoothing
    
    - **Enabled**: {"Yes" if enable_smoothing else "No"}
    - **Window**: {smooth_periods} {data_frequency.lower()} periods{" (recommended for supercycle analysis)" if enable_smoothing else ""}
    - **Method**: Simple Moving Average
    - **Purpose**: Remove short-term noise to reveal underlying regime trend
    - **Trade-off**: Smoothing reduces responsiveness but increases signal quality
    
    #### Probability Function
    
    ```
    P(Supercycle) = 1 / (1 + e^(-1.5 × (Score_Normalized - 2)))
    ```
    
    Where Score_Normalized scales the actual score (0-5) to range 0-4 for the sigmoid function.
    
    - **Score 0-2**: Bear regime (P < 35%)
    - **Score 2-3**: Neutral/Transition (P ≈ 35-65%)
    - **Score 4-5**: Supercycle (P > 65%)
    
    #### Data Sources
    
    - **Yahoo Finance**: Commodity prices, yields, dollar index
    - **Frequency**: {data_frequency} ({len(df)} periods)
    - **History**: {years_back} years
    
    #### Improvements Needed
    
    - [ ] Integrate FRED API for accurate Real Yield (DFII10 or T10YIE)
    - [ ] Add oil prices as additional indicator
    - [ ] Historical backtest performance metrics
    """)

st.markdown("---")

# Footer
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Current Real Yield Assumption**")
    st.info(f"{inflation_assumption}% breakeven inflation")

with col2:
    st.markdown("**Data Quality**")
    data_completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    st.success(f"{data_completeness:.1f}% complete")

with col3:
    st.markdown("**Last Update**")
    st.info(f"{df.index[-1].strftime('%Y-%m-%d')}")

st.markdown("""
<div style='text-align: center; color: gray; margin-top: 20px;'>
    <p>📊 Commodity Supercycle Dashboard v2.0 | Yahoo Finance Data | {frequency} Updates</p>
    <p style='font-size: 0.8em;'>⚠️ For production use, integrate FRED API for accurate Real Yield calculation</p>
</div>
""".format(frequency=data_frequency), unsafe_allow_html=True)
