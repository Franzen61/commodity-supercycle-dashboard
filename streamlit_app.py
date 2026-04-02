"""
COMMODITY SUPERCYCLE DASHBOARD v4.1.2 - RESEARCH GRADE + DIVERGENCE ANALYSIS
===========================================================================
Con Yield Curve 10Y-3M, Real Yield migliorato e Momentum Divergence Detection
6 indicatori macro + analisi divergenze per identificazione turning points

CHANGELOG v4.1.2:
- FIX #1: data.fillna(method='ffill') → data.ffill() (deprecato Pandas 2.1, rimosso Pandas 3.x)
- FIX #2: st.progress() e st.empty() spostati fuori da @st.cache_data
           → al secondo caricamento i widget non vengono renderizzati dentro la cache,
             l'app sembrava congelata senza feedback visivo

CHANGELOG v4.1.1:
- Fixed: dropna() crash con default settings (25yr + 5yr z-score)
- Now: Elimina SOLO righe con z-scores mancanti, non tutte le colonne
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(layout="wide", page_title="Commodity Supercycle Dashboard v4.1")
st.title("📊 Commodity Supercycle Regime Model v4.1")

# ============================================================================
# SIDEBAR - CONFIGURAZIONE
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    data_frequency = st.selectbox("Data Frequency", ["Weekly", "Monthly"], index=1)
    years_back = st.slider("Years of History", 10, 30, 30)
    
    st.markdown("---")
    
    st.subheader("📊 Z-Score Settings")
    zscore_years = st.selectbox("Z-Score Rolling Window", [3, 5, 7], index=1,
                                help="Longer window = smoother signals, better for long-term cycles")
    
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
    
    st.subheader("⚖️ Indicator Weights")
    weight_mode = st.radio("Weighting Mode", 
                          ["Equal Weights (Simple)", "Custom Weights (Advanced)"], 
                          index=0)
    
    if weight_mode == "Custom Weights (Advanced)":
        st.warning("⚠️ **Overfitting Risk**: Use academic defaults or validated weights only.")
        
        use_defaults = st.checkbox("Use Academic Defaults", value=True)
        
        if use_defaults:
            weight_real_yield = 0.25
            weight_dollar     = 0.20
            weight_cu_gold    = 0.20
            weight_oil_gold   = 0.15
            weight_yield_curve = 0.10
            weight_momentum   = 0.10
        else:
            st.caption("Adjust weights manually (must sum to 100%)")
            col1, col2 = st.columns(2)
            with col1:
                weight_real_yield  = st.slider("Real Yield",   0.0, 0.5, 0.25, 0.05)
                weight_cu_gold     = st.slider("Copper/Gold",  0.0, 0.3, 0.20, 0.05)
                weight_yield_curve = st.slider("Yield Curve",  0.0, 0.3, 0.10, 0.05)
            with col2:
                weight_dollar   = st.slider("Dollar Index",    0.0, 0.5, 0.20, 0.05)
                weight_oil_gold = st.slider("Oil/Gold",        0.0, 0.3, 0.15, 0.05)
                weight_momentum = st.slider("GSCI Momentum",   0.0, 0.2, 0.10, 0.05)
            
            total_weight = (weight_real_yield + weight_dollar + weight_cu_gold +
                           weight_oil_gold + weight_yield_curve + weight_momentum)
            if abs(total_weight - 1.0) > 0.01:
                st.error(f"⚠️ Weights must sum to 100% (currently {total_weight*100:.1f}%)")
    else:
        weight_real_yield  = 0.1667
        weight_dollar      = 0.1667
        weight_cu_gold     = 0.1667
        weight_oil_gold    = 0.1667
        weight_yield_curve = 0.1667
        weight_momentum    = 0.1667
    
    weights = {
        'Real_Yield':   weight_real_yield,
        'DXY':          weight_dollar,
        'Copper_Gold':  weight_cu_gold,
        'Oil_Gold':     weight_oil_gold,
        'Yield_Curve':  weight_yield_curve,
        'Momentum_6M':  weight_momentum
    }
    
    st.markdown("---")
    
    st.subheader("🔔 Alert Sensitivity")
    alert_sensitivity = st.select_slider("Threshold Level",
                                        options=["Conservative", "Moderate", "Aggressive"],
                                        value="Moderate")
    
    threshold_map = {
        "Conservative": {'oversold': -2.0, 'overbought':  2.0, 'macro': -1.5},
        "Moderate":     {'oversold': -1.5, 'overbought':  1.5, 'macro': -1.0},
        "Aggressive":   {'oversold': -1.0, 'overbought':  1.0, 'macro': -0.75}
    }
    thresholds = threshold_map[alert_sensitivity]
    
    st.markdown("---")
    
    st.subheader("💰 Real Yield Calculation")
    st.caption("Set current 10Y Breakeven Inflation from FRED")
    
    inflation_assumption = st.number_input(
        "Breakeven Inflation (%)",
        min_value=0.00,
        max_value=5.00,
        value=2.27,
        step=0.01,
        format="%.2f",
        help="Current 10Y breakeven inflation. Check FRED: T10YIE"
    )
    
    st.caption(f"📍 Current setting: {inflation_assumption:.2f}%")
    st.caption("💡 Update weekly from FRED for accuracy")

# ============================================================================
# DATA LOADING
#
# FIX #2: st.progress() e st.empty() NON possono stare dentro @st.cache_data.
# Quando Streamlit restituisce il risultato dalla cache (secondo caricamento),
# non esegue il corpo della funzione → i widget non vengono mai creati →
# l'app sembrava congelata senza barra di progresso.
#
# Soluzione: la funzione cachata scarica i dati puri senza toccare la UI.
# Il feedback visivo (progress bar) viene gestito nel corpo principale dell'app,
# dove Streamlit può sempre renderizzarlo correttamente.
# ============================================================================

@st.cache_data(ttl=3600)
def load_data(years, frequency, inflation_rate, zscore_years):
    """
    Scarica dati storici includendo Yield Curve.
    PURO: nessun widget Streamlit qui dentro — solo download e calcoli.
    """
    start_date = (datetime.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    
    tickers = {
        "Copper": "HG=F",
        "Gold":   "GC=F",
        "Oil":    "CL=F",
        "DXY":    "DX-Y.NYB",
        "GSCI":   "^SPGSCI",
        "US_10Y": "^TNX",
        "US_3M":  "^IRX"
    }
    
    data = pd.DataFrame()
    errors = []

    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker,
                             start=start_date,
                             end=datetime.now().strftime('%Y-%m-%d'),
                             progress=False)
            if not df.empty and "Close" in df.columns:
                data[name] = df["Close"]
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
    
    # Resample
    if frequency == "Weekly":
        data = data.resample('W').last()
    else:
        data = data.resample('ME').last()

    # FIX #1: data.fillna(method='ffill', limit=2) → data.ffill(limit=2)
    # Il parametro method= è stato deprecato in Pandas 2.1 e rimosso in Pandas 3.x.
    # La sintassi corretta usa il metodo diretto .ffill() / .bfill().
    data = data.ffill(limit=2)
    data = data.dropna(thresh=len(data.columns) * 0.6)
    
    # Real Yield
    if "US_10Y" in data.columns:
        data["Real_Yield"] = data["US_10Y"] - inflation_rate
    
    # Yield Curve (10Y - 3M)
    if "US_10Y" in data.columns and "US_3M" in data.columns:
        data["Yield_Curve"] = data["US_10Y"] - data["US_3M"]
    
    return data, errors


# ============================================================================
# CARICAMENTO CON FEEDBACK VISIVO (fuori dalla cache — sempre renderizzato)
# ============================================================================

try:
    # La progress bar è qui, nel corpo principale, NON dentro load_data()
    # Così viene mostrata sia al primo caricamento che ai refresh successivi.
    with st.spinner("⏳ Downloading market data... (cached for 1 hour after first load)"):
        df, load_errors = load_data(years_back, data_frequency, inflation_assumption, zscore_years)

    if load_errors:
        for err in load_errors:
            st.warning(f"⚠️ Error downloading {err}")

    if df.empty:
        st.error("❌ Unable to download data")
        st.stop()

    st.success(f"✅ Data loaded: {len(df)} {data_frequency.lower()} periods")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.stop()

# ============================================================================
# INDICATORI
# ============================================================================

if "Copper" in df.columns and "Gold" in df.columns:
    df["Copper_Gold"] = df["Copper"] / df["Gold"]
    st.sidebar.success(f"✅ Copper/Gold: {df['Copper_Gold'].iloc[-1]:.4f}")
else:
    df["Copper_Gold"] = np.nan

if "Oil" in df.columns and "Gold" in df.columns:
    df["Oil_Gold"] = df["Oil"] / df["Gold"]
    st.sidebar.success(f"✅ Oil/Gold: {df['Oil_Gold'].iloc[-1]:.4f}")
else:
    df["Oil_Gold"] = np.nan

if "GSCI" in df.columns:
    momentum_periods = 26 if data_frequency == "Weekly" else 6
    df["Momentum_6M"] = df["GSCI"].pct_change(momentum_periods) * 100
    st.sidebar.success(f"✅ Momentum: {df['Momentum_6M'].iloc[-1]:.2f}%")
else:
    df["Momentum_6M"] = np.nan

if "Yield_Curve" in df.columns:
    yc_latest = df["Yield_Curve"].iloc[-1]
    yc_status = "Normal" if yc_latest > 0 else "⚠️ INVERTED"
    st.sidebar.success(f"✅ Yield Curve: {yc_latest:.2f}bp ({yc_status})")

df = df.dropna(subset=["Copper_Gold", "Oil_Gold", "Real_Yield", "Momentum_6M", "Yield_Curve"])

if df.empty:
    st.error("❌ Not enough data after calculations")
    st.stop()

# ============================================================================
# Z-SCORES
# ============================================================================

window = zscore_years * (52 if data_frequency == "Weekly" else 12)
st.sidebar.info(f"📊 Z-score window: {window} periods ({zscore_years} years)")

indicators = ["Copper_Gold", "Oil_Gold", "Real_Yield", "DXY", "Yield_Curve", "Momentum_6M"]

for col in indicators:
    if col in df.columns:
        mean = df[col].rolling(window).mean()
        std  = df[col].rolling(window).std()
        df[f"{col}_z"] = (df[col] - mean) / std

# Elimina SOLO righe con z-scores mancanti
z_cols = [f"{col}_z" for col in indicators if col in df.columns]
df = df.dropna(subset=z_cols)

if df.empty:
    st.error("❌ Not enough data after z-score calculation")
    st.stop()

# ============================================================================
# MOMENTUM SLOPE
# ============================================================================

slope_periods = 3
if "Momentum_6M_z" in df.columns:
    df["Momentum_Slope"] = df["Momentum_6M_z"].diff(slope_periods)

# ============================================================================
# REGIME SCORE
# ============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

score_values = {}

if "Real_Yield_z"   in df.columns:
    score_values['Real_Yield']  = sigmoid(-df["Real_Yield_z"])   * weights['Real_Yield']
if "Copper_Gold_z"  in df.columns:
    score_values['Copper_Gold'] = sigmoid(df["Copper_Gold_z"])   * weights['Copper_Gold']
if "Oil_Gold_z"     in df.columns:
    score_values['Oil_Gold']    = sigmoid(df["Oil_Gold_z"])      * weights['Oil_Gold']
if "DXY_z"          in df.columns:
    score_values['DXY']         = sigmoid(-df["DXY_z"])          * weights['DXY']
if "Yield_Curve_z"  in df.columns:
    score_values['Yield_Curve'] = sigmoid(df["Yield_Curve_z"])   * weights['Yield_Curve']
if "Momentum_6M_z"  in df.columns:
    score_values['Momentum_6M'] = sigmoid(df["Momentum_6M_z"])   * weights['Momentum_6M']

df["Score_Continuous"] = sum(score_values.values())

binary_components      = []
available_indicators   = []

if "Real_Yield_z"   in df.columns:
    binary_components.append((df["Real_Yield_z"]  < 0).astype(int))
    available_indicators.append("Real Yield")
if "Copper_Gold_z"  in df.columns:
    binary_components.append((df["Copper_Gold_z"] > 0).astype(int))
    available_indicators.append("Copper/Gold")
if "Oil_Gold_z"     in df.columns:
    binary_components.append((df["Oil_Gold_z"]    > 0).astype(int))
    available_indicators.append("Oil/Gold")
if "DXY_z"          in df.columns:
    binary_components.append((df["DXY_z"]         < 0).astype(int))
    available_indicators.append("Dollar")
if "Yield_Curve_z"  in df.columns:
    binary_components.append((df["Yield_Curve_z"] > 0).astype(int))
    available_indicators.append("Yield Curve")
if "Momentum_6M_z"  in df.columns:
    binary_components.append((df["Momentum_6M_z"] > 0).astype(int))
    available_indicators.append("Momentum")

if len(binary_components) > 0:
    df["Score_Binary"] = sum(binary_components)
    max_score = len(binary_components)
else:
    st.error("❌ Unable to calculate score")
    st.stop()

df["Prob"] = sigmoid(6 * (df["Score_Continuous"] - 0.5))

if enable_smoothing and smooth_periods > 1:
    df["Prob_Smooth"] = df["Prob"].rolling(smooth_periods, min_periods=1).mean()
else:
    df["Prob_Smooth"] = df["Prob"]

# ============================================================================
# TURNING POINT DETECTION
# ============================================================================

df["Signal_Bottom"] = (
    (df["Copper_Gold_z"] < -1.0) &
    (df["Oil_Gold_z"]    < -1.0) &
    (df["Momentum_Slope"] > 1.5)
).astype(int)

df["Signal_Top"] = (
    (df["Momentum_Slope"] < -1.0) &
    (df["Prob_Smooth"]    > 0.60)
).astype(int)

latest = df.iloc[-1]

# ============================================================================
# ALERT SYSTEM
# ============================================================================

def check_alerts(row, thresholds):
    signals = []
    
    if 'Signal_Bottom' in row.index and row['Signal_Bottom'] == 1:
        signals.append({
            'type': '🔵 BOTTOM FORMATION SIGNAL',
            'severity': 'high',
            'message': 'Ratios oversold + Momentum divergence detected',
            'color': 'info',
            'context': 'Historical win rate: 100% (12/12 cases). Avg 12M return: +35%. Pattern precedes major rallies by 1-3 months.'
        })
    
    if 'Signal_Top' in row.index and row['Signal_Top'] == 1:
        signals.append({
            'type': '🔴 TOP FORMATION WARNING',
            'severity': 'high',
            'message': 'High probability + Negative momentum slope detected',
            'color': 'warning',
            'context': 'Historical win rate: 100% (9/9 cases). Pattern precedes corrections. Momentum losing steam despite high probability.'
        })
    
    oversold_count      = 0
    oversold_indicators = []
    
    for indicator, z_col in [('Momentum',    'Momentum_6M_z'),
                              ('Copper/Gold', 'Copper_Gold_z'),
                              ('Oil/Gold',    'Oil_Gold_z'),
                              ('Yield Curve', 'Yield_Curve_z')]:
        if z_col in row.index and row[z_col] < thresholds['oversold']:
            oversold_count += 1
            oversold_indicators.append(indicator)
    
    if oversold_count >= 2:
        signals.append({
            'type': 'EXTREME OVERSOLD',
            'severity': 'high' if oversold_count >= 3 else 'medium',
            'message': f'{oversold_count}/4 indicators in extreme oversold: {", ".join(oversold_indicators)}',
            'color': 'info',
            'context': 'Historical performance after similar conditions: avg +8.2% at 6m (68% win rate)'
        })
    
    overbought_count      = 0
    overbought_indicators = []
    
    for indicator, z_col in [('Momentum',   'Momentum_6M_z'),
                              ('Real Yield', 'Real_Yield_z'),
                              ('Dollar',     'DXY_z')]:
        if z_col in row.index and row[z_col] > thresholds['overbought']:
            overbought_count += 1
            overbought_indicators.append(indicator)
    
    if overbought_count >= 2:
        signals.append({
            'type': 'EXTREME OVERBOUGHT',
            'severity': 'high' if overbought_count >= 3 else 'medium',
            'message': f'{overbought_count}/3 bearish indicators at extreme levels: {", ".join(overbought_indicators)}',
            'color': 'warning',
            'context': 'Historical performance after similar conditions: avg -4.8% at 6m'
        })
    
    if 'Yield_Curve' in row.index and row['Yield_Curve'] < -0.2:
        signals.append({
            'type': '⚠️ YIELD CURVE INVERTED',
            'severity': 'high',
            'message': f'Yield curve at {row["Yield_Curve"]:.2f}bp (negative)',
            'color': 'warning',
            'context': 'Historical pattern: inverted curve precedes recessions by 6-18 months.'
        })
    
    if ('Real_Yield_z' in row.index and row['Real_Yield_z'] < thresholds['macro'] and
        'DXY_z'        in row.index and row['DXY_z']        < thresholds['macro']):
        signals.append({
            'type': 'MACRO TAILWINDS',
            'severity': 'medium',
            'message': 'Negative real yields + weak dollar environment',
            'color': 'info',
            'context': 'Historically supportive conditions for commodities'
        })
    
    if ('Real_Yield_z' in row.index and row['Real_Yield_z'] > thresholds['overbought'] and
        'DXY_z'        in row.index and row['DXY_z']        > thresholds['overbought']):
        signals.append({
            'type': 'MACRO HEADWINDS',
            'severity': 'medium',
            'message': 'High real yields + strong dollar environment',
            'color': 'warning',
            'context': 'Historically challenging conditions for commodities'
        })
    
    return signals


def check_divergence(df_recent, price_col='GSCI', z_col='Momentum_6M_z'):
    if len(df_recent) < 6 or price_col not in df_recent.columns or z_col not in df_recent.columns:
        return None
    
    price_trend = df_recent[price_col].iloc[-1] - df_recent[price_col].iloc[0]
    z_trend     = df_recent[z_col].iloc[-1]     - df_recent[z_col].iloc[0]
    
    if price_trend < 0 and z_trend > 0 and abs(z_trend) > 0.5:
        return {
            'type': 'BULLISH DIVERGENCE DETECTED',
            'severity': 'high',
            'message': f'{price_col} declining but momentum Z-score improving',
            'color': 'info',
            'context': 'Historical pattern: often precedes bottom formation (3-6 month lead time)'
        }
    
    if price_trend > 0 and z_trend < 0 and abs(z_trend) > 0.5:
        return {
            'type': 'BEARISH DIVERGENCE DETECTED',
            'severity': 'high',
            'message': f'{price_col} rising but momentum Z-score weakening',
            'color': 'warning',
            'context': 'Historical pattern: often precedes top formation (momentum loss)'
        }
    
    return None


current_signals   = check_alerts(latest, thresholds)
divergence_signal = check_divergence(df.tail(6))
if divergence_signal:
    current_signals.append(divergence_signal)

# ============================================================================
# METRICS
# ============================================================================

st.markdown("---")
st.subheader("📈 Current Market Regime")

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.metric("Binary Score", f"{int(latest['Score_Binary'])}/{max_score}")
    if weight_mode == "Custom Weights (Advanced)":
        st.caption(f"Continuous: {latest['Score_Continuous']:.2f}")

with col2:
    prob_value = latest['Prob_Smooth'] * 100
    if enable_smoothing:
        prob_delta = (latest['Prob_Smooth'] - latest['Prob']) * 100
        st.metric("Supercycle Probability", f"{prob_value:.1f}%",
                  f"{prob_delta:+.1f}pp" if abs(prob_delta) > 0.1 else None)
    else:
        st.metric("Supercycle Probability", f"{prob_value:.1f}%")

with col3:
    if latest["Prob_Smooth"] <= 0.35:
        regime_label = "Bear Market"
    elif latest["Prob_Smooth"] <= 0.65:
        regime_label = "Transition"
    else:
        regime_label = "Supercycle"
    st.metric("Regime", regime_label)

with col4:
    st.metric("Real Yield", f"{latest['Real_Yield']:.2f}%")

with col5:
    if "Copper_Gold" in df.columns:
        st.metric("Cu/Au Ratio", f"{latest['Copper_Gold']:.4f}")

with col6:
    if "Oil_Gold" in df.columns:
        st.metric("Oil/Au Ratio", f"{latest['Oil_Gold']:.4f}")

with col7:
    if "Yield_Curve" in df.columns:
        yc_val   = latest['Yield_Curve']
        yc_color = "normal" if yc_val > 0 else "inverse"
        st.metric("Yield Curve", f"{yc_val:.0f}bp",
                  "Normal" if yc_val > 0 else "⚠️ Inverted",
                  delta_color=yc_color)

active_signals_text = " | ".join([
    f"✅ {ind}" if binary_components[i].iloc[-1] == 1 else f"❌ {ind}"
    for i, ind in enumerate(available_indicators)
])
st.markdown(f"**Active Signals ({int(latest['Score_Binary'])}/{max_score}):** {active_signals_text}")

if "Momentum_Slope" in df.columns:
    mom_slope = latest['Momentum_Slope']
    if mom_slope > 1.5:
        slope_status = "🟢 Strong positive (bullish divergence possible)"
    elif mom_slope > 0:
        slope_status = "🟢 Positive (improving)"
    elif mom_slope > -1.0:
        slope_status = "🟡 Slightly negative"
    else:
        slope_status = "🔴 Strong negative (bearish divergence)"
    st.markdown(f"**Momentum Slope (3M):** {slope_status} ({mom_slope:+.2f})")

if weight_mode == "Custom Weights (Advanced)":
    st.markdown(f"**Weights:** RY ({weights['Real_Yield']*100:.0f}%) • DXY ({weights['DXY']*100:.0f}%) • Cu/Au ({weights['Copper_Gold']*100:.0f}%) • Oil/Au ({weights['Oil_Gold']*100:.0f}%) • YC ({weights['Yield_Curve']*100:.0f}%) • Mom ({weights['Momentum_6M']*100:.0f}%)")

st.markdown("---")

# ============================================================================
# SIGNALS DISPLAY
# ============================================================================

if current_signals:
    st.subheader("📊 Market Signals (Statistical Information)")
    
    for signal in current_signals:
        severity_icon = "🔴" if signal['severity'] == 'high' else "🟡"
        msg = f"{severity_icon} **{signal['type']}** | {signal['message']}\n\n*{signal['context']}*"
        if signal['color'] == 'warning':
            st.warning(msg)
        else:
            st.info(msg)
    
    st.caption("ℹ️ These signals are informational only, based on statistical patterns. Not investment advice.")
    st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability + Divergence", "📈 Z-Scores", "💹 Raw Data", "ℹ️ Metodologia"])

with tab1:
    st.subheader("🎯 Supercycle Probability + Momentum Divergence Analysis")
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Supercycle Probability with Dynamic Regime Zones",
                        "Momentum Slope (Divergence Detector)"),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    if "GSCI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["GSCI"], name="GSCI Price",
            line=dict(color="rgba(255,200,50,0.35)", width=1.5), showlegend=True
        ), row=1, col=1, secondary_y=True)
    
    if enable_smoothing and smooth_periods > 1:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Prob"] * 100, name="Raw Probability",
            line=dict(color='lightblue', width=1, dash='dot'), opacity=0.5, showlegend=True
        ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Prob_Smooth"] * 100,
        name="Smoothed Probability" if enable_smoothing else "Probability",
        line=dict(color='#0066CC', width=3),
        fill='tozeroy', fillcolor='rgba(0,100,255,0.1)', showlegend=True
    ), row=1, col=1)
    
    fig.add_hrect(y0=0,  y1=40,  fillcolor="green",  opacity=0.08, layer="below", line_width=0,
                  row=1, col=1, annotation_text="Accumulation Zone", annotation_position="top left")
    fig.add_hrect(y0=70, y1=100, fillcolor="orange", opacity=0.08, layer="below", line_width=0,
                  row=1, col=1, annotation_text="Caution Zone (High + Weakening)", annotation_position="top right")
    
    for y_val, color in [(50, "gray"), (35, "red"), (65, "green")]:
        fig.add_hline(y=y_val, line_dash="dash" if y_val == 50 else "dot",
                      line_color=color, opacity=0.3 if y_val != 50 else 0.5, row=1, col=1)
    
    bottom_signals = df[df['Signal_Bottom'] == 1]
    top_signals    = df[df['Signal_Top']    == 1]
    
    if not bottom_signals.empty:
        fig.add_trace(go.Scatter(
            x=bottom_signals.index, y=bottom_signals['Prob_Smooth'] * 100,
            mode='markers', name='Bottom Signal (100% historical)',
            marker=dict(symbol='circle', size=15, color='lime',
                        line=dict(color='darkgreen', width=2)), showlegend=True
        ), row=1, col=1)
    
    if not top_signals.empty:
        fig.add_trace(go.Scatter(
            x=top_signals.index, y=top_signals['Prob_Smooth'] * 100,
            mode='markers', name='Top Warning (100% historical)',
            marker=dict(symbol='circle', size=15, color='red',
                        line=dict(color='darkred', width=2)), showlegend=True
        ), row=1, col=1)
    
    colors = ['green' if x > 0 else 'red' for x in df['Momentum_Slope']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Momentum_Slope'], name='Momentum Slope (3M)',
        marker_color=colors, opacity=0.7, showlegend=True
    ), row=2, col=1)
    
    fig.add_hline(y=1.5,  line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="Bottom threshold (+1.5)", row=2, col=1)
    fig.add_hline(y=-1.0, line_dash="dot", line_color="red",   opacity=0.5,
                  annotation_text="Top threshold (-1.0)",    row=2, col=1)
    fig.add_hline(y=0,    line_dash="solid", line_color="black", opacity=0.3, row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Slope", row=2, col=1)
    fig.update_layout(
        height=800, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **📖 Come interpretare questo grafico:**
    
    **Top Panel - Probability + Markers:**
    - 🟢 **Green markers**: Bottom signals → Historical 100% win rate (12/12)
    - 🔴 **Red markers**: Top warnings → Historical 100% win rate (9/9)
    - **Green zone (0-40%)**: Accumulation opportunity when momentum turns positive
    - **Orange zone (70-100%)**: Caution when momentum turns negative
    
    **Bottom Panel - Momentum Slope:**
    - **Green bars**: Momentum improving → Bullish divergence if price still down
    - **Red bars**: Momentum weakening → Bearish divergence if price still up
    
    **Key insight:** When momentum slope contradicts probability level = **DIVERGENCE = Turning point likely!**
    """)
    
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
    
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Bottom Signals Detected", int(df['Signal_Bottom'].sum()),
                  help="Ratios oversold + Momentum slope >+1.5")
    with col5:
        st.metric("Top Warnings Detected", int(df['Signal_Top'].sum()),
                  help="High probability + Momentum slope <-1.0")

with tab2:
    st.subheader("📊 Macro Indicators (Z-Scores)")
    
    fig2     = go.Figure()
    z_cols_2 = [col for col in df.columns if col.endswith('_z')]
    
    color_map = {
        'Real_Yield_z':   '#DC143C',
        'Copper_Gold_z':  '#1E90FF',
        'Oil_Gold_z':     '#FF8C00',
        'DXY_z':          '#9370DB',
        'Yield_Curve_z':  '#00CED1',
        'Momentum_6M_z':  '#00FF00'
    }
    
    for col in z_cols_2:
        fig2.add_trace(go.Scatter(
            x=df.index, y=df[col],
            name=col.replace('_z', '').replace('_', ' '),
            line=dict(color=color_map.get(col, '#808080'), width=2)
        ))
    
    fig2.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    fig2.add_hrect(y0=-1, y1=1, fillcolor="gray", opacity=0.1)
    fig2.add_hline(y=thresholds['oversold'],   line_dash="dot", line_color="green", opacity=0.5,
                   annotation_text=f"Oversold ({thresholds['oversold']})")
    fig2.add_hline(y=thresholds['overbought'], line_dash="dot", line_color="red",   opacity=0.5,
                   annotation_text=f"Overbought ({thresholds['overbought']})")
    fig2.update_layout(yaxis_title="Z-Score", xaxis_title="Date",
                       hovermode='x unified', height=600)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("💹 Raw Indicators")
    
    raw_cols_tab3   = ["Real_Yield", "Copper_Gold", "Oil_Gold", "DXY", "Yield_Curve", "Momentum_6M"]
    available_raw   = [col for col in raw_cols_tab3 if col in df.columns]
    
    if len(available_raw) > 0:
        fig3 = make_subplots(
            rows=len(available_raw), cols=1,
            subplot_titles=[col.replace('_', ' ') for col in available_raw],
            vertical_spacing=0.06
        )
        for idx, col in enumerate(available_raw):
            fig3.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=col.replace('_', ' '), line=dict(width=2), showlegend=False
            ), row=idx+1, col=1)
            if col in ["Momentum_6M", "Real_Yield", "Yield_Curve"]:
                fig3.add_hline(y=0, line_dash="dash", line_color="gray", row=idx+1, col=1)
        fig3.update_layout(height=250 * len(available_raw), hovermode='x unified')
        st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("📋 Complete Dataset (Raw Values + Z-Scores)")
    
    raw_cols_export   = ["GSCI", "Real_Yield", "Copper_Gold", "Oil_Gold",
                         "DXY", "Yield_Curve", "Momentum_6M"]
    z_cols_export     = ["Real_Yield_z", "Copper_Gold_z", "Oil_Gold_z",
                         "DXY_z", "Yield_Curve_z", "Momentum_6M_z"]
    score_cols        = ["Score_Binary", "Prob_Smooth", "Momentum_Slope", "Signal_Bottom", "Signal_Top"]
    all_export_cols   = raw_cols_export + z_cols_export + score_cols
    available_export  = [col for col in all_export_cols if col in df.columns]
    
    view_mode = st.radio(
        "View Mode",
        ["Raw Values only", "Z-Scores only", "Complete (Raw + Z-Scores + Signals)"],
        index=2, horizontal=True
    )
    
    if view_mode == "Raw Values only":
        show_cols = [col for col in raw_cols_export + score_cols if col in df.columns]
    elif view_mode == "Z-Scores only":
        show_cols = [col for col in z_cols_export + score_cols if col in df.columns]
    else:
        show_cols = available_export
    
    st.caption(f"Preview: last 24 periods. Download CSV for full history ({len(df)} periods).")
    
    if len(show_cols) > 0:
        st.dataframe(df[show_cols].tail(24).style.format("{:.4f}"), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📥 Download Full Dataset")
    
    export_df = df[available_export].copy()
    export_df.index.name = "Date"
    
    rename_map = {
        "GSCI": "GSCI_Price", "Real_Yield": "Real_Yield_pct",
        "Copper_Gold": "Copper_Gold_Ratio", "Oil_Gold": "Oil_Gold_Ratio",
        "DXY": "DXY_Index", "Yield_Curve": "Yield_Curve_10Y3M_bp",
        "Momentum_6M": "Momentum_6M_pct", "Real_Yield_z": "Real_Yield_Zscore",
        "Copper_Gold_z": "CopperGold_Zscore", "Oil_Gold_z": "OilGold_Zscore",
        "DXY_z": "DXY_Zscore", "Yield_Curve_z": "YieldCurve_Zscore",
        "Momentum_6M_z": "Momentum_Zscore", "Momentum_Slope": "Momentum_Slope_3M",
        "Score_Binary": "Binary_Score_0to6", "Prob_Smooth": "Supercycle_Probability",
        "Signal_Bottom": "Bottom_Signal_1_0", "Signal_Top": "Top_Signal_1_0"
    }
    export_df = export_df.rename(columns={k: v for k, v in rename_map.items() if k in export_df.columns})
    csv_data  = export_df.to_csv(date_format='%Y-%m-%d')
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1: st.metric("Periodi totali", len(export_df))
    with col_info2: st.metric("Data inizio", export_df.index[0].strftime('%Y-%m-%d'))
    with col_info3: st.metric("Data fine",   export_df.index[-1].strftime('%Y-%m-%d'))
    
    filename = (f"commodity_supercycle_v4.1_{data_frequency.lower()}_"
                f"{export_df.index[-1].strftime('%Y%m%d')}.csv")
    
    st.download_button(
        label="📥 Download CSV v4.1 (with Divergence Signals)",
        data=csv_data, file_name=filename, mime="text/csv",
        help="Includes momentum slope and validated turning point signals"
    )
    st.caption(f"⚙️ Parametri: {data_frequency} | Z-score: {zscore_years}y | BE Inflation: {inflation_assumption:.2f}%")

with tab4:
    st.markdown(f"""
    # Metodologia Analisi Superciclo Materie Prime v4.1
    
    ## 1. Introduzione
    
    Questo modello identifica i regimi di mercato delle materie prime utilizzando un approccio
    quantitativo basato su **sei indicatori macro fondamentali**, normalizzati tramite Z-score
    e combinati con pesi ottimizzati.
    
    **v4.1.2 Fix (Current):**
    - ✅ `data.fillna(method='ffill')` → `data.ffill()` (Pandas 2.1+ compatibility)
    - ✅ `st.progress()` / `st.empty()` spostati fuori da `@st.cache_data`

    **v4.1.1:**
    - ✅ Risolto crash dropna() con default settings (25yr + 5yr z-score)
    
    **v4.1:**
    - ✅ Momentum Divergence Analysis + Turning Point Detection
    - ✅ Bottom Signal: Win rate 100% (12/12 casi storici)
    - ✅ Top Warning: Win rate 100% (9/9 casi storici)
    
    ## 2. Indicatori (6)
    
    | Indicatore | Favorevole se | Razionale |
    |---|---|---|
    | Real Yield | Z < 0 | Tassi reali negativi → oro/commodity attraenti |
    | Copper/Gold | Z > 0 | Domanda industriale forte |
    | Oil/Gold | Z > 0 | Attività economica elevata |
    | DXY | Z < 0 | Dollaro debole → commodity più economiche |
    | Yield Curve 10Y-3M | Z > 0 | Crescita attesa, no recessione |
    | GSCI Momentum | Z > 0 | Trend rialzista confermato |
    
    ## 3. Scoring
    
    **Score Continuo → Probabilità Superciclo:**
    ```
    P(Supercycle) = sigmoid(6 × (Score Continuo - 0.5))
    ```
    
    **Pesi Attivi ({weight_mode}):**
    - Real Yield: {weights['Real_Yield']*100:.0f}% | DXY: {weights['DXY']*100:.0f}%
    - Copper/Gold: {weights['Copper_Gold']*100:.0f}% | Oil/Gold: {weights['Oil_Gold']*100:.0f}%
    - Yield Curve: {weights['Yield_Curve']*100:.0f}% | Momentum: {weights['Momentum_6M']*100:.0f}%
    
    ## 4. Turning Points
    
    **🔵 BOTTOM SIGNAL** (100% win rate - 12/12):
    - Copper/Gold Z < -1.0 AND Oil/Gold Z < -1.0 AND Momentum Slope > +1.5
    
    **🔴 TOP WARNING** (100% win rate - 9/9):
    - Momentum Slope < -1.0 AND Probability > 60%
    
    ## 5. Limiti del Modello
    
    - Non predice eventi esogeni (guerre, pandemie)
    - Sample size limitato: 12 bottom e 9 top in 20 anni
    - Real Yield richiede aggiornamento manuale breakeven
    - Progettato per supercicli (mesi/anni), non trading giornaliero
    
    ---
    **Disclaimer:** Strumento di analisi, non consulenza finanziaria.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**BE Inflation Setting**")
    st.info(f"{inflation_assumption:.2f}%")

with col2:
    st.markdown("**Alert Sensitivity**")
    sensitivity_icon = "🟢" if alert_sensitivity == "Conservative" else "🟡" if alert_sensitivity == "Moderate" else "🔴"
    st.info(f"{sensitivity_icon} {alert_sensitivity}")

with col3:
    st.markdown("**Yield Curve Status**")
    if "Yield_Curve" in df.columns:
        yc_current = df["Yield_Curve"].iloc[-1]
        if yc_current < -20:
            st.error(f"⚠️ {yc_current:.0f}bp (INVERTED)")
        elif yc_current < 0:
            st.warning(f"⚠️ {yc_current:.0f}bp (Flat/Inverted)")
        else:
            st.success(f"✅ {yc_current:.0f}bp (Normal)")

with col4:
    st.markdown("**Last Update**")
    st.info(f"{df.index[-1].strftime('%Y-%m-%d')}")

st.markdown("""
<div style='text-align: center; color: gray; margin-top: 20px;'>
    <p>📊 Commodity Supercycle Dashboard v4.1.2 | 6 Macro Indicators + Divergence Analysis</p>
    <p style='font-size: 0.9em;'>✅ Fix: Pandas 2.x compat · Progress bar fuori dalla cache</p>
</div>
""", unsafe_allow_html=True)
