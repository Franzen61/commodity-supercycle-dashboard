"""
COMMODITY SUPERCYCLE DASHBOARD
==============================
Dashboard interattiva per analizzare il superciclo delle materie prime

Streamlit App - Versione 1.0
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats

# ============================================================================
# CONFIGURAZIONE PAGINA
# ============================================================================

st.set_page_config(
    page_title="Commodity Supercycle Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titolo
st.title("📊 Superciclo Materie Prime - Dashboard Interattiva")
st.markdown("*Analisi correlazioni storiche e identificazione fasi cicliche*")
st.markdown("---")

# ============================================================================
# SIDEBAR - CONFIGURAZIONE
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configurazione")
    
    # Selezione periodo
    st.subheader("📅 Periodo Analisi")
    years_back = st.slider("Anni di storico", 5, 25, 15)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back*365)
    
    st.write(f"Dal: {start_date.strftime('%Y-%m-%d')}")
    st.write(f"Al: {end_date.strftime('%Y-%m-%d')}")
    
    # Selezione commodity index principale
    st.subheader("📦 Indice Commodity")
    commodity_choice = st.selectbox(
        "Scegli indice principale",
        ["DBC (ETF - più liquido)", "BCOM (Bloomberg Index)", "GSG (iShares GSCI)"]
    )
    
    ticker_map = {
        "DBC (ETF - più liquido)": "DBC",
        "BCOM (Bloomberg Index)": "^BCOM",
        "GSG (iShares GSCI)": "GSG"
    }
    main_commodity = ticker_map[commodity_choice]
    
    # Finestra correlazioni rolling
    st.subheader("📊 Parametri Analisi")
    rolling_window = st.slider("Finestra correlazioni (giorni)", 30, 500, 252)
    
    st.markdown("---")
    st.info("💡 I dati vengono scaricati in tempo reale da Yahoo Finance")

# ============================================================================
# FUNZIONI
# ============================================================================

@st.cache_data(ttl=3600)  # Cache per 1 ora
def load_data(start, end):
    """Scarica dati da Yahoo Finance con caching"""
    
    tickers = {
        'Commodity': main_commodity,
        'Copper': 'HG=F',
        'Gold': 'GC=F',
        'Oil': 'CL=F',
        'DXY': 'DX-Y.NYB',
        'US_10Y': '^TNX',
        'US_2Y': '^IRX',
        'SPX': '^GSPC',
        'VIX': '^VIX',
        'TIP': 'TIP',
    }
    
    data = pd.DataFrame()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, ticker) in enumerate(tickers.items()):
        try:
            status_text.text(f"Scaricamento {name}...")
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                data[name] = df['Close']
            progress_bar.progress((idx + 1) / len(tickers))
        except:
            pass
    
    progress_bar.empty()
    status_text.empty()
    
    return data.fillna(method='ffill', limit=5)

def calculate_indicators(df):
    """Calcola indicatori derivati"""
    indicators = pd.DataFrame(index=df.index)
    
    # Copper/Gold Ratio
    if 'Copper' in df.columns and 'Gold' in df.columns:
        indicators['Copper_Gold_Ratio'] = df['Copper'] / df['Gold']
    
    # Yield Curve
    if 'US_10Y' in df.columns and 'US_2Y' in df.columns:
        indicators['Yield_Curve'] = df['US_10Y'] - df['US_2Y']
    
    # Commodity vs Equity
    if 'Commodity' in df.columns and 'SPX' in df.columns:
        indicators['Commodity_vs_SPX'] = (df['Commodity'] / df['SPX']) * 100
    
    # Momentum 6M
    if 'Commodity' in df.columns:
        indicators['Momentum_6M'] = df['Commodity'].pct_change(126) * 100
    
    return indicators

def identify_regime(df, commodity_col='Commodity'):
    """Identifica regime di mercato attuale"""
    if commodity_col not in df.columns:
        return "N/A", "gray"
    
    momentum = df[commodity_col].pct_change(63).iloc[-1] * 100  # 3 mesi
    volatility = df[commodity_col].pct_change(1).tail(63).std() * 100 * np.sqrt(252)
    
    # Logica semplificata
    if momentum > 10 and volatility < 20:
        return "🚀 ESPANSIONE", "green"
    elif momentum > 10 and volatility > 20:
        return "🔥 BOOM SPECULATIVO", "orange"
    elif momentum < -5 and volatility > 20:
        return "📉 CONTRAZIONE", "red"
    elif momentum < -5:
        return "❄️ RECESSIONE", "blue"
    else:
        return "⚖️ CONSOLIDAMENTO", "gray"

# ============================================================================
# CARICAMENTO DATI
# ============================================================================

with st.spinner("🔄 Caricamento dati da Yahoo Finance..."):
    data = load_data(start_date, end_date)
    indicators = calculate_indicators(data)

if data.empty:
    st.error("❌ Impossibile scaricare i dati. Verifica la connessione.")
    st.stop()

st.success(f"✅ Dati caricati: {len(data)} giorni, {len(data.columns)} strumenti")

# ============================================================================
# METRICS - KPI PRINCIPALI
# ============================================================================

st.header("📈 Situazione Attuale")

col1, col2, col3, col4, col5 = st.columns(5)

# Regime attuale
regime, regime_color = identify_regime(data)

with col1:
    st.metric("Regime Mercato", regime)

# Prezzo commodity
if 'Commodity' in data.columns:
    current_price = data['Commodity'].iloc[-1]
    price_change = ((current_price / data['Commodity'].iloc[-2]) - 1) * 100
    with col2:
        st.metric("Commodity Index", f"{current_price:.2f}", f"{price_change:+.2f}%")

# Dollar Index
if 'DXY' in data.columns:
    dxy_current = data['DXY'].iloc[-1]
    dxy_change = ((dxy_current / data['DXY'].iloc[-2]) - 1) * 100
    with col3:
        st.metric("Dollar Index", f"{dxy_current:.2f}", f"{dxy_change:+.2f}%")

# Yield Curve
if 'Yield_Curve' in indicators.columns:
    yc_current = indicators['Yield_Curve'].iloc[-1]
    yc_status = "⚠️ INVERTITA" if yc_current < 0 else "✅ NORMALE"
    with col4:
        st.metric("Yield Curve (10Y-2Y)", f"{yc_current:.2f}bp", yc_status)

# Copper/Gold
if 'Copper_Gold_Ratio' in indicators.columns:
    cg_current = indicators['Copper_Gold_Ratio'].iloc[-1]
    cg_change = ((cg_current / indicators['Copper_Gold_Ratio'].iloc[-2]) - 1) * 100
    with col5:
        st.metric("Copper/Gold Ratio", f"{cg_current:.4f}", f"{cg_change:+.2f}%")

st.markdown("---")

# ============================================================================
# GRAFICI INTERATTIVI
# ============================================================================

# Tab per organizzare le visualizzazioni
tab1, tab2, tab3, tab4 = st.tabs(["📊 Prezzi & Performance", "🔗 Correlazioni", "📉 Indicatori", "📋 Dati"])

# TAB 1: PREZZI & PERFORMANCE
with tab1:
    st.subheader("Performance Normalizzata (Base 100)")
    
    # Selezione strumenti da plottare
    available_cols = [col for col in data.columns if data[col].notna().sum() > 100]
    selected_instruments = st.multiselect(
        "Seleziona strumenti da visualizzare",
        available_cols,
        default=available_cols[:5] if len(available_cols) >= 5 else available_cols
    )
    
    if selected_instruments:
        # Normalizza a base 100
        normalized = (data[selected_instruments] / data[selected_instruments].iloc[0]) * 100
        
        fig = go.Figure()
        for col in selected_instruments:
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized[col],
                name=col,
                mode='lines'
            ))
        
        fig.update_layout(
            title="Performance Comparativa",
            xaxis_title="Data",
            yaxis_title="Valore Normalizzato (Base 100)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Rendimenti recenti
    st.subheader("📊 Rendimenti Recenti")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'Commodity' in data.columns:
        returns_1w = (data['Commodity'].iloc[-1] / data['Commodity'].iloc[-5] - 1) * 100
        returns_1m = (data['Commodity'].iloc[-1] / data['Commodity'].iloc[-21] - 1) * 100
        returns_3m = (data['Commodity'].iloc[-1] / data['Commodity'].iloc[-63] - 1) * 100
        returns_1y = (data['Commodity'].iloc[-1] / data['Commodity'].iloc[-252] - 1) * 100
        
        col1.metric("1 Settimana", f"{returns_1w:+.2f}%")
        col2.metric("1 Mese", f"{returns_1m:+.2f}%")
        col3.metric("3 Mesi", f"{returns_3m:+.2f}%")
        col4.metric("1 Anno", f"{returns_1y:+.2f}%")

# TAB 2: CORRELAZIONI
with tab2:
    st.subheader("Matrice Correlazioni - Periodo Completo")
    
    # Calcola correlazioni
    corr_matrix = data.corr()
    
    # Heatmap con plotly
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdYlGn',
        aspect='auto',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Correlazioni Rolling con Commodity Index")
    
    if 'Commodity' in data.columns:
        # Calcola rolling correlations
        rolling_corr = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            if col != 'Commodity':
                rolling_corr[col] = data['Commodity'].rolling(rolling_window).corr(data[col])
        
        # Plot
        fig = go.Figure()
        for col in rolling_corr.columns:
            if rolling_corr[col].notna().sum() > 0:
                fig.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr[col],
                    name=col,
                    mode='lines'
                ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.update_layout(
            title=f"Correlazioni Rolling ({rolling_window} giorni)",
            xaxis_title="Data",
            yaxis_title="Correlazione",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: INDICATORI
with tab3:
    st.subheader("Indicatori Derivati")
    
    # Plot ogni indicatore
    for col in indicators.columns:
        if indicators[col].notna().sum() > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators[col],
                mode='lines',
                name=col,
                fill='tozeroy'
            ))
            
            if 'Yield_Curve' in col:
                fig.add_hline(y=0, line_dash="dash", line_color="red", 
                             annotation_text="Inversione")
            
            fig.update_layout(
                title=col.replace('_', ' '),
                xaxis_title="Data",
                yaxis_title="Valore",
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

# TAB 4: DATI
with tab4:
    st.subheader("📋 Dati Grezzi")
    
    # Mostra ultimi dati
    st.write("**Ultimi 20 giorni:**")
    st.dataframe(data.tail(20).style.format("{:.2f}"), use_container_width=True)
    
    # Download dati
    st.subheader("💾 Download Dati")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = data.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Scarica Dati Prezzi (CSV)",
            data=csv_data,
            file_name=f"commodity_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_indicators = indicators.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Scarica Indicatori (CSV)",
            data=csv_indicators,
            file_name=f"commodity_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📊 <b>Commodity Supercycle Dashboard</b> v1.0</p>
    <p><i>Dati in tempo reale da Yahoo Finance • Aggiornamento automatico ogni ora</i></p>
</div>
""", unsafe_allow_html=True)
