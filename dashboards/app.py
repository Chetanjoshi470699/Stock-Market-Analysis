"""
dashboards/app.py
==================
Premium Stock Market Analysis Dashboard (Streamlit)

Run with:
    streamlit run dashboards/app.py
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Path fix so src/ imports work from dashboards/ ───────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.data_collection import StockDataCollector
from src.preprocessing import StockPreprocessor
from src.analysis import TechnicalAnalysis, PortfolioMetrics
from src.model import StockPredictor
from src.visualization import StockVisualizer

# ─────────────────────────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockVision Pro · Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  Premium CSS Injection
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }

  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 40%, #0a1628 100%);
    background-attachment: fixed;
  }

  /* ── Hide Streamlit Chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(13,21,38,0.98) 0%, rgba(10,14,26,0.98) 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
    backdrop-filter: blur(20px);
  }
  [data-testid="stSidebar"] .element-container { padding: 0 0.5rem; }

  /* ── Hero Header ── */
  .hero-header {
    background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(154,117,234,0.08) 50%, rgba(72,187,120,0.06) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #9a75ea, #48bb78);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
  }
  .hero-subtitle {
    color: rgba(163,191,231,0.75);
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: 0.4rem;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(72,187,120,0.15);
    border: 1px solid rgba(72,187,120,0.35);
    color: #68d391;
    padding: 0.2rem 0.75rem;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
  }

  /* ── Section Title ── */
  .section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #a0bbea;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,179,237,0.3), transparent);
    margin-left: 0.5rem;
  }

  /* ── Metric Cards ── */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(16px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-start, #63b3ed), var(--accent-end, #9a75ea));
    opacity: 0.7;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99,179,237,0.3);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 20px rgba(99,179,237,0.08);
  }
  .metric-label {
    font-size: 0.73rem;
    font-weight: 600;
    color: rgba(163,191,231,0.65);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8f0fe;
    font-family: 'Roboto Mono', monospace;
    line-height: 1;
  }
  .metric-delta {
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.35rem;
  }
  .delta-pos { color: #68d391; }
  .delta-neg { color: #fc8181; }
  .delta-neu { color: #a0bbea; }

  /* ── Chart Containers ── */
  .chart-container {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 18px;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease;
  }
  .chart-container:hover { border-color: rgba(99,179,237,0.22); }

  /* ── Info Chips ── */
  .chip {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 100px;
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .chip-blue  { background: rgba(99,179,237,0.15); color: #90cdf4; border: 1px solid rgba(99,179,237,0.25); }
  .chip-green { background: rgba(72,187,120,0.15);  color: #68d391; border: 1px solid rgba(72,187,120,0.25); }
  .chip-red   { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.25); }
  .chip-yellow{ background: rgba(246,224,94,0.12);  color: #f6e05e; border: 1px solid rgba(246,224,94,0.2); }
  .chip-purple{ background: rgba(154,117,234,0.15); color: #b794f4; border: 1px solid rgba(154,117,234,0.25); }

  /* ── Sidebar Labels ── */
  [data-testid="stSidebar"] label {
    color: #a0bbea !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
  }
  [data-testid="stSidebar"] .stSelectbox > div > div,
  [data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #e8f0fe !important;
  }
  [data-testid="stSidebar"] .stSlider .stSlider { color: #63b3ed; }
  [data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #3182ce, #553c9a) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.65rem !important;
    font-size: 0.9rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(49,130,206,0.35) !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(49,130,206,0.5) !important;
  }

  /* ── Table ── */
  .eval-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }
  .eval-table th {
    color: rgba(163,191,231,0.7);
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid rgba(99,179,237,0.15);
    text-align: left;
  }
  .eval-table td {
    padding: 0.75rem 1rem;
    color: #e8f0fe;
    font-family: 'Roboto Mono', monospace;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .eval-table tr:hover td { background: rgba(99,179,237,0.05); }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #63b3ed !important; }

  /* ── Alerts ── */
  .stAlert { border-radius: 12px !important; }

  /* ── Tab styling ── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 0.25rem;
    gap: 0.25rem;
    border: 1px solid rgba(99,179,237,0.1);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    color: rgba(163,191,231,0.7);
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.45rem 1rem;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(49,130,206,0.35), rgba(85,60,154,0.35)) !important;
    color: #e8f0fe !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  Helpers / Caching
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(ticker: str, start_date: str):
    """Full pipeline: fetch → preprocess → indicators. Cached."""
    collector = StockDataCollector([ticker])
    raw = collector.fetch_single(ticker, start=start_date)
    if raw.empty:
        return None, None
    processed = StockPreprocessor(raw, ticker=ticker).run(save=False)
    ta = TechnicalAnalysis(processed)
    indicators = ta.add_all_indicators()
    return processed, indicators


@st.cache_data(show_spinner=False)
def run_model(ticker: str, start_date: str, n_estimators: int, horizon: int):
    """Train RF model & return metrics + predictions. Cached."""
    _, df = load_and_process(ticker, start_date)
    if df is None:
        return None, None
    predictor = StockPredictor(df, target_col="close", horizon=horizon)
    predictor.train_random_forest(n_estimators=n_estimators)
    metrics = predictor.evaluate()
    # Generate predictions for test period
    preds = predictor.model.predict(predictor.X_test)
    return metrics, preds


def fmt_pct(val: float, decimals: int = 2) -> str:
    return f"{val*100:+.{decimals}f}%"

def fmt_num(val: float, decimals: int = 2) -> str:
    return f"{val:,.{decimals}f}"

def signal_chip(rsi: float) -> str:
    if rsi >= 70:
        return '<span class="chip chip-red">⚠ Overbought</span>'
    elif rsi <= 30:
        return '<span class="chip chip-green">✦ Oversold</span>'
    else:
        return '<span class="chip chip-blue">◎ Neutral</span>'


# ─────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0; text-align:center;">
      <div style="font-size:2rem; margin-bottom:0.3rem;">📈</div>
      <div style="font-size:1.15rem; font-weight:800; color:#e8f0fe;">StockVision Pro</div>
      <div style="font-size:0.73rem; color:rgba(163,191,231,0.6); margin-top:0.2rem;">Advanced Market Analytics</div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(99,179,237,0.15);margin:0.8rem 0 1.2rem 0;">
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.78rem;font-weight:700;color:rgba(163,191,231,0.6);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.4rem;">Ticker</p>', unsafe_allow_html=True)
    ticker = st.text_input("", value="AAPL", label_visibility="collapsed").upper().strip()

    st.markdown('<p style="font-size:0.78rem;font-weight:700;color:rgba(163,191,231,0.6);letter-spacing:0.08em;text-transform:uppercase;margin:0.8rem 0 0.4rem 0;">Date Range</p>', unsafe_allow_html=True)
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("From", value=datetime(2021, 1, 1), label_visibility="collapsed")
    with col_e:
        end_date = st.date_input("To", value=datetime.today(), label_visibility="collapsed")

    st.markdown('<p style="font-size:0.78rem;font-weight:700;color:rgba(163,191,231,0.6);letter-spacing:0.08em;text-transform:uppercase;margin:0.8rem 0 0.4rem 0;">ML Model Settings</p>', unsafe_allow_html=True)
    n_estimators = st.slider("RF Trees", 50, 500, 200, 50)
    horizon = st.slider("Forecast Horizon (days)", 1, 10, 1)
    candlestick_period = st.slider("Candlestick Period (days)", 60, 504, 180)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Run Analysis")

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(99,179,237,0.1);margin:1.5rem 0 1rem 0;">
    <div style="font-size:0.7rem;color:rgba(163,191,231,0.35);text-align:center;">
      Data sourced from Yahoo Finance.<br>Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  Hero Header
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
  <div class="hero-badge">Live Market Analytics</div>
  <h1 class="hero-title">Stock Market Dashboard</h1>
  <div class="hero-subtitle">
    Technical analysis · Portfolio metrics · ML predictions &nbsp;|&nbsp;
    <span style="color:#90cdf4;font-weight:600;">{ticker}</span>
    &nbsp;<span class="chip chip-blue">{start_date.strftime('%b %Y')} → {end_date.strftime('%b %Y')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  Main Logic
# ─────────────────────────────────────────────────────────────────
if run_btn or "df_indicators" not in st.session_state or st.session_state.get("last_ticker") != ticker:

    if run_btn:
        # Clear cache for new runs
        load_and_process.clear()
        run_model.clear()

    with st.spinner(f"Fetching and processing data for **{ticker}**…"):
        raw_df, df = load_and_process(ticker, str(start_date))

    if raw_df is None or df is None:
        st.error(f"❌ Could not fetch data for **{ticker}**. Please verify the ticker symbol and try again.")
        st.stop()

    st.session_state["df_indicators"] = df
    st.session_state["last_ticker"] = ticker

    with st.spinner("Training Random Forest model…"):
        ml_metrics, ml_preds = run_model(ticker, str(start_date), n_estimators, horizon)
    st.session_state["ml_metrics"] = ml_metrics
    st.session_state["ml_preds"] = ml_preds

elif "df_indicators" in st.session_state:
    df = st.session_state["df_indicators"]
    ml_metrics = st.session_state.get("ml_metrics")
    ml_preds = st.session_state.get("ml_preds")
else:
    st.info("👈  Configure your settings in the sidebar and click **Run Analysis** to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
#  KPI Metric Cards
# ─────────────────────────────────────────────────────────────────
latest = df.iloc[-1]
prev = df.iloc[-2]
pm = PortfolioMetrics(df["daily_return"].dropna())

price_change = (latest["close"] - prev["close"]) / prev["close"]
vol_change   = (latest["volume"] - prev["volume"]) / (prev["volume"] + 1e-9)
rsi_val      = latest.get("rsi", float("nan"))
sharpe       = pm.sharpe_ratio()
ann_ret      = pm.annualized_return()
max_dd       = pm.max_drawdown()

def delta_html(val: float, is_pct: bool = True, invert: bool = False) -> str:
    color = "delta-pos" if (val > 0) != invert else ("delta-neg" if val != 0 else "delta-neu")
    arrow = "▲" if val > 0 else ("▼" if val < 0 else "—")
    txt = f"{val*100:+.2f}%" if is_pct else f"{val:+.2f}"
    return f'<div class="metric-delta {color}">{arrow} {txt}</div>'

st.markdown(f"""
<div class="metric-grid">

  <div class="metric-card" style="--accent-start:#63b3ed;--accent-end:#4299e1;">
    <div class="metric-label">Current Price</div>
    <div class="metric-value">${latest['close']:.2f}</div>
    {delta_html(price_change)}
  </div>

  <div class="metric-card" style="--accent-start:#9a75ea;--accent-end:#b794f4;">
    <div class="metric-label">Annualised Return</div>
    <div class="metric-value">{ann_ret*100:.1f}%</div>
    {delta_html(ann_ret, invert=False)}
  </div>

  <div class="metric-card" style="--accent-start:#48bb78;--accent-end:#68d391;">
    <div class="metric-label">Sharpe Ratio</div>
    <div class="metric-value">{sharpe:.2f}</div>
    <div class="metric-delta {'delta-pos' if sharpe > 1 else 'delta-neg'}">{'◎ Strong' if sharpe > 1 else '⚠ Low'}</div>
  </div>

  <div class="metric-card" style="--accent-start:#fc8181;--accent-end:#feb2b2;">
    <div class="metric-label">Max Drawdown</div>
    <div class="metric-value">{max_dd*100:.1f}%</div>
    <div class="metric-delta delta-neg">▼ Peak→Trough</div>
  </div>

  <div class="metric-card" style="--accent-start:#f6e05e;--accent-end:#fbd38d;">
    <div class="metric-label">RSI (14)</div>
    <div class="metric-value">{rsi_val:.1f}</div>
    <div style="margin-top:0.35rem;">{signal_chip(rsi_val)}</div>
  </div>

  <div class="metric-card" style="--accent-start:#63b3ed;--accent-end:#9a75ea;">
    <div class="metric-label">Volume</div>
    <div class="metric-value">{latest['volume']/1e6:.1f}M</div>
    {delta_html(vol_change)}
  </div>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  Price & Volume", "📉  Indicators", "🤖  ML Prediction", "📋  Portfolio Metrics"])

# ─── Tab 1: Candlestick + Volume ─────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Candlestick Chart</div>', unsafe_allow_html=True)

    df_cs = df.tail(candlestick_period)
    rows = 2
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # Candlestick
    colors_up   = "#26a69a"
    colors_down = "#ef5350"
    fig.add_trace(go.Candlestick(
        x=df_cs.index, open=df_cs["open"], high=df_cs["high"],
        low=df_cs["low"], close=df_cs["close"],
        increasing_line_color=colors_up, decreasing_line_color=colors_down,
        name="OHLC",
    ), row=1, col=1)

    # SMAs
    for sma, color, label in [("sma_20","#f1c40f","SMA 20"),("sma_50","#e67e22","SMA 50")]:
        if sma in df_cs.columns:
            fig.add_trace(go.Scatter(x=df_cs.index, y=df_cs[sma], name=label,
                                     line=dict(color=color, width=1.5, dash="dot")), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df_cs.columns:
        fig.add_trace(go.Scatter(x=df_cs.index, y=df_cs["bb_upper"], name="BB Upper",
                                  line=dict(color="rgba(100,149,237,0.5)", dash="dot", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_cs.index, y=df_cs["bb_lower"], name="BB Lower",
                                  line=dict(color="rgba(100,149,237,0.5)", dash="dot", width=1),
                                  fill="tonexty", fillcolor="rgba(100,149,237,0.05)"), row=1, col=1)

    # Volume bars
    if "volume" in df_cs.columns:
        vol_colors = [colors_up if c >= o else colors_down
                      for c, o in zip(df_cs["close"], df_cs["open"])]
        fig.add_trace(go.Bar(x=df_cs.index, y=df_cs["volume"]/1e6,
                              name="Volume (M)", marker_color=vol_colors, opacity=0.55), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=540,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.05)", title_text="Volume (M)"),
        xaxis2=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ─── Tab 2: Indicators ───────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    # RSI
    with c1:
        st.markdown('<div class="section-title">RSI (14)</div>', unsafe_allow_html=True)
        df_rsi = df.tail(252)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_rsi.index, y=df_rsi["rsi"], name="RSI",
                                      line=dict(color="#e74c3c", width=2)))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="rgba(252,129,129,0.6)",
                           annotation_text="Overbought", annotation_font_color="#fc8181")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="rgba(104,211,145,0.6)",
                           annotation_text="Oversold", annotation_font_color="#68d391")
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(252,129,129,0.05)", line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(72,187,120,0.05)", line_width=0)
        fig_rsi.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", height=280, showlegend=False,
                               margin=dict(l=5,r=5,t=10,b=5),
                               yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)"),
                               xaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_rsi, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # MACD
    with c2:
        st.markdown('<div class="section-title">MACD</div>', unsafe_allow_html=True)
        if "macd" in df.columns:
            df_macd = df.tail(252)
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df_macd.index, y=df_macd["macd"], name="MACD",
                                           line=dict(color="#3498db", width=2)))
            fig_macd.add_trace(go.Scatter(x=df_macd.index, y=df_macd["macd_signal"], name="Signal",
                                           line=dict(color="#e74c3c", width=1.5)))
            bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df_macd["macd_hist"]]
            fig_macd.add_trace(go.Bar(x=df_macd.index, y=df_macd["macd_hist"],
                                       name="Histogram", marker_color=bar_colors, opacity=0.55))
            fig_macd.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", height=280,
                                    legend=dict(orientation="h", y=1.05, font=dict(size=10)),
                                    margin=dict(l=5,r=5,t=10,b=5),
                                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_macd, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("MACD data unavailable.")

    # Bollinger Band Width + Stochastic
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Bollinger Band Width</div>', unsafe_allow_html=True)
        if "bb_width" in df.columns:
            df_bb = df.tail(252)
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df_bb.index, y=df_bb["bb_width"], name="BB Width",
                                         line=dict(color="#9a75ea", width=2), fill="tozeroy",
                                         fillcolor="rgba(154,117,234,0.08)"))
            fig_bb.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", height=260, showlegend=False,
                                  margin=dict(l=5,r=5,t=10,b=5),
                                  yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                                  xaxis=dict(gridcolor="rgba(255,255,255,0.04)"))
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_bb, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-title">Stochastic Oscillator (%K / %D)</div>', unsafe_allow_html=True)
        if "stoch_k" in df.columns:
            df_stoch = df.tail(252)
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=df_stoch.index, y=df_stoch["stoch_k"], name="%K",
                                            line=dict(color="#f6e05e", width=1.8)))
            fig_stoch.add_trace(go.Scatter(x=df_stoch.index, y=df_stoch["stoch_d"], name="%D",
                                            line=dict(color="#ed8936", width=1.5, dash="dot")))
            fig_stoch.add_hline(y=80, line_dash="dot", line_color="rgba(252,129,129,0.5)")
            fig_stoch.add_hline(y=20, line_dash="dot", line_color="rgba(104,211,145,0.5)")
            fig_stoch.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                     plot_bgcolor="rgba(0,0,0,0)", height=260,
                                     yaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)"),
                                     xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                                     legend=dict(orientation="h", y=1.08, font=dict(size=10)),
                                     margin=dict(l=5,r=5,t=10,b=5))
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_stoch, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)


# ─── Tab 3: ML Predictions ───────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Random Forest · Price Prediction</div>', unsafe_allow_html=True)

    if ml_metrics is not None and ml_preds is not None:
        # Prediction vs Actual chart
        predictor_temp = StockPredictor(df, target_col="close", horizon=horizon)
        y_true = predictor_temp.y_test
        n_pred = min(len(y_true), len(ml_preds))
        idx_test = df.index[-(n_pred):]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=idx_test, y=y_true[-n_pred:], name="Actual",
                                       line=dict(color="#63b3ed", width=2)))
        fig_pred.add_trace(go.Scatter(x=idx_test, y=ml_preds[-n_pred:], name=f"Predicted (H={horizon}d)",
                                       line=dict(color="#f6ad55", width=2, dash="dot")))
        fig_pred.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=380,
            legend=dict(orientation="h", y=1.04, font=dict(size=11)),
            margin=dict(l=10,r=10,t=10,b=10),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title_text="Price (USD)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # Metrics table
        st.markdown('<div class="section-title">Model Evaluation Metrics</div>', unsafe_allow_html=True)
        mdf = ml_metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        metric_styles = [
            ("MAE",      "Mean Absolute Error",    "#63b3ed", "#3182ce"),
            ("RMSE",     "Root Mean Sq. Error",    "#9a75ea", "#553c9a"),
            ("MAPE (%)", "Mean Abs. % Error",      "#f6ad55", "#c05621"),
            ("R²",       "Coefficient of Deter.",  "#68d391", "#276749"),
        ]
        for col, (key, label, c1, c2) in zip([col_m1,col_m2,col_m3,col_m4], metric_styles):
            val = mdf.get(key, float("nan"))
            with col:
                st.markdown(f"""
                <div class="metric-card" style="--accent-start:{c1};--accent-end:{c2};">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="font-size:1.35rem;">{val:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        # Next-day forecast
        st.markdown('<div class="section-title">Next-Day Forecast Signal</div>', unsafe_allow_html=True)
        last_pred = ml_preds[-1]
        curr_price = df["close"].iloc[-1]
        pred_return = (last_pred - curr_price) / curr_price
        direction = "📈 Bullish" if pred_return > 0 else "📉 Bearish"
        dir_color = "#68d391" if pred_return > 0 else "#fc8181"
        st.markdown(f"""
        <div class="metric-card" style="--accent-start:{dir_color};--accent-end:{dir_color};max-width:440px;">
          <div class="metric-label">Model Signal · {horizon}-Day Ahead Forecast</div>
          <div style="display:flex;align-items:baseline;gap:1rem;margin-top:0.3rem;">
            <div class="metric-value">${last_pred:.2f}</div>
            <div style="color:{dir_color};font-weight:700;font-size:1.1rem;">{direction}&nbsp;({pred_return*100:+.2f}%)</div>
          </div>
          <div class="metric-delta delta-neu" style="margin-top:0.5rem;">Current: ${curr_price:.2f} · Model: Random Forest ({n_estimators} trees)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("ML model could not be trained. Please run the analysis first.")


# ─── Tab 4: Portfolio Metrics ─────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Risk-Adjusted Performance</div>', unsafe_allow_html=True)

    sortino = pm.sortino_ratio()
    calmar  = pm.calmar_ratio()
    ann_vol = pm.annualized_volatility()

    cols_p = st.columns(3)
    port_metrics = [
        ("Sortino Ratio",          f"{sortino:.3f}",       "#9a75ea","#553c9a", "Risk-adj (downside only)"),
        ("Calmar Ratio",           f"{calmar:.3f}",        "#48bb78","#276749", "Return / Max Drawdown"),
        ("Annualised Volatility",  f"{ann_vol*100:.2f}%",  "#f6ad55","#c05621", "Std dev × √252"),
    ]
    for col, (lbl, val, c1, c2, note) in zip(cols_p, port_metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent-start:{c1};--accent-end:{c2};">
              <div class="metric-label">{lbl}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-delta delta-neu">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    # Return distribution
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Daily Return Distribution</div>', unsafe_allow_html=True)
    returns = df["daily_return"].dropna()
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=returns, nbinsx=80, name="Returns",
                                    marker_color="#3182ce", opacity=0.75,
                                    histnorm="probability density"))
    # KDE overlay
    from scipy.stats import gaussian_kde
    kde_x = np.linspace(returns.min(), returns.max(), 300)
    kde   = gaussian_kde(returns)(kde_x)
    fig_dist.add_trace(go.Scatter(x=kde_x, y=kde, name="KDE",
                                   line=dict(color="#9a75ea", width=2.5)))
    fig_dist.add_vline(x=float(returns.mean()), line_dash="dot", line_color="#f6e05e",
                        annotation_text="Mean", annotation_font_color="#f6e05e")
    fig_dist.add_vline(x=float(returns.quantile(0.05)), line_dash="dot", line_color="#fc8181",
                        annotation_text="VaR 95%", annotation_font_color="#fc8181")
    fig_dist.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=340,
        legend=dict(orientation="h", y=1.04, font=dict(size=11)),
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(title="Daily Return", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="Density", gridcolor="rgba(255,255,255,0.05)"),
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Drawdown chart
    st.markdown('<div class="section-title">Drawdown Over Time</div>', unsafe_allow_html=True)
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100,
                                 name="Drawdown (%)", fill="tozeroy",
                                 line=dict(color="#fc8181", width=1.5),
                                 fillcolor="rgba(252,129,129,0.12)"))
    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=280, showlegend=False,
        margin=dict(l=10,r=10,t=10,b=10),
        yaxis=dict(title="Drawdown (%)", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
