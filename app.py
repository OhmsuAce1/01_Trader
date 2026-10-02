import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any

from data_loader import (
    fetch_stock_data,
    fetch_ticker_info,
    fetch_screener_batch,
    fetch_sector_performance,
    fetch_market_overview,
    POPULAR_TICKERS,
    ALL_SCREENER_TICKERS,
    SECTOR_ETFS,
    MAJOR_INDICES
)
from indicators import generate_anti_stop_hunt_plan
from charts import (
    create_trading_chart,
    plot_equity_curve,
    create_sector_treemap,
    create_sector_barchart
)
import watchlist_manager
from ai_analyst import fetch_stock_news, analyze_stock_with_gemini, create_sentiment_gauge
from backtester import run_antistophunt_backtest


# Page Configuration
st.set_page_config(
    page_title="Webull US Stock Pro Trader & Anti-Stop Hunt",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Modern Trading UI
st.markdown("""
<style>
    /* Dark Terminal Theme Styles */
    .stApp {
        background-color: #0B0E14;
        color: #E6EDF3;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #151B26 0%, #1A2232 100%);
        border: 1px solid #243044;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700;
        color: #F8FAFC !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111622;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 18px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D4FF !important;
        color: #0B0E14 !important;
    }

    /* Plan card boxes */
    .plan-box {
        background: #151B26;
        border: 1px solid #222E42;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .plan-box-title {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .plan-box-val {
        font-size: 1.35rem;
        font-weight: 700;
    }

    /* Badge tags */
    .badge-green {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.3);
        display: inline-block;
        margin: 2px;
    }
    .badge-red {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(239, 68, 68, 0.3);
        display: inline-block;
        margin: 2px;
    }
    .badge-blue {
        background-color: rgba(0, 212, 255, 0.15);
        color: #38BDF8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(0, 212, 255, 0.3);
        display: inline-block;
        margin: 2px;
    }
    /* Technical Setup Badges */
    .badge-vol {
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        border: 1px solid rgba(245, 158, 11, 0.4);
        display: inline-block;
        margin: 2px;
    }
    .badge-breakout {
        background-color: rgba(236, 72, 153, 0.18);
        color: #F472B6;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        border: 1px solid rgba(236, 72, 153, 0.4);
        display: inline-block;
        margin: 2px;
    }
    .badge-momentum {
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        border: 1px solid rgba(16, 185, 129, 0.4);
        display: inline-block;
        margin: 2px;
    }
    .badge-hunt {
        background-color: rgba(0, 212, 255, 0.18);
        color: #00D4FF;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 700;
        border: 1px solid rgba(0, 212, 255, 0.4);
        display: inline-block;
        margin: 2px;
    }

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem 2rem 0.5rem !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            white-space: nowrap;
            display: flex;
            flex-wrap: nowrap;
            padding: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 12px !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
    }

    /* Screener Card Container for Mobile & Grid */
    .screener-card {
        background: linear-gradient(145deg, #131924 0%, #182232 100%);
        border: 1px solid #243248;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .screener-card:hover {
        border-color: #00D4FF;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "MSTR"
if "search_input" not in st.session_state:
    st.session_state.search_input = st.session_state.selected_ticker
if "chart_period" not in st.session_state:
    st.session_state.chart_period = "6mo"
if "chart_interval" not in st.session_state:
    st.session_state.chart_interval = "1d"

# Gemini API State Initialization
env_gemini_key = os.environ.get("GEMINI_API_KEY", "")
if not env_gemini_key:
    try:
        env_gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        env_gemini_key = ""

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = env_gemini_key
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = "gemini-flash-lite-latest"

if "ai_analysis_results" not in st.session_state:
    st.session_state.ai_analysis_results = {}

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_ai_analysis_cached(
    ticker: str,
    api_key: str,
    model_name: str,
    company_info_json: str,
    news_json: str
) -> Dict[str, Any]:
    comp_info = json.loads(company_info_json) if company_info_json else {}
    news_list = json.loads(news_json) if news_json else []
    res = analyze_stock_with_gemini(
        ticker=ticker,
        api_key=api_key,
        model_name=model_name,
        company_info=comp_info,
        news_list=news_list
    )
    if "error" in res:
        # Don't cache errors in Streamlit cache
        raise RuntimeError(res["error"])
    return res

def get_cached_ai_analysis(
    ticker: str,
    api_key: str,
    model_name: str,
    company_info_json: str,
    news_json: str
) -> Dict[str, Any]:
    try:
        return _fetch_ai_analysis_cached(ticker, api_key, model_name, company_info_json, news_json)
    except Exception as e:
        return {"error": str(e)}


def set_ticker(ticker: str):
    clean = ticker.upper().strip()
    st.session_state.selected_ticker = clean
    st.session_state.search_input = clean

def on_search_change():
    val = st.session_state.search_input.upper().strip()
    if val:
        st.session_state.selected_ticker = val


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ Webull US Stock Pro")
    st.caption("ระบบวิเคราะห์หุ้นสหรัฐฯ & Anti-Stop Hunt")
    st.markdown("---")

    # Search Bar
    st.markdown("#### 🔍 ค้นหา Ticker")
    st.text_input(
        "พิมพ์ Ticker (เช่น NVDA, MSTR):",
        key="search_input",
        on_change=on_search_change
    )

    # Quick Ticker Selectors
    st.markdown("#### ⚡ หุ้นยอดนิยม & หุ้นซิ่ง")
    for category, tickers in POPULAR_TICKERS.items():
        with st.expander(category, expanded=(category.startswith("Bitcoin"))):
            cols = st.columns(3)
            for i, tick in enumerate(tickers):
                col = cols[i % 3]
                col.button(
                    tick,
                    key=f"btn_quick_{tick}",
                    on_click=set_ticker,
                    args=(tick,),
                    use_container_width=True
                )


    st.markdown("---")
    # Chart Parameters
    st.markdown("#### ⚙️ ตั้งค่ากราฟ")
    st.session_state.chart_period = st.selectbox(
        "ช่วงเวลาย้อนหลัง:",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )
    st.session_state.chart_interval = st.selectbox(
        "Timeframe:",
        options=["1d", "1wk"],
        index=0
    )

    st.markdown("---")
    # Quick Watchlist Control
    current_wl = watchlist_manager.load_watchlist()
    is_in_wl = st.session_state.selected_ticker in current_wl

    if is_in_wl:
        if st.button(f"⭐ ลบ {st.session_state.selected_ticker} จาก Watchlist", use_container_width=True):
            watchlist_manager.remove_from_watchlist(st.session_state.selected_ticker)
            st.toast(f"ลบ {st.session_state.selected_ticker} ออกจาก Watchlist แล้ว")
            st.rerun()
    else:
        if st.button(f"⭐ เพิ่ม {st.session_state.selected_ticker} เข้า Watchlist", use_container_width=True):
            watchlist_manager.add_to_watchlist(st.session_state.selected_ticker)
            st.toast(f"บันทึก {st.session_state.selected_ticker} ลง Watchlist แล้ว")
            st.rerun()

    st.markdown("---")
    # Gemini AI API Settings
    st.markdown("#### 🤖 ตั้งค่า Gemini AI")
    gemini_key_input = st.text_input(
        "Gemini API Key:",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="ขอรับ API Key ฟรีได้จาก Google AI Studio (aistudio.google.com)"
    )
    if gemini_key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = gemini_key_input

    st.session_state.gemini_model = st.selectbox(
        "โมเดล Gemini:",
        options=[
            "gemini-flash-lite-latest",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-flash-latest",
            "gemini-3.8-flash"
        ],
        index=0
    )

    st.markdown("---")
    st.caption("พัฒนาสำหรับนักเทรดหุ้นสหรัฐฯ บน Webull | Data by Yahoo Finance & Gemini")

# ==========================================
# MAIN CONTENT
# ==========================================
st.title("📈 Webull US Stock Pro Trader & Anti-Stop Hunt Analyzer")
st.caption("ระบบวิเคราะห์หุ้นสหรัฐฯ รายวัน • ดักจับสัญญาณ Stop Hunt • วิเคราะห์ AI ข่าวสาร • สแกนหุ้น • Watchlist")

# Navigation Tabs
tab_market, tab1, tab_ai, tab2, tab3, tab4 = st.tabs([
    "🌐 ภาพรวมตลาด & Sectors",
    "📊 กราฟ & แผน Anti-Stop Hunt",
    "🤖 AI Deep Dive",
    "🔍 สแกนหุ้น (Screener)",
    "⭐ Watchlist ส่วนตัว",
    "📖 กลยุทธ์ & เคล็ดลับ"
])

# ----------------------------------------------------
# TAB 0: MARKET OVERVIEW & SECTOR HEATMAP
# ----------------------------------------------------
with tab_market:
    st.subheader("🌐 ภาพรวมตลาดสหรัฐฯ & Sector Heatmap")
    st.caption("ดัชนีหลัก, ความกว้างของตลาด (Market Breadth) และกระแสเงินหมุนเวียนกลุ่มอุตสาหกรรม")

    with st.spinner("กำลังอัปเดตข้อมูลดัชนีตลาดและ Sector ETFs..."):
        mkt_info = fetch_market_overview()
        sector_df = fetch_sector_performance()

    # Major Indices Strip
    indices_list = mkt_info.get("indices", [])
    if indices_list:
        idx_cols = st.columns(len(indices_list))
        for i, idx_data in enumerate(indices_list):
            with idx_cols[i]:
                c_1d = idx_data['change_1d']
                c_5d = idx_data['change_5d']
                st.metric(
                    label=f"{idx_data['symbol']} ({idx_data['name'].split(' (')[0]})",
                    value=f"${idx_data['price']:,.2f}",
                    delta=f"{c_1d:+.2f}% วันนี้ (5D: {c_5d:+.2f}%)",
                    delta_color="normal"
                )

    # Market Breadth Strip
    breadth = mkt_info.get("breadth", {})
    if breadth:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #131A26 0%, #1A2436 100%); border: 1px solid #23354E; border-radius: 8px; padding: 12px 16px; margin: 14px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-weight: 700; color: #94A3B8; font-size: 0.88rem;">📊 ความกว้างของตลาด (Market Breadth):</span>
                    <span style="font-weight: 800; font-size: 0.95rem; margin-left: 8px;">{breadth.get('sentiment', '')}</span>
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1;">
                    หุ้นเหนือ EMA 50: <b style="color: #00D4FF;">{breadth.get('pct_above_ema50', 0):.1f}%</b> ({breadth.get('above_ema50', 0)}/{breadth.get('total_stocks', 0)}) | 
                    หุ้นบวก: <b style="color: #4ADE80;">{breadth.get('advances', 0)}</b> | 
                    หุ้นลบ: <b style="color: #F87171;">{breadth.get('declines', 0)}</b> | 
                    A/D Ratio: <b style="color: #FDE047;">{breadth.get('ad_ratio', 1.0):.1f}x</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Sector Performance Section
    st.markdown("### 🗺️ ผลตอบแทนกลุ่มอุตสาหกรรม (S&P 500 Sector ETFs)")

    sec_c1, sec_c2, sec_c3 = st.columns([1, 1, 1])
    with sec_c1:
        timeframe_sel = st.selectbox(
            "ช่วงเวลาเปรียบเทียบผลตอบแทน:",
            options=["1 วัน (1D %)", "5 วัน (5D %)", "1 เดือน (1M %)"],
            index=0
        )
    with sec_c2:
        chart_type_sel = st.radio(
            "รูปแบบการแสดงผล:",
            options=["🗺️ Heatmap (Treemap)", "📊 กราฟแท่งจัดอันดับ (Bar Chart)"],
            index=0,
            horizontal=True
        )
    with sec_c3:
        st.write("")
        btn_reload_sec = st.button("🔄 รีเฟรชข้อมูล Sectors", use_container_width=True)
        if btn_reload_sec:
            fetch_sector_performance.clear()
            fetch_market_overview.clear()
            st.rerun()

    # Map selected timeframe column
    tf_col_map = {
        "1 วัน (1D %)": "1D %",
        "5 วัน (5D %)": "5D %",
        "1 เดือน (1M %)": "1M %"
    }
    target_col = tf_col_map[timeframe_sel]

    if sector_df is not None and not sector_df.empty:
        if chart_type_sel == "🗺️ Heatmap (Treemap)":
            fig_sec = create_sector_treemap(sector_df, timeframe_col=target_col)
            st.plotly_chart(fig_sec, use_container_width=True)
        else:
            fig_bar = create_sector_barchart(sector_df, timeframe_col=target_col)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Sector Detail Table & Quick Picks
        st.markdown("#### 📋 ตารางสรุป 11 กลุ่มอุตสาหกรรม:")
        disp_sec_df = sector_df[["Ticker", "Sector", "Price", "1D %", "5D %", "1M %"]].copy()

        def style_sec_change(val):
            color = '#4ADE80' if val > 0 else ('#F87171' if val < 0 else '#94A3B8')
            return f'color: {color}; font-weight: 700;'

        styled_sec = disp_sec_df.style.map(style_sec_change, subset=['1D %', '5D %', '1M %'])\
                                      .format({
                                          'Price': '${:,.2f}',
                                          '1D %': '{:+,.2f}%',
                                          '5D %': '{:+,.2f}%',
                                          '1M %': '{:+,.2f}%',
                                      })
        st.dataframe(styled_sec, use_container_width=True)

        st.markdown("#### ⚡ คลิกเพื่อดูกราฟ Sector ETF ทันที:")
        sec_pick_cols = st.columns(6)
        for i, row in enumerate(sector_df.itertuples()):
            col = sec_pick_cols[i % 6]
            col.button(
                f"📊 {row.Ticker}",
                key=f"sec_pick_{row.Ticker}",
                on_click=set_ticker,
                args=(row.Ticker,),
                use_container_width=True,
                type="primary" if row.Ticker == st.session_state.selected_ticker else "secondary"
            )
    else:
        st.warning("ไม่สามารถดึงข้อมูล Sector ETFs ได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง")

# ----------------------------------------------------
# TAB 1: CHART & TRADING PLAN
# ----------------------------------------------------
with tab1:
    ticker = st.session_state.selected_ticker

    # Fetch Data
    with st.spinner(f"กำลังดึงข้อมูลหุ้น {ticker}..."):
        df = fetch_stock_data(ticker, period=st.session_state.chart_period, interval=st.session_state.chart_interval)
        info = fetch_ticker_info(ticker)

    if df is None or df.empty:
        st.error(f"❌ ไม่พบข้อมูลราคาสำหรับ Ticker '{ticker}' โปรดตรวจสอบตัวสะกดสัญลักษณ์หุ้นสหรัฐฯ เช่น AAPL, NVDA, MSTR, RIOT")
    else:
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row

        curr_price = float(last_row['Close'])
        price_change = curr_price - float(prev_row['Close'])
        price_change_pct = (price_change / float(prev_row['Close'])) * 100

        # Company Header & Price Metrics
        col_hdr1, col_hdr2 = st.columns([3, 1])
        with col_hdr1:
            st.subheader(f"{ticker} - {info.get('name', ticker)}")
            st.caption(f"Sector: {info.get('sector', 'N/A')} | Market Cap: ${(info.get('market_cap', 0) / 1e9):,.2f}B")

        # Top Metric Cards Strip
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric(
                label="ราคาล่าสุด (Last)",
                value=f"${curr_price:,.2f}",
                delta=f"{price_change:+,.2f} ({price_change_pct:+.2f}%)",
                delta_color="normal"
            )
        with m2:
            st.metric(
                label="กรอบราคาวันนี้ (High - Low)",
                value=f"${float(last_row['High']):,.2f}",
                delta=f"Low: ${float(last_row['Low']):,.2f}",
                delta_color="off"
            )
        with m3:
            rvol = float(last_row['RVOL']) if not np.isnan(last_row['RVOL']) else 1.0
            rvol_status = "🔥 วอลุ่มสูงผิดปกติ" if rvol >= 1.2 else ("ปกติ" if rvol >= 0.8 else "วอลุ่มเบาบาง")
            st.metric(
                label="Relative Vol (RVOL)",
                value=f"{rvol:.2f}x",
                delta=rvol_status,
                delta_color="normal" if rvol >= 1.2 else "off"
            )
        with m4:
            rsi = float(last_row['RSI_14']) if not np.isnan(last_row['RSI_14']) else 50.0
            rsi_label = "🟢 Oversold (<35)" if rsi < 35 else ("🔴 Overbought (>70)" if rsi > 70 else "🟡 Neutral")
            st.metric(
                label="RSI (14 วัน)",
                value=f"{rsi:.1f}",
                delta=rsi_label,
                delta_color="off"
            )
        with m5:
            ema20 = float(last_row['EMA_20']) if not np.isnan(last_row['EMA_20']) else curr_price
            ema_diff = ((curr_price - ema20) / ema20) * 100
            st.metric(
                label="EMA 20 (Momentum)",
                value=f"${ema20:,.2f}",
                delta=f"{ema_diff:+.1f}% vs EMA20",
                delta_color="normal" if curr_price >= ema20 else "inverse"
            )
        with m6:
            ema200 = float(last_row['EMA_200']) if not np.isnan(last_row['EMA_200']) else curr_price
            trend = "Bullish (เหนือ EMA200)" if curr_price >= ema200 else "Bearish (ใต้ EMA200)"
            st.metric(
                label="แนวโน้มระยะยาว (EMA 200)",
                value=f"${ema200:,.2f}",
                delta=trend,
                delta_color="normal" if curr_price >= ema200 else "inverse"
            )

        st.markdown("---")

        # Anti-Stop Hunt Calculation
        plan = generate_anti_stop_hunt_plan(df)

        # Chart Option Checkboxes
        c_opt1, c_opt2, c_opt3, c_opt4, c_opt5, c_opt6 = st.columns(6)
        with c_opt1:
            show_stop_hunt = st.checkbox("🛡️ เส้นแผน Anti-Stop Hunt", value=True)
        with c_opt2:
            show_ema20 = st.checkbox("🟡 EMA 20 (เร็ว)", value=True)
        with c_opt3:
            show_ema50 = st.checkbox("🔵 EMA 50 (กลาง)", value=True)
        with c_opt4:
            show_ema200 = st.checkbox("🟣 EMA 200 (หลัก)", value=True)
        with c_opt5:
            show_rsi_panel = st.checkbox("📊 แสดง RSI", value=True)
        with c_opt6:
            show_macd_panel = st.checkbox("📶 แสดง MACD", value=True)

        # Plotly Chart
        fig = create_trading_chart(
            df=df,
            ticker=ticker,
            plan=plan if show_stop_hunt else None,
            show_ema20=show_ema20,
            show_ema50=show_ema50,
            show_ema200=show_ema200,
            show_stop_hunt_overlay=show_stop_hunt,
            show_rsi=show_rsi_panel,
            show_macd=show_macd_panel
        )
        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------
        # ANTI-STOP HUNT TRADING PLAN SECTION
        # ----------------------------------------------------
        st.markdown("### 🛡️ แผนเทรดป้องกัน Stop Hunt (Anti-Stop Hunt Strategy)")
        st.info("💡 **กลยุทธ์ Anti-Stop Hunt:** ดักทางสถาบันที่ชอบทุบหลุดแนวรับหลอกเพื่อกิน Stop Loss โดยเผื่อบัฟเฟอร์ 1.3x ATR เพื่อให้จุดหนีปลอดภัย ไม่โดนหลอกกิน Stop ฟรี")

        # Plan Summary Metric Cards
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        with p1:
            st.markdown(f"""
            <div class="plan-box" style="border-left: 4px solid #EAB308;">
                <div class="plan-box-title">🟡 จุดเข้า (Entry)</div>
                <div class="plan-box-val" style="color: #FDE047;">${plan['entry_price']}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">ราคาปัจจุบัน / ยืนยันจุดกลับตัว</div>
            </div>
            """, unsafe_allow_html=True)
        with p2:
            st.markdown(f"""
            <div class="plan-box" style="border-left: 4px solid #F97316;">
                <div class="plan-box-title">🟠 โซน Stop รายย่อย</div>
                <div class="plan-box-val" style="color: #FB923C;">${plan['retail_stop']}</div>
                <div style="font-size: 0.75rem; color: #F87171;">⚠️ จุดล่อบอทสถาบันกวาดกิน Stop</div>
            </div>
            """, unsafe_allow_html=True)
        with p3:
            st.markdown(f"""
            <div class="plan-box" style="border-left: 4px solid #EF4444;">
                <div class="plan-box-title">🛡️ Safe Stop Loss</div>
                <div class="plan-box-val" style="color: #F87171;">${plan['safe_stop_loss']}</div>
                <div style="font-size: 0.75rem; color: #FCA5A5;">บัฟเฟอร์ 1.3x ATR (-{plan['risk_pct']}%)</div>
            </div>
            """, unsafe_allow_html=True)
        with p4:
            st.markdown(f"""
            <div class="plan-box" style="border-left: 4px solid #22C55E;">
                <div class="plan-box-title">🎯 เป้าทำกำไร TP 1</div>
                <div class="plan-box-val" style="color: #4ADE80;">${plan['tp1_price']}</div>
                <div style="font-size: 0.75rem; color: #86EFAC;">+{plan['tp1_pct']}% (R:R {plan['rr_tp1']}R)</div>
            </div>
            """, unsafe_allow_html=True)
        with p5:
            st.markdown(f"""
            <div class="plan-box" style="border-left: 4px solid #10B981;">
                <div class="plan-box-title">🚀 เป้าทำกำไร TP 2</div>
                <div class="plan-box-val" style="color: #34D399;">${plan['tp2_price']}</div>
                <div style="font-size: 0.75rem; color: #6EE7B7;">+{plan['tp2_pct']}% (R:R {plan['rr_tp2']}R)</div>
            </div>
            """, unsafe_allow_html=True)
        with p6:
            st.markdown(f"""
            <div class="plan-box" style="border-left: 4px solid #38BDF8;">
                <div class="plan-box-title">🧮 ความผันผวน ATR</div>
                <div class="plan-box-val" style="color: #38BDF8;">${plan['atr_14']}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Swing Low: ${plan['swing_low']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Interactive Webull Position Size Calculator
        st.markdown("#### 🧮 คำนวณจำนวนหุ้น & ขนาด Position (Webull Position Calculator)")
        calc_col1, calc_col2 = st.columns(2)

        with calc_col1:
            account_size = st.number_input("เงินพอร์ตลงทุน ($ USD):", min_value=100.0, value=5000.0, step=500.0)
            risk_pct_trade = st.slider("ความเสี่ยงสูงสุดต่อไม้ (% Risk per Trade):", min_value=0.5, max_value=5.0, value=1.5, step=0.5)

        risk_budget = account_size * (risk_pct_trade / 100.0)
        risk_per_share = plan['risk_per_share']
        suggested_shares = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
        total_investment = suggested_shares * plan['entry_price']
        profit_tp1 = suggested_shares * (plan['tp1_price'] - plan['entry_price'])
        profit_tp2 = suggested_shares * (plan['tp2_price'] - plan['entry_price'])

        with calc_col2:
            st.markdown(f"""
            <div style="background-color: #111622; padding: 14px; border-radius: 8px; border: 1px solid #1E293B;">
                <div style="font-weight: 600; color: #00D4FF; margin-bottom: 8px;">📋 สรุปคำสั่งเทรดบน Webull:</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #94A3B8;">จำนวนหุ้นที่ควรซื้อ (Position Size):</span>
                    <span style="font-weight: 700; color: #F8FAFC;">{suggested_shares:,} หุ้น</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #94A3B8;">มูลค่าเงินลงทุนทั้งหมด:</span>
                    <span style="font-weight: 600; color: #F8FAFC;">${total_investment:,.2f} ({((total_investment/account_size)*100):.1f}% ของพอร์ต)</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #F87171;">ขาดทุนสูงสุดกรณีโดน Stop Loss:</span>
                    <span style="font-weight: 700; color: #F87171;">-${risk_budget:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #4ADE80;">กำไรที่คาดหวัง TP 1 (+1.5R):</span>
                    <span style="font-weight: 700; color: #4ADE80;">+${profit_tp1:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #34D399;">กำไรที่คาดหวัง TP 2:</span>
                    <span style="font-weight: 700; color: #34D399;">+${profit_tp2:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption("💡 **คำแนะนำ Webull:** แนะนำส่งคำสั่งแบบ 'OTO (Bracket Order)' และตั้ง Stop Loss ที่ Safe SL ป้องกันถูกกวาดกิน Stop ฟรี")

        # ----------------------------------------------------
        # ANTI-STOP HUNT BACKTESTING SECTION
        # ----------------------------------------------------
        st.markdown("---")
        st.markdown("### 🧪 ผลการทดสอบกลยุทธ์ย้อนหลัง (Anti-Stop Hunt Backtester)")
        st.caption(f"จำลองการเทรดจริงย้อนหลังตามสัญญาณ Sweep Low Reclaim & Support Wick Rejection บนหุ้น {ticker}")

        bt_c1, bt_c2, bt_c3 = st.columns([1, 1, 1])
        with bt_c1:
            bt_rr = st.slider("เป้าหมายกำไรต่อความเสี่ยง (Target Risk/Reward):", min_value=1.0, max_value=4.0, value=2.0, step=0.5, format="%.1fR")
        with bt_c2:
            bt_period = st.selectbox("ระยะเวลาทดสอบย้อนหลัง (Lookback Period):", options=["6 เดือน (6 Months)", "1 ปี (1 Year)", "2 ปี (2 Years)"], index=1)
        with bt_c3:
            bt_risk_pct = st.slider("ความเสี่ยงต่อไม้ (% Risk per Trade):", min_value=1.0, max_value=5.0, value=2.0, step=0.5, format="%.1f%%")

        period_map = {
            "6 เดือน (6 Months)": ("6mo", 130),
            "1 ปี (1 Year)": ("1y", 252),
            "2 ปี (2 Years)": ("2y", 504),
        }
        chosen_period, chosen_bars = period_map[bt_period]

        with st.spinner(f"กำลังรัน Backtest กลยุทธ์ Anti-Stop Hunt บนหุ้น {ticker}..."):
            bt_df = fetch_stock_data(ticker, period=chosen_period, interval="1d")
            if bt_df is None or len(bt_df) < 30:
                bt_df = df
            bt_results = run_antistophunt_backtest(
                bt_df,
                risk_reward_ratio=bt_rr,
                lookback_bars=chosen_bars,
                risk_per_trade_pct=bt_risk_pct,
                initial_capital=account_size
            )

        if bt_results.get("total_trades", 0) == 0:
            st.info(f"💡 ไม่พบการเทรดที่ตรงเงื่อนไขของ {ticker} ในช่วง {bt_period} ที่เลือก หรือข้อมูลไม่เพียงพอ")
        else:
            # Metrics Row
            bm1, bm2, bm3, bm4, bm5, bm6 = st.columns(6)
            with bm1:
                wr = bt_results['win_rate']
                wr_color = "normal" if wr >= 50 else ("off" if wr >= 40 else "inverse")
                st.metric("Win Rate (%)", f"{wr:.1f}%", f"{bt_results['wins']} ชนะ / {bt_results['losses']} แพ้", delta_color=wr_color)
            with bm2:
                pf = bt_results['profit_factor']
                st.metric("Profit Factor", f"{pf:.2f}", "ยอดเยี่ยม (>1.5)" if pf >= 1.5 else ("ทำกำไรได้ (>1.0)" if pf >= 1.0 else "ขาดทุน (<1.0)"), delta_color="normal" if pf >= 1.0 else "inverse")
            with bm3:
                net_r = bt_results['total_pnl_r']
                net_ret_pct = bt_results['total_return_pct']
                st.metric("กำไรสุทธิสะสม (Net Return)", f"{net_r:+.1f}R", f"{net_ret_pct:+.2f}% (${bt_results['ending_capital'] - bt_results['initial_capital']:+,.2f})", delta_color="normal" if net_r >= 0 else "inverse")
            with bm4:
                st.metric("จำนวนไม้ทั้งหมด (Total Trades)", f"{bt_results['total_trades']} ไม้", f"Avg: {bt_results['avg_trade_r']:+.2f}R / ไม้", delta_color="off")
            with bm5:
                st.metric("Max Drawdown (ย่อลึกสุด)", f"-{bt_results['max_drawdown_pct']:.1f}%", "ความเสี่ยงของพอร์ต", delta_color="inverse")
            with bm6:
                st.metric("ระยะเวลาถือครองเฉลี่ย", f"{bt_results['avg_holding_days']:.1f} วัน", "สไตล์ Swing Trade", delta_color="off")

            # Equity Curve Chart
            st.markdown("#### 📈 กราฟผลกำไรสะสมจำลอง (Equity Curve)")
            eq_fig = plot_equity_curve(bt_results['equity_df'], ticker=ticker, initial_capital=bt_results['initial_capital'])
            st.plotly_chart(eq_fig, use_container_width=True)

            # Trade Log Expander
            trades_list = bt_results['trades']
            with st.expander(f"📋 ประวัติการเปิด-ปิดออเดอร์ทั้งหมด ({len(trades_list)} ไม้)", expanded=False):
                trades_table = pd.DataFrame(trades_list)
                if not trades_table.empty:
                    rename_cols = {
                        "entry_date": "วันที่เข้า (Entry)",
                        "exit_date": "วันที่ออก (Exit)",
                        "setup_type": "ประเภทสัญญาณ",
                        "entry_price": "ราคาเข้า ($)",
                        "stop_loss": "Safe SL ($)",
                        "take_profit": "เป้า TP ($)",
                        "exit_price": "ราคาปิด ($)",
                        "result": "ผลลัพธ์",
                        "pnl_r": "PnL (R)",
                        "pnl_pct": "กำไร (%)",
                        "dollar_pnl": "กำไร ($)",
                        "bars_held": "ถือครอง (วัน)"
                    }
                    trades_table = trades_table.rename(columns=rename_cols)

                    def style_trade_res(val):
                        if val == "WIN":
                            return "background-color: rgba(34, 197, 94, 0.2); color: #4ADE80; font-weight: 700;"
                        elif val == "LOSS":
                            return "background-color: rgba(239, 68, 68, 0.2); color: #F87171; font-weight: 700;"
                        return ""

                    def style_trade_pnl(val):
                        color = "#4ADE80" if val > 0 else ("#F87171" if val < 0 else "#94A3B8")
                        return f"color: {color}; font-weight: 700;"

                    styled_trades = trades_table.style.map(style_trade_res, subset=["ผลลัพธ์"])\
                                                      .map(style_trade_pnl, subset=["PnL (R)", "กำไร (%)", "กำไร ($)"])\
                                                      .format({
                                                          "ราคาเข้า ($)": "${:,.2f}",
                                                          "Safe SL ($)": "${:,.2f}",
                                                          "เป้า TP ($)": "${:,.2f}",
                                                          "ราคาปิด ($)": "${:,.2f}",
                                                          "PnL (R)": "{:+.2f}R",
                                                          "กำไร (%)": "{:+.2f}%",
                                                          "กำไร ($)": "${:+,.2f}",
                                                      })
                    st.dataframe(styled_trades, use_container_width=True)


# ----------------------------------------------------
# TAB AI: AI DEEP DIVE & NEWS SENTIMENT
# ----------------------------------------------------
with tab_ai:
    ai_ticker = st.session_state.selected_ticker
    st.subheader(f"🤖 AI Stock Analyst & News Sentiment: {ai_ticker}")
    st.caption("ขับเคลื่อนด้วย Google Gemini API • วิเคราะห์ Sentiment ข่าวสารล่าสุด, ปัจจัยขับเคลื่อนราคา (Catalysts) และประเมินความเสี่ยง")

    # API Key check
    api_key_to_use = st.session_state.gemini_api_key.strip()

    if not api_key_to_use:
        st.info("""
        🔑 **กรุณาระบุ Gemini API Key เพื่อเปิดใช้งานระบบ AI Stock Analyst**
        
        คุณสามารถขอ API Key ได้ฟรี (มีโควตาฟรีใช้งานได้สูงมาก) จาก Google AI Studio:
        1. ไปที่เว็บไซต์ 👉 [Google AI Studio (Get API Key)](https://aistudio.google.com/app/apikey)
        2. กดปุ่ม **"Create API key"** แล้วคัดลอก Key มาวางในช่องด้านล่างนี้ หรือในแถบด้านข้าง (Sidebar)
        """)
        quick_key = st.text_input(
            "วาง Gemini API Key ที่นี่เพื่อเริ่มใช้งาน:",
            type="password",
            placeholder="AIzaSy...",
            key="quick_gemini_key_input"
        )
        if quick_key:
            st.session_state.gemini_api_key = quick_key.strip()
            st.rerun()
    else:
        # Fetch company info & latest news for the selected ticker
        with st.spinner(f"กำลังดึงพาดหัวข่าวล่าสุดของ {ai_ticker} จาก Yahoo Finance..."):
            stock_news = fetch_stock_news(ai_ticker, limit=8)
            comp_info = fetch_ticker_info(ai_ticker)

        col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
        with col_act1:
            st.markdown(f"**พาดหัวข่าวที่พบ:** `{len(stock_news)} รายการ` | **โมเดล:** `{st.session_state.gemini_model}`")
        with col_act2:
            btn_analyze = st.button("🧠 สั่ง Gemini วิเคราะห์ทันที", type="primary", use_container_width=True)
        with col_act3:
            btn_clear_cache = st.button("🔄 ล้างแคช & วิเคราะห์ใหม่", use_container_width=True)

        if btn_clear_cache:
            _fetch_ai_analysis_cached.clear()
            st.session_state.ai_analysis_results.clear()
            st.toast("ล้างแคชข้อมูลการวิเคราะห์ทั้งหมดแล้ว")
            st.rerun()

        if btn_analyze:
            _fetch_ai_analysis_cached.clear()
            st.session_state.ai_analysis_results.pop(ai_ticker, None)
            with st.spinner(f"กำลังส่งข้อมูลให้ Gemini ({st.session_state.gemini_model}) วิเคราะห์ Sentiment และปัจจัยขับเคลื่อน..."):
                comp_info_json = json.dumps(comp_info)
                news_json = json.dumps(stock_news)
                analysis_res = get_cached_ai_analysis(
                    ticker=ai_ticker,
                    api_key=api_key_to_use,
                    model_name=st.session_state.gemini_model,
                    company_info_json=comp_info_json,
                    news_json=news_json
                )
                if "error" not in analysis_res:
                    st.session_state.ai_analysis_results[ai_ticker] = analysis_res
                else:
                    st.error(f"❌ {analysis_res['error']}")

        result = st.session_state.ai_analysis_results.get(ai_ticker, {})

        if not result and not btn_analyze:
            st.markdown(f"""
            <div style="background-color: #111622; border: 1px dashed #334155; border-radius: 8px; padding: 28px; text-align: center; margin-top: 16px;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #00D4FF; margin-bottom: 8px;">
                    พร้อมวิเคราะห์ข้อมูลและ Sentiment ของหุ้น {ai_ticker}
                </div>
                <div style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 18px;">
                    ตรวจพบพาดหัวข่าวล่าสุด {len(stock_news)} รายการจาก Yahoo Finance • คลิกปุ่ม <b>"🧠 สั่ง Gemini วิเคราะห์ทันที"</b> ด้านบน เพื่อเริ่มประมวลผล
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif result and "error" not in result:

            score = int(result.get("sentiment_score", 0))
            label = result.get("sentiment_label", "Neutral")
            reasoning = result.get("sentiment_reasoning", "")
            catalysts = result.get("key_catalysts", [])
            risks = result.get("risks", [])
            takeaway = result.get("actionable_takeaway", "")
            model_used = result.get("model_used", st.session_state.gemini_model)

            # ROW 1: Sentiment Gauge & Score Card
            st.markdown("---")
            g_col1, g_col2 = st.columns([1, 1])

            with g_col1:
                gauge_fig = create_sentiment_gauge(score, label)
                st.plotly_chart(gauge_fig, use_container_width=True)

            with g_col2:
                badge_cls = "badge-green" if score >= 35 else ("badge-red" if score <= -35 else "badge-blue")
                st.markdown(f"""
                <div class="plan-box" style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span class="plan-box-title">คะแนนความเชื่อมั่นรวม (News Sentiment)</span>
                        <span class="{badge_cls}">{label}</span>
                    </div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin-bottom: 8px;">
                        {score:+d} <span style="font-size: 1rem; color: #94A3B8; font-weight: normal;">/ 100 คะแนน</span>
                    </div>
                    <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.5;">
                        {reasoning}
                    </div>
                    <div style="margin-top: 12px; font-size: 0.75rem; color: #64748B;">
                        ⚡ โมเดลที่ประมวลผล: {model_used} | บันทึกแคชประหยัด Token 30 นาที
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ROW 2: Catalysts & Risks
            st.markdown("---")
            c_col1, c_col2 = st.columns(2)

            with c_col1:
                st.markdown("#### 🚀 ปัจจัยขับเคลื่อนราคาสำคัญ (Key Catalysts & Drivers)")
                if catalysts:
                    for i, cat in enumerate(catalysts, 1):
                        st.markdown(f"""
                        <div style="background-color: #111622; border-left: 3px solid #00D4FF; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                            <div style="color: #E2E8F0; font-size: 0.9rem; font-weight: 500;">
                                {cat}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ไม่มีข้อมูลปัจจัยขับเคลื่อนเฉพาะ")

            with c_col2:
                st.markdown("#### ⚠️ ความเสี่ยงและจุดที่ควรระวัง (Risks & Headwinds)")
                if risks:
                    for i, rk in enumerate(risks, 1):
                        st.markdown(f"""
                        <div style="background-color: #111622; border-left: 3px solid #EF4444; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                            <div style="color: #FCA5A5; font-size: 0.9rem; font-weight: 500;">
                                {rk}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("ไม่มีข้อมูลความเสี่ยงเฉพาะ")

            # ROW 3: Actionable Webull Takeaway
            if takeaway:
                st.markdown("---")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #151B26 0%, #1E293B 100%); border: 1px solid #38BDF8; border-radius: 8px; padding: 16px;">
                    <div style="font-weight: 700; color: #38BDF8; font-size: 1rem; margin-bottom: 6px;">
                        💡 สรุปมุมมองเชิงกลยุทธ์สำหรับนักเทรด Webull (Actionable Strategy)
                    </div>
                    <div style="color: #F1F5F9; font-size: 0.95rem; line-height: 1.6;">
                        {takeaway}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ROW 4: News list accordion
            st.markdown("---")
            with st.expander(f"📰 รายการพาดหัวข่าวล่าสุดที่นำมาวิเคราะห์ ({len(stock_news)} ข่าว)", expanded=False):
                if stock_news:
                    for item in stock_news:
                        time_str = f" • {item['publish_time']}" if item.get('publish_time') else ""
                        link_html = f"<a href='{item['link']}' target='_blank' style='color: #00D4FF; text-decoration: none;'>อ่านข่าวต้นฉบับ ↗</a>" if item.get('link') else ""
                        st.markdown(f"""
                        - **{item.get('title', '')}**  
                          <span style='color: #94A3B8; font-size: 0.8rem;'>สำนักข่าว: {item.get('publisher', 'Unknown')}{time_str}</span> | {link_html}
                        """, unsafe_allow_html=True)
                else:
                    st.write("ไม่พบรายการข่าวเพิ่มเติมในขณะนี้")

# ----------------------------------------------------
# TAB 2: DAILY STOCK SCREENER & TECHNICAL SETUPS
# ----------------------------------------------------
with tab2:
    st.subheader("🔍 ระบบสแกนหุ้นประจำวัน (Daily Stock Screener & Technical Setups)")
    st.caption("คัดกรองหุ้นสหรัฐฯ ยอดนิยม 48 ตัวบน Webull ด้วยระบบ Multi-Threading และตรวจจับ 4 Technical Setups ยอดนิยมอัตโนมัติ")

    # Screener Category & Filter Selector
    sc_col1, sc_col2 = st.columns([1, 2])
    with sc_col1:
        screener_group = st.selectbox(
            "เลือกกลุ่มหุ้นที่ต้องการสแกน:",
            options=["ทั้งหมด (หุ้นซิ่ง & ยอดฮิต Webull)"] + list(POPULAR_TICKERS.keys()),
            index=0
        )

    # Determine tickers to scan
    if screener_group == "ทั้งหมด (หุ้นซิ่ง & ยอดฮิต Webull)":
        target_tickers = ALL_SCREENER_TICKERS
    else:
        target_tickers = POPULAR_TICKERS[screener_group]

    with sc_col2:
        filter_criteria = st.multiselect(
            "เงื่อนไขการกรองสัญญาณเทคนิค (Technical Setups):",
            options=[
                "🚀 Volume Surge (Volume วันนี้ > 200% ของค่าเฉลี่ย 20 วัน)",
                "💥 Breakout 20D / 52W High (ราคาทะลุแนวต้าน 20 วัน หรือทำ New High)",
                "📈 Bullish Momentum (RSI 50-70 และราคาอยู่เหนือ EMA 20 & EMA 50)",
                "🪤 Anti-Stop Hunt Trigger (สัญญาณ Sweep Low Reclaim หรือ Wick Reject)",
                "🎯 RSI < 35 (โซน Oversold หาจังหวะกลับตัว Dip-Buy)"
            ],
            default=[]
        )

    # View Mode & Action Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        view_mode = st.radio(
            "โหมดการแสดงผล (View Mode):",
            options=["📱 มุมมองการ์ด (Mobile Friendly)", "🖥️ มุมมองตาราง (Desktop Table)"],
            index=0,
            horizontal=True
        )
    with ctrl_col2:
        st.write("")
        btn_refresh_screener = st.button("🔄 ล้างแคช & สแกนใหม่", use_container_width=True)
    with ctrl_col3:
        st.write("")
        btn_scan = st.button("🚀 สแกนข้อมูลทันที", use_container_width=True, type="primary")

    if btn_refresh_screener:
        fetch_screener_batch.clear()
        st.toast("ล้างแคชข้อมูลสแกนหุ้นเรียบร้อย กำลังดึงข้อมูลล่าสุด...")
        st.rerun()

    # Run Screener with High-Performance Multi-Threading
    with st.spinner(f"กำลังสแกนและวิเคราะห์สัญญาณเทคนิค {len(target_tickers)} ตัวแบบ Multi-Threading..."):
        screener_df = fetch_screener_batch(target_tickers)

    if screener_df is None or screener_df.empty:
        st.warning("ไม่สามารถดึงข้อมูลสแกนหุ้นได้ในขณะนี้ กรุณากดปุ่ม 'ล้างแคช & สแกนใหม่' เพื่อลองอีกครั้ง")
    else:
        filtered_df = screener_df.copy()

        # Apply Filters
        if "🚀 Volume Surge (Volume วันนี้ > 200% ของค่าเฉลี่ย 20 วัน)" in filter_criteria:
            filtered_df = filtered_df[filtered_df["Vol Surge"] == True]
        if "💥 Breakout 20D / 52W High (ราคาทะลุแนวต้าน 20 วัน หรือทำ New High)" in filter_criteria:
            filtered_df = filtered_df[filtered_df["Breakout"] == True]
        if "📈 Bullish Momentum (RSI 50-70 และราคาอยู่เหนือ EMA 20 & EMA 50)" in filter_criteria:
            filtered_df = filtered_df[filtered_df["Bullish Momentum"] == True]
        if "🪤 Anti-Stop Hunt Trigger (สัญญาณ Sweep Low Reclaim หรือ Wick Reject)" in filter_criteria:
            filtered_df = filtered_df[filtered_df["Anti-Stop Hunt Trigger"] == True]
        if "🎯 RSI < 35 (โซน Oversold หาจังหวะกลับตัว Dip-Buy)" in filter_criteria:
            filtered_df = filtered_df[filtered_df["RSI (14)"] < 35]

        # Screener Metrics Strip
        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        with sm1:
            st.metric("หุ้นที่ผ่านเงื่อนไข", f"{len(filtered_df)} / {len(screener_df)} ตัว")
        with sm2:
            vol_count = len(screener_df[screener_df["Vol Surge"] == True])
            st.metric("🚀 Vol Surge (>200%)", f"{vol_count} ตัว")
        with sm3:
            bo_count = len(screener_df[screener_df["Breakout"] == True])
            st.metric("💥 Breakout High", f"{bo_count} ตัว")
        with sm4:
            bull_count = len(screener_df[screener_df["Bullish Momentum"] == True])
            st.metric("📈 Bull Momentum", f"{bull_count} ตัว")
        with sm5:
            hunt_count = len(screener_df[screener_df["Anti-Stop Hunt Trigger"] == True])
            st.metric("🪤 Anti-Stop Hunt", f"{hunt_count} ตัว")

        st.markdown("---")

        # Instant Anti-Stop Hunt Plan Box for currently selected stock
        active_t = st.session_state.selected_ticker
        active_match = screener_df[screener_df["Ticker"] == active_t]
        if not active_match.empty:
            arow = active_match.iloc[0]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #00D4FF; border-radius: 10px; padding: 16px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 8px;">
                    <div>
                        <span style="font-size: 1.25rem; font-weight: 800; color: #00D4FF;">🎯 แผนเทรดด่วน Anti-Stop Hunt: {active_t}</span>
                        <span style="margin-left: 10px; font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">${arow['Price']:,.2f}</span>
                        <span style="margin-left: 6px; font-weight: 600; color: {'#4ADE80' if arow['Change %'] >= 0 else '#F87171'};">({arow['Change %']:+.2f}%)</span>
                    </div>
                    <div style="color: #94A3B8; font-size: 0.85rem;">
                        RVOL (20D): <b style="color: #F8FAFC;">{arow['RVOL (20D)']:.2f}x</b> | RSI: <b style="color: #F8FAFC;">{arow['RSI (14)']:.1f}</b>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 10px;">
                    <div style="background: #111827; padding: 8px 12px; border-radius: 6px; border: 1px solid #374151;">
                        <div style="font-size: 0.72rem; color: #9CA3AF;">จุดเข้าซื้อ (Entry)</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8;">${arow['Entry Price']:,.2f}</div>
                    </div>
                    <div style="background: #111827; padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(239, 68, 68, 0.4);">
                        <div style="font-size: 0.72rem; color: #F87171;">Safe Stop Loss (ดักล่า)</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #F87171;">${arow['Safe Stop Loss']:,.2f} (-{arow['Risk %']:.1f}%)</div>
                    </div>
                    <div style="background: #111827; padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(34, 197, 94, 0.4);">
                        <div style="font-size: 0.72rem; color: #4ADE80;">Target TP 1 (+1.5R)</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #4ADE80;">${arow['TP1']:,.2f}</div>
                    </div>
                    <div style="background: #111827; padding: 8px 12px; border-radius: 6px; border: 1px solid rgba(52, 211, 153, 0.4);">
                        <div style="font-size: 0.72rem; color: #34D399;">Target TP 2 (+2.5R)</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #34D399;">${arow['TP2']:,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if filtered_df.empty:
            st.info("💡 ไม่พบหุ้นที่ตรงกับเงื่อนไขทั้งหมดที่คุณเลือกพร้อมกัน ลองลดเงื่อนไขการกรองลง")
        else:
            # ----------------------------------------------------
            # VIEW OPTION 1: MOBILE & TOUCH-FRIENDLY CARDS
            # ----------------------------------------------------
            if view_mode == "📱 มุมมองการ์ด (Mobile Friendly)":
                st.caption(f"แสดงผลการ์ดแบบ Responsive สำหรับหน้าจอมือถือและแท็บเล็ต ({len(filtered_df)} รายการ)")
                
                # Render 2 columns on desktop, 1 column automatically on mobile
                card_cols = st.columns(2)
                for idx, (_, r) in enumerate(filtered_df.iterrows()):
                    ticker_name = r["Ticker"]
                    price = float(r["Price"])
                    chg = float(r["Change %"])
                    rvol_val = float(r.get("RVOL (20D)", 1.0))
                    rsi_val = float(r.get("RSI (14)", 50.0))
                    safe_sl = float(r.get("Safe Stop Loss", price * 0.95))
                    tp1_val = float(r.get("TP1", price * 1.05))

                    target_col = card_cols[idx % 2]
                    with target_col:
                        badges_html = ""
                        if r.get("Vol Surge"):
                            badges_html += f'<span class="badge-vol">🚀 Vol Surge ({rvol_val:.1f}x)</span>'
                        if r.get("Breakout"):
                            b_txt = "🌟 52W High" if r.get("52W High") else "💥 Breakout 20D"
                            badges_html += f'<span class="badge-breakout">{b_txt}</span>'
                        if r.get("Bullish Momentum"):
                            badges_html += f'<span class="badge-momentum">📈 Bull Momentum (RSI {rsi_val:.0f})</span>'
                        if r.get("Anti-Stop Hunt Trigger"):
                            badges_html += '<span class="badge-hunt">🪤 Anti-Stop Hunt</span>'

                        if not badges_html:
                            badges_html = '<span style="color: #64748B; font-size: 0.75rem;">ปกติ (ไม่มี Setup พิเศษ)</span>'

                        price_color = '#4ADE80' if chg >= 0 else '#F87171'

                        st.markdown(f"""
                        <div class="screener-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                                <div>
                                    <span style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC;">{ticker_name}</span>
                                    <span style="margin-left: 8px; font-size: 0.85rem; color: #94A3B8;">Webull Pro</span>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC;">${price:,.2f}</div>
                                    <div style="font-size: 0.88rem; font-weight: 700; color: {price_color};">{chg:+,.2f}%</div>
                                </div>
                            </div>
                            <div style="margin-bottom: 10px; min-height: 28px;">
                                {badges_html}
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; background: rgba(15, 23, 42, 0.6); padding: 8px 10px; border-radius: 6px; font-size: 0.82rem; margin-bottom: 12px; border: 1px solid #1E293B; text-align: center;">
                                <div><span style="color:#94A3B8;">RVOL:</span> <b style="color: #38BDF8;">{rvol_val:.1f}x</b></div>
                                <div><span style="color:#94A3B8;">RSI:</span> <b style="color: #F8FAFC;">{rsi_val:.0f}</b></div>
                                <div><span style="color:#94A3B8;">Safe SL:</span> <b style="color: #F87171;">${safe_sl:,.2f}</b></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        btn_label = f"⚡ Click to Analyze {ticker_name}"
                        st.button(
                            btn_label,
                            key=f"card_btn_{ticker_name}_{idx}",
                            on_click=set_ticker,
                            args=(ticker_name,),
                            use_container_width=True,
                            type="primary" if ticker_name == st.session_state.selected_ticker else "secondary"
                        )
                        st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

            # ----------------------------------------------------
            # VIEW OPTION 2: DESKTOP SUMMARY TABLE
            # ----------------------------------------------------
            else:
                st.caption(f"แสดงผลตารางสรุปข้อมูลแบบละเอียด ({len(filtered_df)} รายการ)")
                disp_cols = [
                    "Ticker", "Price", "Change %", "RVOL (20D)", "RSI (14)",
                    "Setups Badges", "Safe Stop Loss", "TP1", "R:R TP1", "EMA 20", "EMA 50"
                ]
                display_table = filtered_df[disp_cols].copy()

                def highlight_change(val):
                    color = '#4ADE80' if val > 0 else ('#F87171' if val < 0 else '#94A3B8')
                    return f'color: {color}; font-weight: 700;'

                def highlight_rsi(val):
                    if val < 35:
                        return 'background-color: rgba(34, 197, 94, 0.2); color: #4ADE80; font-weight: 700;'
                    elif val > 70:
                        return 'background-color: rgba(239, 68, 68, 0.2); color: #F87171; font-weight: 700;'
                    return ''

                def highlight_rvol(val):
                    if val >= 2.0:
                        return 'background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; font-weight: 700;'
                    elif val >= 1.2:
                        return 'color: #38BDF8; font-weight: 600;'
                    return ''

                styled_table = display_table.style.map(highlight_change, subset=['Change %'])\
                                                  .map(highlight_rsi, subset=['RSI (14)'])\
                                                  .map(highlight_rvol, subset=['RVOL (20D)'])\
                                                  .format({
                                                      'Price': '${:,.2f}',
                                                      'Change %': '{:+,.2f}%',
                                                      'RVOL (20D)': '{:.2f}x',
                                                      'RSI (14)': '{:.1f}',
                                                      'Safe Stop Loss': '${:,.2f}',
                                                      'TP1': '${:,.2f}',
                                                      'R:R TP1': '{:.1f}R',
                                                      'EMA 20': '${:,.2f}',
                                                      'EMA 50': '${:,.2f}',
                                                  })

                st.dataframe(styled_table, use_container_width=True, height=480)

                # Quick Select Action Bar
                st.markdown("#### ⚡ Click to Analyze (คลิกเลือกหุ้นเพื่อเปิดดูกราฟและแผน Anti-Stop Hunt):")
                pick_cols = st.columns(6)
                for i, row in enumerate(filtered_df.itertuples()):
                    col = pick_cols[i % 6]
                    col.button(
                        f"📊 {row.Ticker}",
                        key=f"scr_pick_{row.Ticker}",
                        on_click=set_ticker,
                        args=(row.Ticker,),
                        use_container_width=True,
                        type="primary" if row.Ticker == st.session_state.selected_ticker else "secondary"
                    )

# ----------------------------------------------------
# TAB 3: PERSONAL WATCHLIST
# ----------------------------------------------------
with tab3:
    st.subheader("⭐ ระบบ Watchlist ส่วนตัว (My Watchlist)")
    st.caption("บันทึกหุ้นที่คุณเฝ้าติดตาม พร้อมสรุปราคาและสัญญาณเทคนิคอลแบบเรียลไทม์")

    # Add Ticker Row
    wl_col1, wl_col2 = st.columns([3, 1])
    with wl_col1:
        new_ticker = st.text_input("เพิ่มหุ้นเข้า Watchlist (พิมพ์ Ticker เช่น COIN, MARA, NVDA):", key="new_wl_input").upper().strip()
    with wl_col2:
        st.write("")
        st.write("")
        if st.button("➕ เพิ่มเข้า Watchlist", use_container_width=True, type="primary"):
            if new_ticker:
                added = watchlist_manager.add_to_watchlist(new_ticker)
                if added:
                    st.success(f"เพิ่ม {new_ticker} เรียบร้อยแล้ว!")
                    st.rerun()
            else:
                st.warning("กรุณาระบุสัญลักษณ์หุ้น")

    current_watchlist = watchlist_manager.load_watchlist()

    if not current_watchlist:
        st.info("ยังไม่มีหุ้นใน Watchlist คุณสามารถพิมพ์ Ticker แล้วกดเพิ่มด้านบนได้เลย")
    else:
        st.markdown(f"**รายการหุ้นที่คุณติดตามทั้งหมด ({len(current_watchlist)} ตัว):**")
        with st.spinner("กำลังอัปเดตข้อมูลราคา Watchlist..."):
            wl_data = fetch_screener_batch(current_watchlist)

        if wl_data is not None and not wl_data.empty:
            wl_disp_cols = [c for c in ["Ticker", "Price", "Change %", "RVOL (20D)", "RSI (14)", "Setups Badges", "Safe Stop Loss", "TP1", "EMA 20"] if c in wl_data.columns]
            wl_disp = wl_data[wl_disp_cols].copy()

            def style_wl_change(val):
                color = '#4ADE80' if val > 0 else ('#F87171' if val < 0 else '#94A3B8')
                return f'color: {color}; font-weight: 700;'

            fmt_dict = {
                'Price': '${:,.2f}',
                'Change %': '{:+,.2f}%',
                'RVOL (20D)': '{:.2f}x',
                'RSI (14)': '{:.1f}',
                'Safe Stop Loss': '${:,.2f}',
                'TP1': '${:,.2f}',
                'EMA 20': '${:,.2f}',
            }
            # Only format columns that exist
            active_fmt = {k: v for k, v in fmt_dict.items() if k in wl_disp.columns}

            styled_wl = wl_disp.style.map(style_wl_change, subset=['Change %'])\
                                     .format(active_fmt)

            st.dataframe(styled_wl, use_container_width=True)

            # Manage Watchlist Action Buttons
            st.markdown("#### ⚡ จัดการหุ้นใน Watchlist:")
            for ticker_item in current_watchlist:
                w_col1, w_col2, w_col3 = st.columns([2, 2, 6])
                with w_col1:
                    st.button(
                        f"📊 วิเคราะห์ {ticker_item}",
                        key=f"wl_view_{ticker_item}",
                        on_click=set_ticker,
                        args=(ticker_item,),
                        use_container_width=True
                    )
                with w_col2:
                    if st.button(f"🗑️ ลบ {ticker_item}", key=f"wl_del_{ticker_item}", use_container_width=True):
                        watchlist_manager.remove_from_watchlist(ticker_item)
                        st.toast(f"ลบ {ticker_item} สำเร็จ")
                        st.rerun()
                with w_col3:
                    st.write("")

# ----------------------------------------------------
# TAB 4: TRADING GUIDE & WEBULL TIPS
# ----------------------------------------------------
with tab4:
    st.subheader("📖 คู่มือและเทคนิคการเทรดหุ้นสหรัฐฯ (Anti-Stop Hunt & Webull Guide)")
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("""
        ### 1. ทำความเข้าใจปรากฏการณ์ Stop Hunt ในตลาดหุ้นสหรัฐฯ
        - **Stop Hunt คืออะไร?**
          ในตลาดหุ้นสหรัฐฯ โดยเฉพาะกลุ่มหุ้นซิ่ง Volatility สูงอย่างเหมืองบิทคอยน์ (MSTR, MARA, RIOT, COIN) หรือหุ้นเทคฯ สถาบันใหญ่ (Institutional Market Makers) จะเห็น Order Book และทราบว่ารายย่อยส่วนใหญ่จะวาง Stop Loss ไว้ที่:
          1. ใต้จุดต่ำสุดล่าสุด (Swing Low) พอดี
          2. ใต้เส้นค่าเฉลี่ยกลมๆ เช่น EMA 20 หรือ EMA 50
        
        - **ผลลัพธ์:**
          ราคาจะถูกทุบลงมาทะลุแนวรับเพียงเล็กน้อย (Fake Breakdown / Liquidity Sweep) เพื่อกินคำสั่ง Stop Loss ของรายย่อย แล้วเกิดแรงเด้งดึงราคากลับขึ้นไปอย่างรวดเร็ว (V-Shape Reversal)
        
        - **วิธีแก้ทาง (Anti-Stop Hunt Strategy):**
          1. **ไม่ตั้ง Stop ชิดแนวรับ:** คำนวณบัฟเฟอร์ความผันผวนด้วยค่า **ATR (Average True Range)** เสมอ โดยถอย Stop Loss ลงมา 1.2 - 1.5 เท่าของ ATR
          2. **เข้าซื้อแบบ Reclaim Confirmation:** รอให้ราคาแหย่หลุดแนวรับแล้วดีดกลับมายืนเหนือแนวรับได้แท่งแรก ค่อยตามซื้อ
          3. **แบ่งไม้ Limit Buy ในโซน Sweep:** วางซื้อส่วนลดในโซนใต้ Swing Low เพื่อเก็บของต้นทุนถูกที่สุด
        """)

    with g_col2:
        st.markdown("""
        ### 2. วิธีตั้งคำสั่งเทรดบน Webull ให้ปลอดภัย
        - **ใช้คำสั่ง Bracket Order (OTO - One Triggers Others):**
          บนแอปพลิเคชัน Webull เมื่อทำการส่งคำสั่งซื้อ ให้ติ๊กเปิดฟังก์ชัน **Take Profit / Stop Loss (Bracket)** ไว้ตั้งแต่ตอนส่งออเดอร์ เพื่อให้ระบบวางแผนตัดขาดทุนและทำกำไรอัตโนมัติ
        
        - **เลือกใช้ Stop-Limit หรือ Stop-Market?**
          - สำหรับหุ้นสภาพคล่องสูง (เช่น AAPL, NVDA, TSLA) สามารถใช้ **Stop Market** ได้เพื่อการันตีการหนีทัน
          - สำหรับหุ้นซิ่งเหวี่ยงแรง (เช่น RIOT, MARA, SOXL) แนะนำให้ตั้ง **Stop Price** ที่ระดับ Safe SL และกำหนด **Limit Price** ให้มี Buffer เผื่อ Slippage เล็กน้อย
        
        - **การคำนวณขนาดไม้ (Position Sizing):**
          อย่าเสี่ยงเกิน 1-2% ของเงินพอร์ตทั้งหมดในแต่ละไม้เด็ดขาด หากหุ้นมีค่า ATR สูง ให้ลดจำนวนหุ้นลงเพื่อให้ความเสียหายสูงสุดยังอยู่ในงบความเสี่ยงที่รับได้
        """)
