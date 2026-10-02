import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import streamlit as st
import concurrent.futures
from indicators import compute_indicators, detect_technical_setups, generate_anti_stop_hunt_plan

POPULAR_TICKERS = {
    "Bitcoin & Crypto Proxies 🪙": [
        "MSTR", "COIN", "MARA", "RIOT", "CLSK", "CIFR", "HUT", "IBIT", "CORZ", "WULF", "CONL", "BITO"
    ],
    "Mega Cap & AI Leaders 🚀": [
        "NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "PLTR", "ARM", "SMCI", "AVGO", "TSM"
    ],
    "High Beta & Webull Favorites 🔥": [
        "SOXL", "TQQQ", "HOOD", "CELH", "CVNA", "IONQ", "DKNG", "APP", "SOFI", "ASTS", "RKLB", "COHR", "CRWD", "PANW", "UBER"
    ],
    "Leveraged ETFs & High Volatility ⚡": [
        "NVDL", "MSTX", "UPRO", "FNGU", "LABU", "MSTZ", "SOXS", "SQQQ"
    ]
}

# Flattens all unique tickers into screener list (~48 tickers)
ALL_SCREENER_TICKERS = sorted(list({
    tick
    for group in POPULAR_TICKERS.values()
    for tick in group
}))

# Major S&P Sector ETFs
SECTOR_ETFS = {
    "XLK": "Technology (เทคโนโลยี)",
    "XLF": "Financials (การเงิน/ธนาคาร)",
    "XLE": "Energy (พลังงาน/น้ำมัน)",
    "XLV": "Health Care (การแพทย์/ยา)",
    "XLI": "Industrials (อุตสาหกรรม)",
    "XLY": "Consumer Discretionary (สินค้าฟุ่มเฟือย)",
    "XLP": "Consumer Staples (สินค้าจำเป็น)",
    "XLU": "Utilities (สาธารณูปโภค)",
    "XLB": "Materials (วัสดุ/ปิโตรเคมี)",
    "XLC": "Communication (สื่อสาร/บริการมีเดีย)",
    "IYR": "Real Estate (อสังหาริมทรัพย์/REITs)"
}

# Major US Market Indices
MAJOR_INDICES = {
    "SPY": "S&P 500 (ตลาดรวมสหรัฐฯ)",
    "QQQ": "Nasdaq 100 (หุ้นเทคฯ ขนาดใหญ่)",
    "DIA": "Dow Jones (หุ้นอุตสาหกรรมหลัก)",
    "IWM": "Russell 2000 (หุ้นขนาดกลาง-เล็ก)"
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_sector_performance() -> pd.DataFrame:
    """
    Downloads historical data for all 11 Sector ETFs and computes returns:
    1-Day (1D), 5-Day (5D), and 1-Month (1M).
    """
    symbols = list(SECTOR_ETFS.keys())
    try:
        batch_df = yf.download(
            " ".join(symbols),
            period="3mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception:
        batch_df = None

    rows = []
    for s in symbols:
        try:
            if batch_df is not None and not batch_df.empty and s in batch_df.columns.levels[0]:
                sdf = batch_df[s].dropna(subset=['Close'])
            else:
                sdf = yf.Ticker(s).history(period="3mo", interval="1d", auto_adjust=False)

            if sdf is None or len(sdf) < 2:
                continue

            close = sdf['Close']
            curr_p = float(close.iloc[-1])
            prev_p = float(close.iloc[-2])
            chg_1d = ((curr_p - prev_p) / prev_p) * 100

            p_5d = float(close.iloc[-6]) if len(close) >= 6 else prev_p
            chg_5d = ((curr_p - p_5d) / p_5d) * 100

            p_1m = float(close.iloc[-22]) if len(close) >= 22 else float(close.iloc[0])
            chg_1m = ((curr_p - p_1m) / p_1m) * 100

            vol = int(sdf['Volume'].iloc[-1]) if 'Volume' in sdf.columns else 0

            rows.append({
                "Ticker": s,
                "Sector": SECTOR_ETFS[s],
                "Price": round(curr_p, 2),
                "1D %": round(chg_1d, 2),
                "5D %": round(chg_5d, 2),
                "1M %": round(chg_1m, 2),
                "Volume": vol
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_overview() -> Dict[str, Any]:
    """
    Fetches major index performance (SPY, QQQ, DIA, IWM) and computes
    overall Market Breadth (% above EMA50, Advance/Decline ratio) from active stocks.
    """
    idx_symbols = list(MAJOR_INDICES.keys())
    try:
        idx_df = yf.download(
            " ".join(idx_symbols),
            period="1mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception:
        idx_df = None

    indices_data = []
    for sym in idx_symbols:
        try:
            if idx_df is not None and not idx_df.empty and sym in idx_df.columns.levels[0]:
                df_s = idx_df[sym].dropna(subset=['Close'])
            else:
                df_s = yf.Ticker(sym).history(period="1mo", interval="1d", auto_adjust=False)

            if df_s is not None and len(df_s) >= 2:
                close = df_s['Close']
                p_last = float(close.iloc[-1])
                p_prev = float(close.iloc[-2])
                chg_1d = ((p_last - p_prev) / p_prev) * 100

                p_5d = float(close.iloc[-6]) if len(close) >= 6 else p_prev
                chg_5d = ((p_last - p_5d) / p_5d) * 100

                indices_data.append({
                    "symbol": sym,
                    "name": MAJOR_INDICES[sym],
                    "price": round(p_last, 2),
                    "change_1d": round(chg_1d, 2),
                    "change_5d": round(chg_5d, 2),
                })
        except Exception:
            continue

    # Market Breadth from Screener stocks
    screener_df = fetch_screener_batch(ALL_SCREENER_TICKERS[:36])
    advances = 0
    declines = 0
    above_ema50 = 0
    total_stocks = len(screener_df) if screener_df is not None else 0

    if total_stocks > 0:
        advances = int((screener_df['Change %'] > 0).sum())
        declines = int((screener_df['Change %'] < 0).sum())
        above_ema50 = int((screener_df['Price'] > screener_df['EMA 50']).sum())
        pct_above_ema50 = round((above_ema50 / total_stocks) * 100, 1)
        ad_ratio = round(advances / declines, 2) if declines > 0 else float(advances)
    else:
        pct_above_ema50 = 50.0
        ad_ratio = 1.0

    if pct_above_ema50 >= 65.0:
        breadth_sentiment = "🟢 ตลาดแข็งแกร่ง (Strong Bullish Breadth)"
    elif pct_above_ema50 >= 45.0:
        breadth_sentiment = "🟡 ตลาดทรงตัว / คัดเลือกรายตัว (Neutral / Selective)"
    else:
        breadth_sentiment = "🔴 ตลาดอ่อนแอ / เสี่ยงปรับฐาน (Bearish Breadth)"

    return {
        "indices": indices_data,
        "breadth": {
            "advances": advances,
            "declines": declines,
            "ad_ratio": ad_ratio,
            "above_ema50": above_ema50,
            "total_stocks": total_stocks,
            "pct_above_ema50": pct_above_ema50,
            "sentiment": breadth_sentiment
        }
    }

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetches historical OHLCV data using yfinance and calculates indicators.
    Cached for 5 minutes.
    """
    try:
        ticker = ticker.upper().strip()
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)

        if df is None or df.empty or len(df) < 5:
            return None

        # Clean index
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None)

        df = compute_indicators(df)
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """
    Fetches summary info for a single ticker.
    """
    try:
        ticker = ticker.upper().strip()
        t = yf.Ticker(ticker)
        info = t.info or {}
        
        # Fast fallback if info dict is incomplete
        fast_info = getattr(t, "fast_info", None)

        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector", "N/A")
        market_cap = info.get("marketCap", 0)
        
        prev_close = info.get("previousClose") or (fast_info.previous_close if fast_info else None) or 0.0
        current_price = info.get("currentPrice") or (fast_info.last_price if fast_info else None) or prev_close

        change = current_price - prev_close if prev_close else 0.0
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        return {
            "symbol": ticker,
            "name": name,
            "sector": sector,
            "market_cap": market_cap,
            "current_price": round(current_price, 2),
            "previous_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "day_high": round(info.get("dayHigh") or current_price, 2),
            "day_low": round(info.get("dayLow") or current_price, 2),
            "fifty_two_week_high": round(info.get("fiftyTwoWeekHigh") or 0.0, 2),
            "fifty_two_week_low": round(info.get("fiftyTwoWeekLow") or 0.0, 2),
        }
    except Exception:
        return {
            "symbol": ticker,
            "name": ticker,
            "sector": "N/A",
            "market_cap": 0,
            "current_price": 0.0,
            "previous_close": 0.0,
            "change": 0.0,
            "change_pct": 0.0,
            "day_high": 0.0,
            "day_low": 0.0,
            "fifty_two_week_high": 0.0,
            "fifty_two_week_low": 0.0,
        }

def _process_single_df(ticker: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Helper to compute indicators and extract screener row for a single ticker."""
    try:
        if df is None or len(df) < 15:
            return None

        # Clean index and missing values
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None)
        df = df.dropna(subset=['Close'])
        if len(df) < 15:
            return None

        df = compute_indicators(df)
        setups = detect_technical_setups(df)
        plan = generate_anti_stop_hunt_plan(df)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        price = float(last['Close'])
        prev_close = float(prev['Close'])
        chg_pct = ((price - prev_close) / prev_close) * 100

        ema20 = float(last['EMA_20']) if not np.isnan(last['EMA_20']) else price
        ema50 = float(last['EMA_50']) if not np.isnan(last['EMA_50']) else price
        ema200 = float(last['EMA_200']) if not np.isnan(last['EMA_200']) else price

        rsi = float(last['RSI_14']) if not np.isnan(last['RSI_14']) else 50.0
        volume = float(last['Volume'])
        vol_avg20 = float(last.get('Vol_SMA_20', volume)) if not np.isnan(last.get('Vol_SMA_20', volume)) else volume
        rvol_20 = float(last.get('RVOL_20', 1.0)) if not np.isnan(last.get('RVOL_20', 1.0)) else 1.0
        rvol_10 = float(last.get('RVOL', 1.0)) if not np.isnan(last.get('RVOL', 1.0)) else 1.0

        swing_low = float(last.get('Swing_Low_15', price * 0.95))
        atr = float(last.get('ATR_14', price * 0.03))

        badges_text = " • ".join([b["label"] for b in setups["badges"]]) if setups["badges"] else "—"

        return {
            "Ticker": ticker,
            "Price": round(price, 2),
            "Change %": round(chg_pct, 2),
            "Volume": int(volume),
            "Avg Vol (20D)": int(vol_avg20),
            "RVOL (20D)": round(rvol_20, 2),
            "RVOL (10D)": round(rvol_10, 2),
            "RSI (14)": round(rsi, 1),
            "EMA 20": round(ema20, 2),
            "EMA 50": round(ema50, 2),
            "EMA 200": round(ema200, 2),
            "Setups Badges": badges_text,
            "Badges List": setups["badges"],
            "Matched Setups": setups["matched_setups"],
            "Vol Surge": setups["is_vol_surge"],
            "Breakout": setups["is_breakout"],
            "Breakout 20D": setups["is_breakout_20d"],
            "52W High": setups["is_52w_high"],
            "Bullish Momentum": setups["is_bullish_momentum"],
            "Anti-Stop Hunt Trigger": setups["is_anti_stop_hunt_trigger"],
            "Safe Stop Loss": plan.get("safe_stop_loss", round(price * 0.95, 2)),
            "Entry Price": plan.get("entry_price", round(price, 2)),
            "TP1": plan.get("tp1_price", round(price * 1.05, 2)),
            "TP2": plan.get("tp2_price", round(price * 1.10, 2)),
            "Risk %": plan.get("risk_pct", 5.0),
            "R:R TP1": plan.get("rr_tp1", 1.5),
            "R:R TP2": plan.get("rr_tp2", 2.5),
            "Swing Low": round(swing_low, 2),
            "ATR": round(atr, 2),
        }
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_screener_batch(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches and screens a list of tickers utilizing multi-threaded batch download,
    calculating key technical setups:
    - Volume Surge: Volume > 200% of 20-Day SMA
    - Breakout 20D / 52W High
    - Bullish Momentum: RSI between 50-70 & Price > EMA 20 & EMA 50
    - Anti-Stop Hunt Trigger: Liquidity sweep reclaim / lower wick rejection
    """
    if not tickers:
        return pd.DataFrame()

    rows = []
    failed_tickers = []
    batch_df = None

    # Step 1: Batch download all tickers in 1 request (High Performance)
    try:
        ticker_str = " ".join(tickers)
        batch_df = yf.download(
            ticker_str,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True
        )
    except Exception:
        batch_df = None

    # Step 2: Parse batch data
    for ticker in tickers:
        processed = False
        try:
            if batch_df is not None and not batch_df.empty:
                if len(tickers) == 1:
                    df = batch_df.copy()
                    item = _process_single_df(ticker, df)
                    if item:
                        rows.append(item)
                        processed = True
                else:
                    if ticker in batch_df.columns.levels[0]:
                        df = batch_df[ticker].dropna(how="all").copy()
                        item = _process_single_df(ticker, df)
                        if item:
                            rows.append(item)
                            processed = True
        except Exception:
            pass

        if not processed:
            failed_tickers.append(ticker)

    # Step 3: ThreadPool fallback for any missing or failed tickers
    if failed_tickers:
        def _fetch_fallback(t):
            try:
                single_df = yf.Ticker(t).history(period="1y", interval="1d", auto_adjust=False)
                return _process_single_df(t, single_df)
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(failed_tickers))) as executor:
            future_to_tick = {executor.submit(_fetch_fallback, t): t for t in failed_tickers}
            for future in concurrent.futures.as_completed(future_to_tick):
                res = future.result()
                if res:
                    rows.append(res)

    if not rows:
        return pd.DataFrame()

    df_result = pd.DataFrame(rows)
    # Sort primarily by Volume Surge, Breakout, Bullish Momentum, or RVOL
    if "RVOL (20D)" in df_result.columns:
        df_result = df_result.sort_values(by="RVOL (20D)", ascending=False).reset_index(drop=True)
    return df_result
