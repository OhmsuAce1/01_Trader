import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators on OHLCV DataFrame:
    - EMA 20, 50, 200
    - RSI (14)
    - MACD (12, 26, 9)
    - ATR (14)
    - Volume 10-day SMA & Relative Volume (RVOL)
    """
    if df is None or len(df) < 5:
        return df

    df = df.copy()

    # Ensure Close, High, Low, Open are float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Exponential Moving Averages
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # RSI (14) using Wilder's Smoothing
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ATR (14) - Average True Range
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    # Volume Moving Averages (10-day & 20-day) & Relative Volume
    df['Vol_SMA_10'] = df['Volume'].rolling(window=10, min_periods=3).mean()
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20, min_periods=5).mean()
    df['RVOL'] = df['Volume'] / (df['Vol_SMA_10'] + 1e-6)
    df['RVOL_20'] = df['Volume'] / (df['Vol_SMA_20'] + 1e-6)

    # 15-day / 20-day Swing Low and Swing High
    df['Swing_Low_15'] = df['Low'].rolling(window=15, min_periods=5).min()
    df['Swing_High_15'] = df['High'].rolling(window=15, min_periods=5).max()

    # 20-day Resistance / Breakout Level (Prior 20 days excluding current day)
    df['High_20D_Prior'] = df['High'].shift(1).rolling(window=20, min_periods=5).max()
    df['Low_20D_Prior'] = df['Low'].shift(1).rolling(window=20, min_periods=5).min()

    # 52-Week High (up to 252 trading days)
    lookback_52w = min(252, len(df))
    df['High_52W_Prior'] = df['High'].shift(1).rolling(window=lookback_52w, min_periods=20).max()

    # Candle Structure & Wick Ratios (for Pinbar / Stop Hunt Sweep detection)
    body_bottom = np.minimum(df['Open'], df['Close'])
    body_top = np.maximum(df['Open'], df['Close'])
    candle_range = (df['High'] - df['Low']).replace(0, np.nan)
    df['Lower_Wick_Ratio'] = (body_bottom - df['Low']) / candle_range
    df['Upper_Wick_Ratio'] = (df['High'] - body_top) / candle_range

    return df

def detect_technical_setups(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates key technical setups on the latest candle:
    1. Volume Surge: Volume today > 200% of 20-Day Average Volume (RVOL_20 > 2.0)
    2. Breakout 20D / 52W High: Price breaks above 20-day prior high or 52W high
    3. Bullish Momentum: RSI(14) between 50 and 70 AND Price > EMA 20 & EMA 50
    4. Anti-Stop Hunt Trigger: Liquidity sweep below support/EMA reclaimed OR strong lower wick rejection (>=40%)
    """
    if df is None or len(df) < 15:
        return {
            "is_vol_surge": False,
            "is_breakout_20d": False,
            "is_52w_high": False,
            "is_breakout": False,
            "is_bullish_momentum": False,
            "is_anti_stop_hunt_trigger": False,
            "matched_setups": [],
            "badges": []
        }

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(last['Close'])
    high = float(last['High'])
    low = float(last['Low'])
    open_p = float(last['Open'])
    volume = float(last['Volume'])

    vol_sma20 = float(last.get('Vol_SMA_20', 0)) if not np.isnan(last.get('Vol_SMA_20', 0)) else 0
    rvol_20 = float(last.get('RVOL_20', 1.0)) if not np.isnan(last.get('RVOL_20', 1.0)) else 1.0

    rsi = float(last.get('RSI_14', 50.0)) if not np.isnan(last.get('RSI_14', 50.0)) else 50.0
    ema20 = float(last.get('EMA_20', price)) if not np.isnan(last.get('EMA_20', price)) else price
    ema50 = float(last.get('EMA_50', price)) if not np.isnan(last.get('EMA_50', price)) else price

    high_20d_prior = float(last.get('High_20D_Prior', 0)) if not np.isnan(last.get('High_20D_Prior', 0)) else 0
    high_52w_prior = float(last.get('High_52W_Prior', 0)) if not np.isnan(last.get('High_52W_Prior', 0)) else 0

    lower_wick_ratio = float(last.get('Lower_Wick_Ratio', 0)) if not np.isnan(last.get('Lower_Wick_Ratio', 0)) else 0
    swing_low_prior = float(prev.get('Swing_Low_15', low)) if not np.isnan(prev.get('Swing_Low_15', low)) else low

    # 1. Volume Surge (>200% of 20-day Average)
    is_vol_surge = (rvol_20 >= 2.0) and (volume >= 100_000)

    # 2. Breakout 20D / 52W High
    is_breakout_20d = (high_20d_prior > 0) and (price >= high_20d_prior)
    is_52w_high = (high_52w_prior > 0) and ((price >= high_52w_prior) or (high >= high_52w_prior * 0.998))
    is_breakout = is_breakout_20d or is_52w_high

    # 3. Bullish Momentum: RSI(14) between 50 and 70 & Price > EMA 20 & EMA 50
    is_bullish_momentum = (50.0 <= rsi <= 70.0) and (price > ema20) and (price > ema50)

    # 4. Anti-Stop Hunt Trigger:
    # A) Sweep Low Reclaim: Intraday low dipped below previous swing low, but close reclaimed above it
    sweep_low_reclaim = (low < swing_low_prior) and (price >= swing_low_prior)
    # B) EMA Reclaim / Wick Reject: Pierced EMA 20 or EMA 50 but closed above with long lower wick
    ema_reclaim = (low < ema20) and (price >= ema20) and (lower_wick_ratio >= 0.35)
    # C) Strong Pinbar/Wick Rejection: Lower wick >= 40% of candle range near support or close > open
    wick_reject = (lower_wick_ratio >= 0.40) and (price >= low + 0.5 * (high - low))

    is_anti_stop_hunt_trigger = sweep_low_reclaim or ema_reclaim or wick_reject

    # Build List of Matched Badges
    matched_setups = []
    badges = []

    if is_vol_surge:
        matched_setups.append("Volume Surge")
        badges.append({
            "name": "Volume Surge",
            "label": f"🚀 Vol Surge ({rvol_20:.1f}x)",
            "color": "#F59E0B",   # Amber / Orange
            "bg": "rgba(245, 158, 11, 0.18)"
        })

    if is_breakout:
        if is_52w_high and is_breakout_20d:
            b_label = "🌟 52W & 20D High"
        elif is_52w_high:
            b_label = "🌟 52W High"
        else:
            b_label = "💥 Breakout 20D"
        matched_setups.append("Breakout 20D / 52W High")
        badges.append({
            "name": "Breakout",
            "label": b_label,
            "color": "#EC4899",   # Pink/Magenta
            "bg": "rgba(236, 72, 153, 0.18)"
        })

    if is_bullish_momentum:
        matched_setups.append("Bullish Momentum")
        badges.append({
            "name": "Bullish Momentum",
            "label": f"📈 Bull Momentum (RSI {rsi:.0f})",
            "color": "#10B981",   # Emerald Green
            "bg": "rgba(16, 185, 129, 0.18)"
        })

    if is_anti_stop_hunt_trigger:
        tag_desc = "Sweep Reclaim" if sweep_low_reclaim else "Wick Reject"
        matched_setups.append("Anti-Stop Hunt Trigger")
        badges.append({
            "name": "Anti-Stop Hunt Trigger",
            "label": f"🪤 Anti-Stop Hunt ({tag_desc})",
            "color": "#00D4FF",   # Electric Cyan
            "bg": "rgba(0, 212, 255, 0.18)"
        })

    return {
        "is_vol_surge": is_vol_surge,
        "rvol_20": round(rvol_20, 2),
        "is_breakout_20d": is_breakout_20d,
        "is_52w_high": is_52w_high,
        "is_breakout": is_breakout,
        "is_bullish_momentum": is_bullish_momentum,
        "is_anti_stop_hunt_trigger": is_anti_stop_hunt_trigger,
        "lower_wick_ratio": round(lower_wick_ratio, 2),
        "matched_setups": matched_setups,
        "badges": badges
    }

def generate_anti_stop_hunt_plan(df: pd.DataFrame, custom_entry: float = None) -> Dict[str, Any]:
    """
    Calculates an Anti-Stop Hunt trading plan based on recent price action,
    support liquidity levels, and Average True Range (ATR).

    Why Anti-Stop Hunt?
    Market makers and institutional algorithmic orders frequently drive price
    just below conspicuous retail support levels / swing lows to trigger clustered
    stop-loss market sell orders, absorbing that liquidity to enter large longs.
    A safe plan places the stop below this 'hunt buffer' and targets a favorable R:R.
    """
    if df is None or len(df) < 15:
        return {}

    last_row = df.iloc[-1]
    current_price = float(last_row['Close'])
    atr = float(last_row['ATR_14']) if not np.isnan(last_row['ATR_14']) else current_price * 0.03

    # Look back over the past 20 bars (excluding today) to find the key structural support
    lookback = min(20, len(df) - 1)
    recent_window = df.iloc[-lookback-1 : -1]

    recent_swing_low = float(recent_window['Low'].min())
    recent_swing_high = float(recent_window['High'].max())

    # If current price is near or below swing low, find the previous secondary support
    if current_price < recent_swing_low:
        # Extend lookback to 40 bars
        ext_window = df.iloc[-min(40, len(df)) : -1]
        recent_swing_low = float(ext_window['Low'].min())

    # Stop Hunt Danger Zone:
    # Retail stops are typically concentrated right at the Swing Low or within 0.25 * ATR below it.
    retail_stop = round(recent_swing_low, 2)
    hunt_danger_lower = round(recent_swing_low - (0.5 * atr), 2)

    # Anti-Stop Hunt Safe Stop Loss:
    # Placed safely below the institutional sweep zone (1.2 to 1.5 * ATR below swing low)
    safe_stop_loss = round(recent_swing_low - (1.3 * atr), 2)
    if safe_stop_loss >= current_price:
        # If stock broke structure deeply, fallback to 1.5 * ATR below current price
        safe_stop_loss = round(current_price - (1.5 * atr), 2)

    # Entry price:
    entry_price = round(custom_entry if custom_entry and custom_entry > 0 else current_price, 2)

    # Risk per share
    risk_per_share = max(round(entry_price - safe_stop_loss, 2), 0.01)
    risk_pct = round((risk_per_share / entry_price) * 100, 2)

    # Take Profit Targets:
    # TP1: 1.5x Risk (Quick profit taking / partial scaling)
    tp1_price = round(entry_price + (1.5 * risk_per_share), 2)
    tp1_pct = round(((tp1_price - entry_price) / entry_price) * 100, 2)

    # TP2: 2.5x Risk or Swing High (Whichever offers superior R:R)
    target_by_rr = entry_price + (2.5 * risk_per_share)
    tp2_price = round(max(target_by_rr, recent_swing_high), 2)
    tp2_pct = round(((tp2_price - entry_price) / entry_price) * 100, 2)

    # Alternate Dip Buy Entry (Front-running the sweep reversal)
    dip_buy_entry = round(recent_swing_low - (0.2 * atr), 2)

    # Risk-Reward Ratio for TP1 and TP2
    rr_tp1 = 1.5
    rr_tp2 = round((tp2_price - entry_price) / risk_per_share, 2) if risk_per_share > 0 else 0

    return {
        "current_price": current_price,
        "atr_14": round(atr, 2),
        "swing_low": round(recent_swing_low, 2),
        "swing_high": round(recent_swing_high, 2),
        "retail_stop": retail_stop,
        "hunt_danger_lower": hunt_danger_lower,
        "safe_stop_loss": safe_stop_loss,
        "entry_price": entry_price,
        "dip_buy_entry": dip_buy_entry,
        "risk_per_share": risk_per_share,
        "risk_pct": risk_pct,
        "tp1_price": tp1_price,
        "tp1_pct": tp1_pct,
        "tp2_price": tp2_price,
        "tp2_pct": tp2_pct,
        "rr_tp1": rr_tp1,
        "rr_tp2": rr_tp2,
    }
