import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from indicators import compute_indicators

def run_antistophunt_backtest(
    df: pd.DataFrame,
    risk_reward_ratio: float = 2.0,
    lookback_bars: Optional[int] = None,
    risk_per_trade_pct: float = 2.0,
    initial_capital: float = 10000.0,
    max_holding_bars: int = 30
) -> Dict[str, Any]:
    """
    Backtests the Anti-Stop Hunt strategy on historical price action:
    1. Identifies Liquidity Sweep Reclaim and Lower-Wick Rejection signals.
    2. Places Safe Stop Loss below the institutional stop-hunt zone (Swing Low - 1.3x ATR).
    3. Targets Take Profit based on the chosen Risk-to-Reward ratio (e.g., 2.0R).
    4. Simulates forward execution bar-by-bar without lookahead bias.
    5. Calculates key metrics: Win Rate, Profit Factor, Total R, Max Drawdown, and Equity Curve.
    """
    if df is None or len(df) < 30:
        return {
            "error": "ข้อมูลราคาไม่เพียงพอสำหรับการทำ Backtest (ต้องมีอย่างน้อย 30 วัน)",
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl_r": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": [],
            "equity_df": pd.DataFrame()
        }

    # Ensure indicators are calculated
    if 'ATR_14' not in df.columns or 'Swing_Low_15' not in df.columns:
        df = compute_indicators(df)

    # Slice lookback period if specified
    if lookback_bars and lookback_bars > 0 and len(df) > lookback_bars:
        df = df.iloc[-lookback_bars:].copy()

    trades = []
    in_trade = False
    current_trade = {}
    
    dates = df.index
    n_bars = len(df)

    # Start loop after initial window for indicators
    start_idx = 20

    for i in range(start_idx, n_bars - 1):
        # 1. Manage Active Trade
        if in_trade:
            high_j = float(df['High'].iloc[i])
            low_j = float(df['Low'].iloc[i])
            close_j = float(df['Close'].iloc[i])
            open_j = float(df['Open'].iloc[i])
            date_j = dates[i]
            
            bars_held = i - current_trade['entry_bar_idx']
            entry_p = current_trade['entry_price']
            sl_p = current_trade['stop_loss']
            tp_p = current_trade['take_profit']
            risk_unit = current_trade['risk_per_share']

            hit_tp = high_j >= tp_p
            hit_sl = low_j <= sl_p

            if hit_sl and hit_tp:
                # Same-bar conflict: Conservative execution assumes SL hit first
                # unless Open opened above TP
                if open_j >= tp_p:
                    hit_sl = False
                else:
                    hit_tp = False

            if hit_tp:
                pnl_r = risk_reward_ratio
                pnl_pct = ((tp_p - entry_p) / entry_p) * 100
                dollar_pnl = current_trade['position_dollars'] * (pnl_pct / 100.0)

                trades.append({
                    "entry_date": current_trade['entry_date'],
                    "exit_date": str(date_j)[:10],
                    "entry_price": round(entry_p, 2),
                    "exit_price": round(tp_p, 2),
                    "stop_loss": round(sl_p, 2),
                    "take_profit": round(tp_p, 2),
                    "result": "WIN",
                    "pnl_r": round(pnl_r, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "dollar_pnl": round(dollar_pnl, 2),
                    "bars_held": bars_held,
                    "setup_type": current_trade['setup_type']
                })
                in_trade = False
                continue

            elif hit_sl:
                pnl_r = -1.0
                pnl_pct = ((sl_p - entry_p) / entry_p) * 100
                dollar_pnl = current_trade['position_dollars'] * (pnl_pct / 100.0)

                trades.append({
                    "entry_date": current_trade['entry_date'],
                    "exit_date": str(date_j)[:10],
                    "entry_price": round(entry_p, 2),
                    "exit_price": round(sl_p, 2),
                    "stop_loss": round(sl_p, 2),
                    "take_profit": round(tp_p, 2),
                    "result": "LOSS",
                    "pnl_r": round(pnl_r, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "dollar_pnl": round(dollar_pnl, 2),
                    "bars_held": bars_held,
                    "setup_type": current_trade['setup_type']
                })
                in_trade = False
                continue

            elif bars_held >= max_holding_bars or i == n_bars - 2:
                # Time-based exit at market close
                pnl_pct = ((close_j - entry_p) / entry_p) * 100
                pnl_r = (close_j - entry_p) / risk_unit if risk_unit > 0 else 0
                dollar_pnl = current_trade['position_dollars'] * (pnl_pct / 100.0)

                trades.append({
                    "entry_date": current_trade['entry_date'],
                    "exit_date": str(date_j)[:10],
                    "entry_price": round(entry_p, 2),
                    "exit_price": round(close_j, 2),
                    "stop_loss": round(sl_p, 2),
                    "take_profit": round(tp_p, 2),
                    "result": "WIN" if pnl_r > 0 else "LOSS",
                    "pnl_r": round(pnl_r, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "dollar_pnl": round(dollar_pnl, 2),
                    "bars_held": bars_held,
                    "setup_type": current_trade['setup_type']
                })
                in_trade = False
                continue

        # 2. Check for New Entry Setup on Bar i (if not currently in a trade)
        if not in_trade:
            price_i = float(df['Close'].iloc[i])
            low_i = float(df['Low'].iloc[i])
            high_i = float(df['High'].iloc[i])
            atr_i = float(df['ATR_14'].iloc[i]) if not np.isnan(df['ATR_14'].iloc[i]) else price_i * 0.03
            
            # Prior swing low (past 15 bars prior to bar i)
            swing_low_prior = float(df['Low'].iloc[max(0, i-15):i].min())
            lower_wick_ratio = float(df.get('Lower_Wick_Ratio', pd.Series(0)).iloc[i]) if 'Lower_Wick_Ratio' in df.columns else 0.0
            ema20_i = float(df['EMA_20'].iloc[i]) if 'EMA_20' in df.columns else price_i

            # Setup A: Sweep Low Reclaim (Low dipped below swing low, closed back above)
            sweep_low_reclaim = (low_i < swing_low_prior) and (price_i >= swing_low_prior)

            # Setup B: Strong Lower Wick Rejection (>=40%) at support or above EMA20
            wick_reject = (lower_wick_ratio >= 0.40) and (price_i >= low_i + 0.5 * (high_i - low_i)) and (price_i >= ema20_i * 0.98 or low_i <= swing_low_prior * 1.02)

            if sweep_low_reclaim or wick_reject:
                # Realistic entry at next bar open
                entry_price = float(df['Open'].iloc[i+1])
                entry_date = str(dates[i+1])[:10]

                # Safe Stop Loss: Under the institutional sweep zone
                safe_stop_loss = round(swing_low_prior - (1.3 * atr_i), 2)
                if safe_stop_loss >= entry_price:
                    safe_stop_loss = round(entry_price - (1.3 * atr_i), 2)

                risk_per_share = max(entry_price - safe_stop_loss, entry_price * 0.01)
                take_profit = round(entry_price + (risk_reward_ratio * risk_per_share), 2)

                # Dollar sizing based on account risk budget
                account_risk_budget = initial_capital * (risk_per_trade_pct / 100.0)
                shares = max(int(account_risk_budget / risk_per_share), 1)
                position_dollars = shares * entry_price

                setup_type = "Sweep Low Reclaim" if sweep_low_reclaim else "Support Wick Rejection"

                in_trade = True
                current_trade = {
                    "entry_bar_idx": i + 1,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "stop_loss": safe_stop_loss,
                    "take_profit": take_profit,
                    "risk_per_share": risk_per_share,
                    "position_dollars": position_dollars,
                    "setup_type": setup_type
                }

    # 3. Calculate Aggregated Metrics & Equity Curve
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_pnl_r": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "wins": 0,
            "losses": 0,
            "avg_trade_r": 0.0,
            "avg_holding_days": 0.0,
            "trades": [],
            "equity_df": pd.DataFrame()
        }

    trades_df = pd.DataFrame(trades)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = len(trades_df[trades_df['result'] == 'LOSS'])
    total_trades = len(trades_df)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0

    gross_profit = trades_df[trades_df['dollar_pnl'] > 0]['dollar_pnl'].sum()
    gross_loss = abs(trades_df[trades_df['dollar_pnl'] < 0]['dollar_pnl'].sum())
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    total_pnl_r = round(trades_df['pnl_r'].sum(), 2)
    avg_trade_r = round(trades_df['pnl_r'].mean(), 2)
    avg_holding_days = round(trades_df['bars_held'].mean(), 1)

    # Build Equity Curve Series
    equity_series = [initial_capital]
    cum_r_series = [0.0]
    peak_equity = initial_capital
    max_drawdown_pct = 0.0

    current_eq = initial_capital
    current_r = 0.0

    for _, tr in trades_df.iterrows():
        current_eq += tr['dollar_pnl']
        current_r += tr['pnl_r']
        equity_series.append(current_eq)
        cum_r_series.append(current_r)

        if current_eq > peak_equity:
            peak_equity = current_eq
        dd = ((peak_equity - current_eq) / peak_equity) * 100 if peak_equity > 0 else 0.0
        if dd > max_drawdown_pct:
            max_drawdown_pct = dd

    total_return_pct = round(((current_eq - initial_capital) / initial_capital) * 100, 2)
    max_drawdown_pct = round(max_drawdown_pct, 2)

    # Construct Equity Curve DataFrame
    trade_dates = ["Start"] + trades_df['exit_date'].tolist()
    equity_df = pd.DataFrame({
        "Date": trade_dates,
        "Equity": equity_series,
        "Cumulative_R": cum_r_series
    })

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "profit_factor": profit_factor,
        "total_pnl_r": total_pnl_r,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_trade_r": avg_trade_r,
        "avg_holding_days": avg_holding_days,
        "initial_capital": initial_capital,
        "ending_capital": round(current_eq, 2),
        "trades": trades,
        "trades_df": trades_df,
        "equity_df": equity_df
    }
