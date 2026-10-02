import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional

def create_trading_chart(
    df: pd.DataFrame,
    ticker: str,
    plan: Optional[Dict[str, Any]] = None,
    show_ema20: bool = True,
    show_ema50: bool = True,
    show_ema200: bool = True,
    show_stop_hunt_overlay: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True
) -> go.Figure:
    """
    Constructs a comprehensive dark-mode multi-panel trading chart (Webull style):
    - Panel 1: Candlestick + EMAs + Anti-Stop Hunt Trading Plan Overlay
    - Panel 2: Volume + 10D SMA
    - Panel 3: RSI (14) with 70/30 bands
    - Panel 4: MACD (12, 26, 9) with histogram
    """
    if df is None or df.empty:
        return go.Figure()

    # Determine row counts and heights dynamically
    rows = 2 # Price + Volume are mandatory
    row_heights = [0.55, 0.15]
    panel_titles = [f"{ticker} Candlestick Chart", "Volume (10D SMA)"]

    if show_rsi:
        rows += 1
        row_heights = [0.45, 0.15, 0.20]
        panel_titles.append("RSI (14)")

    if show_macd:
        rows += 1
        if show_rsi:
            row_heights = [0.44, 0.14, 0.21, 0.21]
            panel_titles.append("MACD (12, 26, 9)")
        else:
            row_heights = [0.50, 0.16, 0.34]
            panel_titles.append("MACD (12, 26, 9)")

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=panel_titles,
        row_heights=row_heights
    )

    # 1. Candlestick
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="OHLC",
        increasing_line_color="#00F0A8",
        increasing_fillcolor="#00F0A8",
        decreasing_line_color="#FF3B69",
        decreasing_fillcolor="#FF3B69",
    )
    fig.add_trace(candlestick, row=1, col=1)

    # EMAs
    if show_ema20 and 'EMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['EMA_20'],
                line=dict(color="#FFD700", width=1.5),
                name="EMA 20"
            ),
            row=1, col=1
        )
    if show_ema50 and 'EMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['EMA_50'],
                line=dict(color="#00D4FF", width=1.5),
                name="EMA 50"
            ),
            row=1, col=1
        )
    if show_ema200 and 'EMA_200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['EMA_200'],
                line=dict(color="#BD00FF", width=2.0),
                name="EMA 200"
            ),
            row=1, col=1
        )

    # Anti-Stop Hunt Levels Overlay
    if show_stop_hunt_overlay and plan:
        entry = plan.get("entry_price")
        safe_sl = plan.get("safe_stop_loss")
        retail_sl = plan.get("retail_stop")
        tp1 = plan.get("tp1_price")
        tp2 = plan.get("tp2_price")

        # Shaded Stop Hunt Danger Zone
        if retail_sl and safe_sl and retail_sl > safe_sl:
            fig.add_hrect(
                y0=safe_sl,
                y1=retail_sl,
                fillcolor="rgba(239, 68, 68, 0.15)",
                line_width=0,
                layer="below",
                annotation_text="⚠️ Stop Hunt Liquidity Sweep Zone",
                annotation_position="bottom right",
                annotation_font=dict(color="#F87171", size=10),
                row=1, col=1
            )

        # Lines
        if entry:
            fig.add_hline(
                y=entry, line_dash="dash", line_color="#EAB308", line_width=1.5,
                annotation_text=f"🟡 Entry (${entry})",
                annotation_position="top left",
                annotation_font=dict(color="#FDE047", size=11),
                row=1, col=1
            )
        if retail_sl:
            fig.add_hline(
                y=retail_sl, line_dash="dot", line_color="#F97316", line_width=1.2,
                annotation_text=f"🟠 Retail Stop (${retail_sl})",
                annotation_position="bottom left",
                annotation_font=dict(color="#FB923C", size=10),
                row=1, col=1
            )
        if safe_sl:
            fig.add_hline(
                y=safe_sl, line_dash="solid", line_color="#EF4444", line_width=2.0,
                annotation_text=f"🛡️ Anti-Stop Hunt SL (${safe_sl})",
                annotation_position="bottom left",
                annotation_font=dict(color="#F87171", size=11),
                row=1, col=1
            )
        if tp1:
            fig.add_hline(
                y=tp1, line_dash="dash", line_color="#22C55E", line_width=1.5,
                annotation_text=f"🟢 TP 1 (${tp1} / +{plan.get('tp1_pct')}%)",
                annotation_position="top left",
                annotation_font=dict(color="#4ADE80", size=11),
                row=1, col=1
            )
        if tp2:
            fig.add_hline(
                y=tp2, line_dash="dash", line_color="#10B981", line_width=1.5,
                annotation_text=f"🎯 TP 2 (${tp2} / +{plan.get('tp2_pct')}%)",
                annotation_position="top left",
                annotation_font=dict(color="#34D399", size=11),
                row=1, col=1
            )

    # 2. Volume panel
    colors = ['#00F0A8' if c >= o else '#FF3B69' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['Volume'],
            marker_color=colors,
            name="Volume",
            opacity=0.85
        ),
        row=2, col=1
    )
    if 'Vol_SMA_10' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Vol_SMA_10'],
                line=dict(color="#E2E8F0", width=1.2),
                name="10D Avg Vol"
            ),
            row=2, col=1
        )

    current_row = 3

    # 3. RSI Panel
    if show_rsi and 'RSI_14' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['RSI_14'],
                line=dict(color="#A855F7", width=1.8),
                name="RSI 14"
            ),
            row=current_row, col=1
        )
        # Overbought (70) & Oversold (30) lines
        fig.add_hline(
            y=70, line_dash="dash", line_color="#EF4444", line_width=1.0,
            annotation_text="Overbought (70)", annotation_position="top right",
            annotation_font=dict(color="#F87171", size=10),
            row=current_row, col=1
        )
        fig.add_hline(
            y=30, line_dash="dash", line_color="#22C55E", line_width=1.0,
            annotation_text="Oversold (30)", annotation_position="bottom right",
            annotation_font=dict(color="#4ADE80", size=10),
            row=current_row, col=1
        )
        fig.add_hline(
            y=50, line_dash="dot", line_color="#64748B", line_width=1.0,
            row=current_row, col=1
        )
        current_row += 1

    # 4. MACD Panel
    if show_macd and 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['MACD'],
                line=dict(color="#38BDF8", width=1.5),
                name="MACD"
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['MACD_Signal'],
                line=dict(color="#FB923C", width=1.5),
                name="Signal"
            ),
            row=current_row, col=1
        )
        hist_colors = ['#22C55E' if h >= 0 else '#EF4444' for h in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['MACD_Hist'],
                marker_color=hist_colors,
                name="Histogram",
                opacity=0.7
            ),
            row=current_row, col=1
        )

    # Overall Dark Styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0E131C",
        margin=dict(l=10, r=10, t=30, b=10),
        height=850,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="#1E293B",
            linecolor="#334155",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="#64748B"
        ),
        hovermode="x unified"
    )

    # Style all y-axes
    for i in range(1, rows + 1):
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#1E293B",
            linecolor="#334155",
            zerolinecolor="#334155",
            row=i, col=1
        )

    return fig

def plot_equity_curve(
    equity_df: pd.DataFrame,
    ticker: str,
    initial_capital: float = 10000.0
) -> go.Figure:
    """
    Renders an interactive Equity Curve chart for the Anti-Stop Hunt strategy backtest.
    Visualizes capital progression over time with a breakeven benchmark.
    """
    if equity_df is None or equity_df.empty or len(equity_df) < 2:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B0E14",
            plot_bgcolor="#0E131C",
            annotations=[dict(text="ไม่มีข้อมูลการเทรดที่ปิดเพื่อวาด Equity Curve", showarrow=False, font=dict(size=14, color="#94A3B8"))],
            height=300
        )
        return empty_fig

    fig = go.Figure()

    # Breakeven Baseline
    fig.add_trace(
        go.Scatter(
            x=equity_df['Date'],
            y=[initial_capital] * len(equity_df),
            mode='lines',
            line=dict(color='#64748B', width=1.5, dash='dash'),
            name='เงินทุนเริ่มต้น (Initial Capital)',
            hoverinfo='skip'
        )
    )

    # Determine curve color based on ending capital
    ending_capital = equity_df['Equity'].iloc[-1]
    line_color = '#10B981' if ending_capital >= initial_capital else '#EF4444'
    fill_color = 'rgba(16, 185, 129, 0.12)' if ending_capital >= initial_capital else 'rgba(239, 68, 68, 0.12)'

    # Equity Curve Line
    fig.add_trace(
        go.Scatter(
            x=equity_df['Date'],
            y=equity_df['Equity'],
            mode='lines+markers',
            line=dict(color=line_color, width=2.5),
            marker=dict(size=6, color=line_color),
            fill='tonexty' if ending_capital >= initial_capital else 'none',
            fillcolor=fill_color,
            name=f'พอร์ตจำลอง ({ticker})',
            customdata=equity_df['Cumulative_R'],
            hovertemplate="<b>วันที่: %{x}</b><br>มูลค่าพอร์ต: $%{y:,.2f}<br>กำไรสะสม: %{customdata:+.2f}R<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0E131C",
        margin=dict(l=10, r=10, t=25, b=10),
        height=340,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#1E293B",
            linecolor="#334155"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1E293B",
            linecolor="#334155",
            tickformat="$,.0f"
        ),
        hovermode="x unified"
    )

    return fig

def create_sector_treemap(sector_df: pd.DataFrame, timeframe_col: str = "1D %") -> go.Figure:
    """
    Builds a Finviz-style Heatmap Treemap of Sector ETF performance.
    """
    if sector_df is None or sector_df.empty:
        return go.Figure()

    labels = []
    parents = []
    values = []
    colors = []
    customdata = []

    for _, row in sector_df.iterrows():
        val = row[timeframe_col]
        sign = "+" if val >= 0 else ""
        short_name = row['Sector'].split(" (")[0]
        label_text = f"<b>{row['Ticker']}</b><br>{short_name}<br><b>{sign}{val:.2f}%</b>"
        
        labels.append(label_text)
        parents.append("")
        values.append(10)  # Uniform sizing for balanced tiles
        colors.append(val)
        customdata.append([row['Ticker'], row['Sector'], row['Price'], val])

    # Dynamic color scale based on min/max with zero center
    max_abs = max(abs(min(colors)), abs(max(colors)), 2.0)

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale=[
                [0.0, '#DC2626'],    # Dark Red
                [0.35, '#EF4444'],   # Red
                [0.5, '#1E293B'],    # Slate/Neutral
                [0.65, '#22C55E'],   # Green
                [1.0, '#10B981']     # Dark Green
            ],
            cmin=-max_abs,
            cmax=max_abs,
            cmid=0,
            showscale=True,
            colorbar=dict(
                title=dict(text=f"{timeframe_col}", side="top", font=dict(color="#94A3B8", size=10)),
                ticksuffix="%",
                thickness=12,
                len=0.75,
                tickfont=dict(color="#94A3B8", size=9)
            )
        ),
        textfont=dict(size=14, family="sans-serif"),
        hovertemplate="<b>%{customdata[0]} - %{customdata[1]}</b><br>ราคา: $%{customdata[2]:,.2f}<br>ผลตอบแทน: %{customdata[3]:+.2f}%<extra></extra>",
        customdata=customdata
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0E131C",
        margin=dict(l=5, r=5, t=15, b=5),
        height=380
    )
    return fig

def create_sector_barchart(sector_df: pd.DataFrame, timeframe_col: str = "1D %") -> go.Figure:
    """
    Builds a sorted horizontal bar chart comparing Sector performance.
    """
    if sector_df is None or sector_df.empty:
        return go.Figure()

    df_sorted = sector_df.sort_values(by=timeframe_col, ascending=True).reset_index(drop=True)
    
    colors = ['#10B981' if v >= 0 else '#EF4444' for v in df_sorted[timeframe_col]]
    labels = [f"{'+' if v>=0 else ''}{v:.2f}%" for v in df_sorted[timeframe_col]]

    fig = go.Figure(go.Bar(
        x=df_sorted[timeframe_col],
        y=[f"{row['Ticker']} - {row['Sector'].split(' (')[0]}" for _, row in df_sorted.iterrows()],
        orientation='h',
        marker=dict(color=colors, line=dict(color='#0B0E14', width=1)),
        text=labels,
        textposition='outside',
        textfont=dict(color="#F8FAFC", size=11, family="sans-serif"),
        hovertemplate="<b>%{y}</b><br>ผลตอบแทน: %{x:+.2f}%<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B0E14",
        plot_bgcolor="#0E131C",
        margin=dict(l=10, r=40, t=15, b=10),
        height=420,
        xaxis=dict(
            showgrid=True,
            gridcolor="#1E293B",
            ticksuffix="%",
            zeroline=True,
            zerolinecolor="#475569",
            zerolinewidth=1.5
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#E2E8F0")
        )
    )
    return fig


