# backend/backtest.py
import base64
from io import BytesIO
import os
import asyncio
from backend.fetchYfinance import fetch_yfinance_data
from backtesting import Backtest
from backend.strategy import AdvancedTrendStrategyWithSR
import logging
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


async def run_backtest(
    ticker: str, start_date: str, end_date: str, cash: float, commission: float
):
    # logger.debug(f"Running backtest for {ticker} from {start_date} to {end_date}")
    data = await fetch_yfinance_data(ticker, start_date, end_date)

    # logger.debug(f"Data fetched: {len(data)} rows")

    if len(data) < 200:
        # logger.error("Data length too short for backtest")
        raise ValueError("Insufficient data for backtest")

    bt = Backtest(
        data,
        AdvancedTrendStrategyWithSR,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
    )

    stats = bt.run()
    results = bt._results
    trades = stats['_trades']
    equity = results['_equity_curve']

    # Create figure with 2 subplots: Price + Equity
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Price with Trades", "Equity Curve")
    )

    # Price Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ), row=1, col=1)

    # Optional SMA
    if 'SMA' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA'],
            mode='lines',
            name='SMA',
            line=dict(color='blue')
        ), row=1, col=1)

    # Buy/Sell Markers
    if not trades.empty:
        # Check for the correct column names
        if "EntryTime" in trades.columns and "EntryPrice" in trades.columns:
            buy_trades = trades[
                trades["Size"] > 0
            ]  # Assuming 'Size' > 0 indicates a buy
            sell_trades = trades[
                trades["Size"] < 0
            ]  # Assuming 'Size' < 0 indicates a sell

            fig.add_trace(
                go.Scatter(
                    x=buy_trades["EntryTime"],
                    y=buy_trades["EntryPrice"],
                    mode="markers",
                    name="Buy",
                    marker=dict(symbol="triangle-up", color="green", size=10),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=sell_trades["EntryTime"],
                    y=sell_trades["EntryPrice"],
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", color="red", size=10),
                ),
                row=1,
                col=1,
            )

    # Equity Curve
    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity['Equity'],
        mode='lines',
        name='Equity',
        line=dict(color='green')
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title="Backtest Result",
        xaxis2_title="Time",
        yaxis_title="Price",
        yaxis2_title="Equity",
        template='plotly_white',
        showlegend=True,
        height=800
    )

    # --- Convert to base64 PNG ---
    buffer = BytesIO()
    pio.write_image(fig, buffer, format='png', width=1280, height=800)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return stats, img_base64
