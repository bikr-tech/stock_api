import base64
from fastapi import FastAPI, Query, HTTPException
# from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import os
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
import plotly.io as pio
from io import BytesIO
import numpy as np
from backend.backtest import run_backtest
from fastapi.staticfiles import StaticFiles
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Analysis API", description="A local API for stock technical analysis"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChartRequest(BaseModel):
    ticker: str
    period: str = "1y"
    interval: str = "1d"


def identify_patterns(df):
    patterns = []
    if (df["Close"].iloc[-3] < df["Close"].iloc[-2] < df["Close"].iloc[-1]) and (
        df["Volume"].iloc[-1] > df["Volume"].rolling(20).mean().iloc[-1]
    ):
        patterns.append(
            {
                "name": "Bullish Trend",
                "confidence": 0.75,
                "description": "Higher highs with increasing volume",
            }
        )
    return patterns


def calculate_support_resistance(df, lookback=20):
    support = df["Close"].rolling(lookback).min().iloc[-1]
    resistance = df["Close"].rolling(lookback).max().iloc[-1]
    return [
        {"type": "support", "price": round(float(support), 2), "strength": "medium"},
        {
            "type": "resistance",
            "price": round(float(resistance), 2),
            "strength": "medium",
        },
    ]


def calculate_fibonacci(df, levels=[0.236, 0.382, 0.5, 0.618, 0.786]):
    high, low = df["High"].max(), df["Low"].min()
    diff = high - low
    return {
        "high": round(float(high), 2),
        "low": round(float(low), 2),
        "levels": {
            f"{level*100:.1f}%": round(float(high - level * diff), 2)
            for level in levels
        },
    }


def generate_signals(df):
    signals = []
    last_close, sma_200, rsi = (
        df["Close"].iloc[-1],
        df["SMA_200"].iloc[-1],
        df["RSI"].iloc[-1],
    )
    if last_close > sma_200 and 50 < rsi < 70:
        signals.append(
            {
                "type": "buy",
                "confidence": 0.7,
                "entry": round(float(last_close), 2),
                "stop_loss": round(float(df["Close"].rolling(20).min().iloc[-1]), 2),
                "targets": [
                    round(float(last_close * 1.05), 2),
                    round(float(last_close * 1.1), 2),
                ],
            }
        )
    return signals

def determine_trend(df):
    if df["Close"].iloc[-1] > df["Close"].iloc[0] * 1.1:
        return "Up"
    elif df["Close"].iloc[-1] < df["Close"].iloc[0] * 0.9:
        return "Down"
    else:
        return "Sideways"   

def calculate_risk_reward(entry, stop_loss, target):    
    risk = entry - stop_loss
    reward = target - entry
    return round(reward / risk, 2) if risk != 0 else None 

def generate_trade_metrics(df, current_price, trend, patterns):
    confidence_score = 0

    # --- RSI-Based Confidence ---
    rsi_value = df["RSI"].iloc[-1]
    if 45 < rsi_value < 60:
        confidence_score += 2
    elif 60 <= rsi_value <= 70:
        confidence_score += 1
    elif rsi_value > 70 or rsi_value < 30:
        confidence_score -= 1

    # --- MACD-Based Confidence ---
    macd = df["MACD_12_26_9"].iloc[-1]
    macd_signal = df["MACDs_12_26_9"].iloc[-1]
    if macd > macd_signal:
        confidence_score += 2
    else:
        confidence_score -= 1

    # --- Trend-Based Confidence ---
    if trend == "Up":
        confidence_score += 2
    elif trend == "Down":
        confidence_score -= 1

    # --- Pattern-Based Confidence ---
    if patterns:
        confidence_score += min(len(patterns), 3)  # max +3 for multiple confirmations

    # Clamp score between 0-10
    confidence = f"{min(max(confidence_score + 5, 0), 10)}/10"

    # --- Dynamic Entry/Stop Loss/Take Profit ---
    entry_price = round(current_price * 0.99, 2)
    stop_loss = round(current_price * 0.96, 2)
    take_profit = round(current_price * 1.05, 2)

    risk = entry_price - stop_loss
    reward = take_profit - entry_price

    if risk > 0:
        risk_reward = round(reward / risk, 2)
        risk_reward_str = f"1:{risk_reward}"
    else:
        risk_reward_str = "N/A"

    return {
        "confidence": confidence,
        "risk_reward": risk_reward_str,
        "entry": entry_price,
        "stop_loss": stop_loss,
        "take_profit": [
            round(current_price * 1.02, 2),
            round(current_price * 1.05, 2),
        ],
        "range": f"${round(current_price * 0.98, 2)}-${round(current_price, 2)}",
    }


@app.get("/", description="Root endpoint with a welcome message")
def read_root():
    return {"message": "Welcome to your local stock API!"}


@app.get("/stock", description="Fetch raw stock historical data")
def get_stock_data(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    period: str = Query("1mo", description="Time period (e.g., 1mo, 3mo, 1y)"),
    interval: str = Query("1d", description="Interval (e.g., 1d, 1h, 5m)"),
):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise HTTPException(
                status_code=404, detail=f"No data for ticker '{ticker}'"
            )
        return hist.reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")


@app.get("/rsi", description="Calculate RSI for a stock")
def get_rsi(
    ticker: str = Query(..., description="Stock ticker symbol"),
    period: str = Query("1mo", description="Time period"),
    interval: str = Query("1d", description="Interval"),
    rsi_period: int = Query(14, description="RSI lookback period", ge=2),
):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise HTTPException(
                status_code=404, detail=f"No data for ticker '{ticker}'"
            )
        hist["RSI"] = ta.rsi(hist["Close"], length=rsi_period)
        return hist[["RSI"]].dropna().reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating RSI: {str(e)}")


@app.get("/sma", description="Calculate SMA for a stock")
def get_sma(
    ticker: str = Query(..., description="Stock ticker symbol"),
    period: str = Query("3mo", description="Time period"),
    interval: str = Query("1d", description="Interval"),
    sma_period: int = Query(20, description="SMA lookback period", ge=2),
):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise HTTPException(
                status_code=404, detail=f"No data for ticker '{ticker}'"
            )
        hist["SMA"] = ta.sma(hist["Close"], length=sma_period)
        return hist[["SMA"]].dropna().reset_index().to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating SMA: {str(e)}")


@app.get("/macd", description="Calculate MACD for a stock")
def get_macd(
    ticker: str = Query(..., description="Stock ticker symbol"),
    period: str = Query("3mo", description="Time period"),
    interval: str = Query("1d", description="Interval"),
):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise HTTPException(
                status_code=404, detail=f"No data for ticker '{ticker}'"
            )
        macd = ta.macd(hist["Close"])
        result = hist.copy()
        result["MACD"] = macd["MACD_12_26_9"]
        result["MACD_signal"] = macd["MACDs_12_26_9"]
        result["MACD_hist"] = macd["MACDh_12_26_9"]
        return (
            result[["MACD", "MACD_signal", "MACD_hist"]]
            .dropna()
            .reset_index()
            .to_dict(orient="records")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating MACD: {str(e)}")


@app.get("/bbands", description="Calculate Bollinger Bands for a stock")
def get_bbands(
    ticker: str = Query(..., description="Stock ticker symbol"),
    period: str = Query("3mo", description="Time period"),
    interval: str = Query("1d", description="Interval"),
):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise HTTPException(
                status_code=404, detail=f"No data for ticker '{ticker}'"
            )
        bb = ta.bbands(hist["Close"])
        result = hist.copy()
        result["BBL"] = bb["BBL_20_2.0"]
        result["BBM"] = bb["BBM_20_2.0"]
        result["BBU"] = bb["BBU_20_2.0"]
        return (
            result[["BBL", "BBM", "BBU"]]
            .dropna()
            .reset_index()
            .to_dict(orient="records")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating Bollinger Bands: {str(e)}"
        )


@app.get(
    "/chart",
    response_class=JSONResponse,
    description="Generate a technical analysis chart",
)
def get_analysis_chart(
    ticker: str = Query(..., description="Stock ticker symbol"),
    period: str = Query("1y", description="Time period"),
    interval: str = Query("1d", description="Interval"),
    rsi_period: int = Query(14, description="RSI lookback period", ge=2),
    bb_period: int = Query(20, description="Bollinger Bands period", ge=2),
):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"No data for ticker '{ticker}'"
            )

        df["RSI"] = ta.rsi(df["Close"], length=rsi_period)
        df["SMA_200"] = ta.sma(df["Close"], length=200)
        macd = ta.macd(df["Close"])
        bb = ta.bbands(df["Close"], length=bb_period)
        df = df.join([macd, bb]).dropna()

        high, low = df["High"].max(), df["Low"].min()
        diff = high - low
        fib_levels = {
            high - level * diff: f"{level*100:.1f}%"
            for level in [0.236, 0.382, 0.5, 0.618, 0.786]
        }

        patterns = identify_patterns(df)
        sr_levels = calculate_support_resistance(df)
        fib_results = calculate_fibonacci(df, levels=fib_levels)
        signals = generate_signals(df)

        # Determine trend
        trend = determine_trend(df)

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.15, 0.2, 0.15],
            vertical_spacing=0.02,
            specs=[
                [{"type": "candlestick"}],
                [{"type": "bar"}],
                [{"type": "scatter"}],
                [{"type": "scatter"}],
            ],
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
        for col, color in [
            ("BBU_20_2.0", "rgba(255,0,0,0.5)"),
            ("BBM_20_2.0", "rgba(0,255,0,0.5)"),
            ("BBL_20_2.0", "rgba(255,0,0,0.5)"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    line=dict(color=color),
                    name=col.split("_")[0],
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color="rgba(100,149,237,0.6)",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], line=dict(color="orange"), name="RSI"),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        for col, color, name in [
            ("MACD_12_26_9", "blue", "MACD"),
            ("MACDs_12_26_9", "red", "Signal"),
        ]:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], line=dict(color=color), name=name),
                row=4,
                col=1,
            )
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACDh_12_26_9"],
                marker_color=np.where(df["MACDh_12_26_9"] < 0, "red", "green"),
                name="Histogram",
            ),
            row=4,
            col=1,
        )
        for price, level in fib_levels.items():
            fig.add_hline(
                y=price,
                line_dash="dot",
                line_color="purple",
                annotation_text=level,
                annotation_position="top right",
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_200"],
                line=dict(color="yellow", width=2),
                name="200 SMA",
            ),
            row=1,
            col=1,
        )

        fig.update_layout(
            height=1200,
            width=1800,
            title=f"{ticker.upper()} Technical Analysis | {period} | {interval}",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)

        img_bytes = BytesIO()
        fig.write_image(img_bytes, format="png", engine="kaleido")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        # img_bytes.seek(0)
        # return StreamingResponse(
        #     img_bytes,
        #     media_type="image/png",
        #     headers={
        #         "Content-Disposition": f"attachment; filename={ticker}_analysis.png"
        #     },
        # )
        current_price = float(df["Close"].iloc[-1])
        trade_metrics = generate_trade_metrics(df, current_price, trend, patterns)
        response = {
            "metadata": {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "last_updated": df.index[-1].strftime("%Y-%m-%d"),
            },
            "price_data": {
                "current": float(df["Close"].iloc[-1]),
                "high": float(df["High"].max()),
                "low": float(df["Low"].min()),
                "volume": float(df["Volume"].mean()),
            },
            "indicators": {
                "rsi": float(df["RSI"].iloc[-1]),
                "macd": {
                    "value": float(df["MACD_12_26_9"].iloc[-1]),
                    "signal": float(df["MACDs_12_26_9"].iloc[-1]),
                    "histogram": float(df["MACDh_12_26_9"].iloc[-1]),
                },
                "bollinger": {
                    "upper": float(df["BBU_20_2.0"].iloc[-1]),
                    "middle": float(df["BBM_20_2.0"].iloc[-1]),
                    "lower": float(df["BBL_20_2.0"].iloc[-1]),
                },
            },
            "analysis": {
                "trend": trend,
                "patterns": patterns,
                "support_resistance": sr_levels,
                "fibonacci": fib_results,
                "signals": signals,
                "trade_metrics": trade_metrics,
            },
            "image": f"data:image/png;base64,{img_base64}",
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")


@app.get("/advanced_analysis", description="Perform advanced technical analysis")
def advanced_technical_analysis(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL, MSFT)"),
    period: str = Query("1y", description="Time period (e.g., 1mo, 3mo, 1y)"),
    interval: str = Query("1d", description="Interval (e.g., 1d, 1wk, 1h)"),
    rsi_period: int = Query(14, description="RSI lookback period", ge=2),
    bb_period: int = Query(20, description="Bollinger Bands period", ge=2),
    fib_levels: Optional[str] = Query(
        "0.236,0.382,0.5,0.618,0.786", description="Comma-separated Fibonacci levels"
    ),
):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"No data for ticker '{ticker}'"
            )

        # Add technical indicators
        fib_levels = [float(level) for level in fib_levels.split(",")]
        df["RSI"] = ta.rsi(df["Close"], length=rsi_period)
        df["SMA_200"] = ta.sma(df["Close"], length=200)
        macd = ta.macd(df["Close"])
        bb = ta.bbands(df["Close"], length=bb_period)
        df = df.join([macd, bb]).dropna()

        patterns = identify_patterns(df)
        sr_levels = calculate_support_resistance(df)
        fib_results = calculate_fibonacci(df, levels=fib_levels)
        signals = generate_signals(df)

        # Determine trend
        trend = determine_trend(df)

        return {
            "metadata": {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "last_updated": df.index[-1].strftime("%Y-%m-%d"),
            },
            "price_data": {
                "current": float(df["Close"].iloc[-1]),
                "high": float(df["High"].max()),
                "low": float(df["Low"].min()),
                "volume": float(df["Volume"].mean()),
            },
            "indicators": {
                "rsi": float(df["RSI"].iloc[-1]),
                "macd": {
                    "value": float(df["MACD_12_26_9"].iloc[-1]),
                    "signal": float(df["MACDs_12_26_9"].iloc[-1]),
                    "histogram": float(df["MACDh_12_26_9"].iloc[-1]),
                },
                "bollinger": {
                    "upper": float(df["BBU_20_2.0"].iloc[-1]),
                    "middle": float(df["BBM_20_2.0"].iloc[-1]),
                    "lower": float(df["BBL_20_2.0"].iloc[-1]),
                },
            },
            "analysis": {
                "patterns": patterns,
                "support_resistance": sr_levels,
                "fibonacci": fib_results,
                "signals": signals,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in advanced analysis: {str(e)}"
        )


@app.post("/analyze", description="Analyze stock and return chart and analysis in JSON")
async def analyze_stock(req: ChartRequest):
    try:
        # Fetch historical data
        stock = yf.Ticker(req.ticker)
        df = stock.history(period=req.period, interval=req.interval)
        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"No data for ticker '{req.ticker}'"
            )

        df.dropna(inplace=True)

        # Add technical indicators
        df["RSI_14"] = ta.rsi(df["Close"], length=14)
        df["EMA_200"] = ta.ema(df["Close"], length=200)
        if "Volume" in df.columns:
            df["vol_avg20"] = df["Volume"].rolling(window=20).mean()
        else:
            df["vol_avg20"] = None

        # Determine trend
        trend = determine_trend(df)

        # Create Plotly chart
        fig = go.Figure()

        # Candlestick plot for price
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Candles",
            )
        )

        # EMA 200 plot
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA_200"],
                mode="lines",
                name="EMA 200",
                line=dict(color="blue"),
            )
        )

        # RSI 14 plot on a separate y-axis
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI_14"],
                mode="lines",
                name="RSI (14)",
                yaxis="y2",
                line=dict(color="orange"),
            )
        )

        # Adding pattern detection lines (e.g., reversal signals)
        reversal_signals = df[df["RSI_14"] < 30]  # Example: Define reversal signals based on RSI
        for idx in reversal_signals.index:
            fig.add_shape(
                type="line",
                x0=idx,
                y0=df.loc[idx, "Low"] * 0.95,
                x1=idx,
                y1=df.loc[idx, "Low"],
                line=dict(color="purple", width=2),
            )

        # Update layout with multiple y-axes and better spacing
        fig.update_layout(
            title=f"{req.ticker} Technical Analysis | {req.period} | {req.interval}",
            xaxis_rangeslider_visible=False,
            yaxis_title="Price",
            yaxis2=dict(
                title="RSI",
                overlaying="y",
                side="right",
                range=[0, 100],  # Limit RSI axis between 0 and 100
            ),
            template="plotly_dark",
            height=800,
            width=1200,
        )

        # Generate image and encode as base64
        img_bytes = BytesIO()
        img = pio.write_image(fig, img_bytes, format="png", engine="kaleido")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        # Generate analysis
        current_price = float(df["Close"].iloc[-1])
        analysis = {
            "trend": trend,
            "pattern_detected": bool(len(reversal_signals) > 0),
            "bull_flag": {
                "range": f"${round(current_price * 0.98, 2)}–${round(current_price, 2)}",
                "entry": round(current_price * 0.99, 2),
                "stop_loss": round(current_price * 0.96, 2),
                "take_profit": [
                    round(current_price * 1.02, 2),
                    round(current_price * 1.05, 2),
                ],
                "confidence": "8/10",
                "risk_reward": "1:2.7",
            },
        }

        # Return JSON response
        # return {"image": f"data:image/png;base64,{img_base64}", "analysis": analysis}
        return img

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing stock: {str(e)}")


@app.get("/backtest")
async def backtest_endpoint(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)"),
    start_date: str = Query("2018-01-01", description="Start date (YYYY-MM-DD)"),
    end_date: str = Query("2023-12-31", description="End date (YYYY-MM-DD)"),
    cash: float = Query(100000, description="Initial cash for backtest", ge=1000),
    commission: float = Query(
        0.001, description="Commission per trade (e.g., 0.001 = 0.1%)", ge=0, le=0.1
    ),
):
    try:
        # logger.debug(f"Starting backtest endpoint for {ticker}")
        stats, img_base64 = await run_backtest(
            ticker, start_date, end_date, cash, commission
        )
        # Extract key metrics
        sharpe_ratio = stats.get("Sharpe Ratio", 0)
        win_rate = stats.get("Win Rate [%]", 0)
        total_return = stats.get("Return [%]", 0)
        max_drawdown = stats.get("Max. Drawdown [%]", 0)

        # Generate recommendations based on metrics
        recommendation = "Hold"
        recommendation_reason = []

        # Buy recommendation: High Sharpe Ratio and positive return
        if sharpe_ratio > 1.5 and total_return > 0:
            recommendation = "Buy"
            recommendation_reason.append(
                f"High Sharpe Ratio ({sharpe_ratio:.2f}) and positive return ({total_return:.2f}%)."
            )

        # Sell recommendation: Low Sharpe Ratio or high drawdown
        elif sharpe_ratio < 0.5 or max_drawdown < -20:
            recommendation = "Sell"
            recommendation_reason.append(
                f"Low Sharpe Ratio ({sharpe_ratio:.2f}) or high drawdown ({max_drawdown:.2f}%)."
            )

        # Hold recommendation: Moderate performance
        else:
            recommendation_reason.append(
                f"Moderate Sharpe Ratio ({sharpe_ratio:.2f}) and return ({total_return:.2f}%)."
            )

        # Prepare the response
        result = {
            "metadata": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "cash": cash,
                "commission": commission,
                "duration_days": 1824,
                "backtest_timestamp": datetime.utcnow().isoformat(),
            },
            "performance": {
                "return": {
                    "total_percent": round(stats["Return [%]"], 2),
                    "annualized_percent": round(stats["Return (Ann.) [%]"], 2),
                    "buy_hold_percent": round(stats["Buy & Hold Return [%]"], 2),
                    "cagr_percent": round(stats["CAGR [%]"], 2),
                },
                "risk": {
                    "max_drawdown_percent": round(stats["Max. Drawdown [%]"], 2),
                    "avg_drawdown_percent": round(stats["Avg. Drawdown [%]"], 2),
                    "volatility_annual_percent": round(
                        stats["Volatility (Ann.) [%]"], 2
                    ),
                },
                "ratios": {
                    "sharpe": round(stats["Sharpe Ratio"], 2),
                    "sortino": round(stats["Sortino Ratio"], 2),
                    "calmar": round(stats["Calmar Ratio"], 2),
                    "profit_factor": round(stats["Profit Factor"], 2),
                },
                "trades": {
                    "total_count": int(stats["# Trades"]),
                    "win_rate_percent": round(stats["Win Rate [%]"], 2),
                    "best_trade_percent": round(stats["Best Trade [%]"], 2),
                    "worst_trade_percent": round(stats["Worst Trade [%]"], 2),
                    "avg_trade_percent": round(stats["Avg. Trade [%]"], 2),
                    "expectancy_percent": round(stats["Expectancy [%]"], 2),
                },
                "equity": {
                    "final": round(stats["Equity Final [$]"], 2),
                    "peak": round(stats["Equity Peak [$]"], 2),
                    "commissions": round(stats["Commissions [$]"], 2),
                },
                "durations": {
                    "exposure_time_percent": round(stats["Exposure Time [%]"], 2),
                    "max_drawdown_days": 1027,
                    "avg_drawdown_days": 74,
                    "max_trade_days": 72,
                    "avg_trade_days": 11,
                },
            },
            "additional_metrics": {
                "alpha_percent": round(stats["Alpha [%]"], 2),
                "beta": round(stats["Beta"], 2),
                "system_quality_number": round(stats["SQN"], 2),
            },
            "recommendation": {
                "action": recommendation,
                "reason": recommendation_reason,
            },
            "image": f"data:image/png;base64,{img_base64}",
        }

        logger.debug("Backtest endpoint completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in backtest endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")
