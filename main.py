from fastapi import FastAPI, Query
import yfinance as yf
import pandas_ta as ta

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to your local stock API!"}


@app.get("/stock")
def get_stock_data(
    ticker: str = Query(...), period: str = Query("1mo"), interval: str = Query("1d")
):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist.reset_index().to_dict(orient="records")


@app.get("/rsi")
def get_rsi(
    ticker: str = Query(...),
    period: str = Query("1mo"),
    interval: str = Query("1d"),
    rsi_period: int = Query(14),
):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    hist["RSI"] = ta.rsi(hist["Close"], length=rsi_period)
    return hist[["RSI"]].dropna().reset_index().to_dict(orient="records")


@app.get("/sma")
def get_sma(
    ticker: str = Query(...),
    period: str = Query("3mo"),
    interval: str = Query("1d"),
    sma_period: int = Query(20),
):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    hist["SMA"] = ta.sma(hist["Close"], length=sma_period)
    return hist[["SMA"]].dropna().reset_index().to_dict(orient="records")


@app.get("/macd")
def get_macd(
    ticker: str = Query(...), period: str = Query("3mo"), interval: str = Query("1d")
):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
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


@app.get("/bbands")
def get_bbands(
    ticker: str = Query(...), period: str = Query("3mo"), interval: str = Query("1d")
):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    bb = ta.bbands(hist["Close"])
    result = hist.copy()
    result["BBL"] = bb["BBL_20_2.0"]
    result["BBM"] = bb["BBM_20_2.0"]
    result["BBU"] = bb["BBU_20_2.0"]
    return (
        result[["BBL", "BBM", "BBU"]].dropna().reset_index().to_dict(orient="records")
    )
