# backend/fetchYfinance.py
import yfinance as yf
from fastapi import HTTPException
import pandas as pd


async def fetch_yfinance_data(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker '{ticker}'")
        data.columns = [col.capitalize() for col in data.columns]
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        return data[required_cols].sort_index()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")
