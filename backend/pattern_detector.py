# backend/pattern_detector.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
import io
import time
import logging

from backend.fetchYfinance import fetch_yfinance_data


# ✅ Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PatternDetector:
    def __init__(self, ticker: str, period: str = "1y", interval: str = "1d"):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = None
        self.patterns = {}
        self.signals = None
        self._download_data()

    async def _download_data(self, retries=3, backoff_factor=2):
        for attempt in range(retries):
            try:
                end_date = pd.Timestamp.today()
                start_date = end_date - pd.Timedelta(self.period)
                data = await fetch_yfinance_data(self.ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                if data is None or data.empty:
                    raise ValueError("Empty data returned from yfinance")

                logging.debug(f"Raw data columns: {data.columns}")

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[-1] for col in data.columns]
                    logging.debug(f"Flattened MultiIndex columns: {data.columns}")

                column_map = {
                    "open": "Open",
                    "OPEN": "Open",
                    "Open": "Open",
                    "high": "High",
                    "HIGH": "High",
                    "High": "High",
                    "low": "Low",
                    "LOW": "Low",
                    "Low": "Low",
                    "close": "Close",
                    "CLOSE": "Close",
                    "Close": "Close",
                    "volume": "Volume",
                    "VOLUME": "Volume",
                    "Volume": "Volume",
                }
                available_columns = [col for col in data.columns]
                normalized_columns = {}
                for col in available_columns:
                    for key, value in column_map.items():
                        if col.lower() == key.lower():
                            normalized_columns[col] = value
                            break
                    else:
                        normalized_columns[col] = col

                data = data.rename(columns=normalized_columns)
                logging.debug(f"Normalized columns: {data.columns}")

                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(
                        f"Missing required columns: {missing_cols}. Available columns: {data.columns}"
                    )

                data = data[required_cols].copy()

                for col in ["Open", "High", "Low", "Close"]:
                    data[col] = pd.to_numeric(data[col], errors="coerce")
                data["Volume"] = pd.to_numeric(
                    data["Volume"], errors="coerce", downcast="integer"
                )

                self.data = data.dropna()
                self.signals = pd.DataFrame(index=self.data.index)
                logging.info(
                    f"Successfully downloaded data for {self.ticker}: {len(self.data)} rows"
                )
                logging.debug(f"Processed data columns: {self.data.columns}")
                logging.debug(f"Data types: {self.data.dtypes}")
                return

            except Exception as e:
                logging.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"Failed to download data for {self.ticker} after {retries} attempts"
                    )
                time.sleep(backoff_factor * (2**attempt))

    def _safe_indicator(self, func, *args, **kwargs):
        try:
            logging.debug(
                f"Calculating {func.__name__} with args: {args}, kwargs: {kwargs}"
            )
            for arg in args:
                if isinstance(arg, pd.Series):
                    logging.debug(
                        f"Input series dtype: {arg.dtype}, length: {len(arg)}, non-null: {arg.notna().sum()}"
                    )
                    if not np.issubdtype(arg.dtype, np.number):
                        logging.warning(
                            f"Non-numeric input for {func.__name__}: {arg.dtype}"
                        )
            result = func(*args, **kwargs)
            if result is None or (
                isinstance(result, (pd.Series, pd.DataFrame))
                and len(result) != len(self.data)
            ):
                logging.warning(
                    f"Indicator {func.__name__} returned invalid length or None"
                )
                return (
                    pd.Series(np.nan, index=self.data.index)
                    if not isinstance(result, pd.DataFrame)
                    else pd.DataFrame(
                        np.nan, index=self.data.index, columns=result.columns
                    )
                )
            return result
        except Exception as e:
            logging.error(f"Error calculating indicator {func.__name__}: {str(e)}")
            return pd.Series(np.nan, index=self.data.index)

    def calculate_indicators(self):
        if self.data is None or self.data.empty:
            raise ValueError("No data available for indicator calculation")

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {self.data.columns}")

        for col in ["Open", "High", "Low", "Close"]:
            if not np.issubdtype(self.data[col].dtype, np.number):
                raise ValueError(f"Column {col} is not numeric: {self.data[col].dtype}")

        logging.debug(f"Data shape before indicators: {self.data.shape}")
        logging.debug(f"Data columns before indicators: {self.data.columns}")
        logging.debug(f"Data types: {self.data.dtypes}")

        self.data["SMA_50"] = self._safe_indicator(
            ta.sma, self.data["Close"], length=50
        )
        self.data["SMA_200"] = self._safe_indicator(
            ta.sma, self.data["Close"], length=200
        )
        self.data["RSI_14"] = self._safe_indicator(
            ta.rsi, self.data["Close"], length=14
        )

        macd = self._safe_indicator(
            ta.macd, self.data["Close"], fast=12, slow=26, signal=9
        )
        if isinstance(macd, pd.DataFrame) and len(macd) == len(self.data):
            macd.columns = [str(col) for col in macd.columns]
            self.data[macd.columns] = macd

        adx = self._safe_indicator(
            ta.adx, self.data["High"], self.data["Low"], self.data["Close"], length=14
        )
        if isinstance(adx, pd.DataFrame) and len(adx) == len(self.data):
            adx.columns = [str(col) for col in adx.columns]
            self.data[adx.columns] = adx

        self.data["ATR_14"] = self._safe_indicator(
            ta.atr, self.data["High"], self.data["Low"], self.data["Close"], length=14
        )

        indicator_cols = ["SMA_50", "SMA_200", "RSI_14", "ATR_14"]
        for col in self.data.columns:
            if isinstance(col, str) and (
                col.startswith("MACD_") or col.startswith("ADX_")
            ):
                indicator_cols.append(col)

        logging.debug(f"Indicator columns: {indicator_cols}")
        self.data[indicator_cols] = self.data[indicator_cols].ffill()
        self.data.dropna(
            subset=["Open", "High", "Low", "Close", "Volume"], inplace=True
        )
        self.signals = self.signals.reindex(self.data.index)
        logging.info(f"Indicators calculated, data length: {len(self.data)}")

    def detect_trend(self):
        required_cols = ["SMA_50", "SMA_200"]
        if not all(col in self.data for col in required_cols):
            self.calculate_indicators()

        self.signals["Trend"] = np.select(
            [
                self.data["SMA_50"] > self.data["SMA_200"],
                self.data["SMA_50"] < self.data["SMA_200"],
            ],
            ["Uptrend", "Downtrend"],
            default="Neutral",
        )

        adx_cols = ["ADX_ADX_14", "ADX_DMP_14", "ADX_DMN_14"]
        if all(col in self.data for col in adx_cols):
            strong_trend = self.data["ADX_ADX_14"] > 25
            self.signals["Trend_Strength"] = np.select(
                [
                    strong_trend & (self.data["ADX_DMP_14"] > self.data["ADX_DMN_14"]),
                    strong_trend & (self.data["ADX_DMN_14"] > self.data["ADX_DMP_14"]),
                ],
                ["Strong Uptrend", "Strong Downtrend"],
                default="Weak/No Trend",
            )
        else:
            self.signals["Trend_Strength"] = "Weak/No Trend"
        logging.info("Trend detection completed")

    def plot_results(self) -> io.BytesIO:
        if self.data is None or self.signals is None:
            raise ValueError("No data or signals available for plotting")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        ax1.plot(self.data.index, self.data["Close"], label="Close Price", color="blue")
        ax1.plot(self.data.index, self.data["SMA_50"], label="SMA 50", color="orange")
        ax1.plot(self.data.index, self.data["SMA_200"], label="SMA 200", color="red")
        ax1.set_title(f"{self.ticker} Price and Moving Averages")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)

        if "RSI_14" in self.data:
            ax2.plot(
                self.data.index, self.data["RSI_14"], label="RSI 14", color="purple"
            )
            ax2.axhline(70, color="red", linestyle="--", alpha=0.5)
            ax2.axhline(30, color="green", linestyle="--", alpha=0.5)
            ax2.set_title("RSI")
            ax2.set_ylabel("RSI")
            ax2.legend()
            ax2.grid(True)

        plt.xlabel("Date")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        return buf

    def run_analysis(self):
        try:
            if self.data is None or self.data.empty:
                self._download_data()
            self.calculate_indicators()
            self.detect_trend()
            return {
                "status": "success",
                "data": self.data.to_dict(orient="index"),
                "signals": self.signals.to_dict(orient="index"),
            }
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            raise
