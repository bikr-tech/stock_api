# backend/strategy.py
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_support_resistance(data: pd.DataFrame, window=10):
    support_levels = []
    resistance_levels = []
    for i in range(window, len(data) - window):
        is_local_low = True
        for j in range(i - window, i + window + 1):
            if data.iloc[j]["Low"] < data.iloc[i]["Low"]:
                is_local_low = False
                break
        if is_local_low:
            support_levels.append(data.iloc[i]["Low"])
        is_local_high = True
        for j in range(i - window, i + window + 1):
            if data.iloc[j]["High"] > data.iloc[i]["High"]:
                is_local_high = False
                break
        if is_local_high:
            resistance_levels.append(data.iloc[i]["High"])
    support_levels = sorted(list(set(support_levels)))
    resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
    return support_levels, resistance_levels


class AdvancedTrendStrategyWithSR(Strategy):
    sma_long_period = 20  # Further reduced
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    rsi_entry_level = 40  # Lowered
    rsi_exit_level = 30  # Lowered
    atr_period = 14
    atr_multiplier = 1.5
    risk_reward_ratio = 2.0
    atr_min_threshold = 0
    sr_lookback = 20

    def init(self):
        self._macd_df = None

        def get_macd_df():
            if self._macd_df is None:
                self._macd_df = ta.macd(
                    pd.Series(self.data.Close),
                    fast=self.macd_fast,
                    slow=self.macd_slow,
                    signal=self.macd_signal,
                )
            return self._macd_df

        self.sma_long = self.I(
            ta.sma, pd.Series(self.data.Close), length=self.sma_long_period
        )
        self.macd = self.I(
            lambda: get_macd_df()[
                f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
            ]
        )
        self.macd_signal = self.I(
            lambda: get_macd_df()[
                f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
            ]
        )
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=self.rsi_period)
        self.atr = self.I(
            ta.atr,
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
            pd.Series(self.data.Close),
            length=self.atr_period,
        )
        support_levels, resistance_levels = find_support_resistance(
            self.data.df, window=self.sr_lookback
        )
        AdvancedTrendStrategyWithSR.support_levels = support_levels
        AdvancedTrendStrategyWithSR.resistance_levels = resistance_levels
        self.long_entry_price = None
        self.short_entry_price = None

    def next(self):
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]
        in_uptrend = current_price > self.sma_long[-1]
        in_downtrend = current_price < self.sma_long[-1]
        lower_resistances = [
            r
            for r in AdvancedTrendStrategyWithSR.resistance_levels
            if r < current_price
        ]
        upper_supports = [
            s for s in AdvancedTrendStrategyWithSR.support_levels if s > current_price
        ]
        nearest_resistance = min(lower_resistances) if lower_resistances else None
        nearest_support = max(upper_supports) if upper_supports else None

        if self.position:
            if self.position.is_long:
                long_stop_loss = (
                    self.long_entry_price - self.atr_multiplier * current_atr
                )
                long_take_profit = self.long_entry_price + self.risk_reward_ratio * (
                    self.long_entry_price - long_stop_loss
                )
                self.position.sl = long_stop_loss
                self.position.tp = long_take_profit
            elif self.position.is_short:
                short_stop_loss = (
                    self.short_entry_price + self.atr_multiplier * current_atr
                )
                short_take_profit = self.short_entry_price - self.risk_reward_ratio * (
                    short_stop_loss - self.short_entry_price
                )
                self.position.sl = short_stop_loss
                self.position.tp = short_take_profit
        else:
            long_stop_loss = current_price - self.atr_multiplier * current_atr
            long_take_profit = current_price + self.risk_reward_ratio * (
                current_price - long_stop_loss
            )
            short_stop_loss = current_price + self.atr_multiplier * current_atr
            short_take_profit = current_price - self.risk_reward_ratio * (
                short_stop_loss - current_price
            )

        # Removed S/R filter for simplicity
        if (
            not self.position
            and in_uptrend
            and crossover(self.macd, self.macd_signal)
            and self.rsi[-1] > self.rsi_entry_level
        ):
            self.buy(sl=long_stop_loss, tp=long_take_profit)
            self.long_entry_price = current_price
        elif (
            not self.position
            and in_downtrend
            and crossover(self.macd_signal, self.macd)
            and self.rsi[-1] < self.rsi_entry_level
        ):
            self.sell(sl=short_stop_loss, tp=short_take_profit)
            self.short_entry_price = current_price
        if self.position.is_long and self.rsi[-1] < self.rsi_exit_level:
            self.position.close()
        elif self.position.is_short and self.rsi[-1] > self.rsi_exit_level:
            self.position.close()
