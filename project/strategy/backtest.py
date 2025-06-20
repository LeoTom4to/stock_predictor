# strategy/backtest.py

import numpy as np
import pandas as pd

def backtest_strategy(signals, close_prices):
    """
    简单回测策略：当天买入，次日收盘卖出，记录收益
    """
    signals = np.array(signals)
    close_prices = np.array(close_prices)

    returns = np.zeros(len(signals) - 1)
    for i in range(len(returns)):
        if signals[i] == 1:
            returns[i] = (close_prices[i + 1] - close_prices[i]) / close_prices[i]
        else:
            returns[i] = 0

    return returns
