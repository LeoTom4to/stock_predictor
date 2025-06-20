# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np


def plot_signals(close_prices, signals, title="Close Price with Buy Signals"):
    """
    Plot close prices with buy signals.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(close_prices, label='Close Price', linewidth=1.5)

    # 获取买入信号索引
    buy_indices = np.where(signals == 1)[0]

    # 修复：确保买入信号索引不超出close_prices的范围
    buy_indices = buy_indices[buy_indices < len(close_prices)]

    plt.scatter(buy_indices, close_prices[buy_indices], marker='^', color='g', label='Buy Signal')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_returns(strategy_returns, benchmark_returns=None, title="Cumulative Return Curve"):
    """
    Plot cumulative returns of strategy and benchmark.
    """
    strategy_curve = np.cumprod(1 + strategy_returns) - 1
    plt.figure(figsize=(12, 5))
    plt.plot(strategy_curve, label='Strategy Cumulative Return', linewidth=1.8)

    if benchmark_returns is not None:
        benchmark_curve = np.cumprod(1 + benchmark_returns) - 1
        plt.plot(benchmark_curve, label='Benchmark Cumulative Return', linestyle='--')

    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.show()
