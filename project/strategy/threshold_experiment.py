# strategy/threshold_experiment.py

import numpy as np
from project.strategy.signal_generator import generate_signals
from project.strategy.backtest import backtest_strategy
from project.strategy.metrics import compute_strategy_metrics
import matplotlib.pyplot as plt

def run_threshold_experiment(probs, close_prices, thresholds=np.arange(0.4, 0.75, 0.05)):
    """
    批量运行多个阈值下的策略表现
    """
    win_rates = []
    profits = []
    sharpes = []

    for th in thresholds:
        signals = generate_signals(probs, threshold=th)
        returns = backtest_strategy(signals, close_prices)
        metrics = compute_strategy_metrics(returns)
        win_rates.append(metrics['Win Rate'])  # 胜率
        profits.append(metrics['Cumulative Return'])  # 累计收益
        sharpes.append(metrics['Profit Factor'])  # 盈亏比

        print(f"Threshold: {round(th, 2)} => Win Rate: {metrics['Win Rate']}, "
              f"Return: {metrics['Cumulative Return']}, P/L Ratio: {metrics['Profit Factor']}")

    return thresholds, win_rates, profits, sharpes


def plot_threshold_performance(thresholds, win_rates, profits, sharpes):
    """
    可视化不同阈值下的策略表现趋势
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(thresholds, win_rates, marker='o')
    plt.title("Win Rate vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Win Rate")

    plt.subplot(1, 3, 2)
    plt.plot(thresholds, profits, marker='o', color='g')
    plt.title("Cumulative Return vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Cumulative Return")

    plt.subplot(1, 3, 3)
    plt.plot(thresholds, sharpes, marker='o', color='r')
    plt.title("Profit Factor vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Profit Factor")

    plt.tight_layout()
    plt.show()
