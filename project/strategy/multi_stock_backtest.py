# strategy/multi_stock_backtest.py

from project.strategy.backtest import backtest_strategy
from project.strategy.metrics import compute_strategy_metrics

def backtest_multiple_stocks(df, signals_col='signal'):
    """
    对多只股票进行回测，返回每只股票的策略评估指标
    """
    results = {}
    grouped = df.groupby('ts_code')
    for code, group in grouped:
        close = group['close'].values
        signals = group[signals_col].values
        returns = backtest_strategy(signals, close)
        metrics = compute_strategy_metrics(returns)
        results[code] = metrics

    return results
