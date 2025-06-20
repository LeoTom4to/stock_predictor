import numpy as np

def compute_strategy_metrics(returns):
    """
    计算策略评估指标：胜率、盈亏比、累计收益
    """
    returns = np.array(returns)
    total = len(returns)
    win = np.sum(returns > 0)
    loss = np.sum(returns < 0)

    win_rate = win / total if total > 0 else 0
    avg_win = np.mean(returns[returns > 0]) if win > 0 else 0
    avg_loss = np.mean(np.abs(returns[returns < 0])) if loss > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else np.inf
    cumulative_return = np.cumprod(1 + returns)[-1] - 1

    return {
        'Win Rate': round(win_rate, 3),             # 胜率
        'Profit Factor': round(profit_factor, 3),   # 盈亏比
        'Cumulative Return': round(cumulative_return, 3)  # 累计收益
    }
