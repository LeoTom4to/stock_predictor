import numpy as np
import pandas as pd

def generate_classification_target(df, horizon=3):
    """
    根据未来 horizon 日收益率是否为正，生成二分类标签
    """
    df = df.copy()

    # 分组计算未来收益（每只股票单独处理）
    def compute_label(group):
        close_prices = group['close'].values
        future_returns = (np.roll(close_prices, -horizon) - close_prices) / close_prices
        group['label'] = (future_returns > 0).astype(int)
        group.iloc[-horizon:, group.columns.get_loc('label')] = np.nan  # 最后几天无法计算标签
        return group

    df = df.groupby('ts_code', group_keys=False).apply(compute_label)
    df = df.dropna(subset=['label'])  # 去除无法打标签的行
    df['label'] = df['label'].astype(int)
    return df
