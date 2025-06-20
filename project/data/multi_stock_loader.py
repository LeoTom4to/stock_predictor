# data/multi_stock_loader.py

import pandas as pd
from project.data.data_fetch import get_stock_data
from project.data.feature_engineering import add_technical_indicators
from project.data.target_generator import generate_classification_target

def load_multiple_stocks(stock_list, start_date, end_date, horizon=3):
    """
    加载多只股票数据，添加技术指标和目标标签。
    返回合并后的 DataFrame，含 'ts_code' 字段标识股票来源。
    """
    all_data = []

    for code in stock_list:
        try:
            df = get_stock_data(code, start_date, end_date)
            if df.empty or len(df) < horizon + 5:
                print(f"⚠️ 数据不足，跳过: {code}")
                continue
            df = add_technical_indicators(df)
            df = generate_classification_target(df, horizon=horizon)
            df['ts_code'] = code
            all_data.append(df)
        except Exception as e:
            print(f" 加载失败 {code}: {e}")

    if not all_data:
        raise ValueError("所有股票数据加载失败或为空。请检查代码或网络连接。")

    return pd.concat(all_data, ignore_index=True)
