# data/data_fetch.py

import tushare as ts
import pandas as pd
from project.config import TUSHARE_TOKEN

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

def get_stock_data(ts_code, start_date, end_date):
    """
    获取单只股票的日线行情
    """
    df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df = df.sort_values('trade_date').reset_index(drop=True)
    df = df[['trade_date', 'open', 'high', 'low', 'close', 'vol']]
    df.rename(columns={'vol': 'volume'}, inplace=True)
    return df

def get_multi_stock_data(stock_list, start_date, end_date):
    """
    获取多只股票的历史数据，并添加股票编码列
    """
    all_data = []
    for code in stock_list:
        df = get_stock_data(code, start_date, end_date)
        df['ts_code'] = code
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)
