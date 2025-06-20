# data/feature_engineering.py (Enhanced Feature Engineering)

import talib
import numpy as np
import pandas as pd

def add_technical_indicators(df):
    """
    添加常用技术指标及衍生特征
    Add common technical indicators and derived features
    """
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'])

    # 衍生特征 Derived features
    df['pct_chg_1d'] = df['close'].pct_change()  # 一日涨跌幅
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']  # 收盘开盘比
    df['high_close_ratio'] = (df['high'] - df['close']) / df['close']  # 高收比
    df['low_close_ratio'] = (df['close'] - df['low']) / df['close']  # 低收比
    df['volatility_3d'] = df['close'].rolling(3).std()  # 3日波动率
    df['volatility_7d'] = df['close'].rolling(7).std()  # 7日波动率
    df['volume_change'] = df['volume'].pct_change()  # 成交量变动

    df = df.fillna(0)
    return df