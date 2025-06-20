import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_lstm_data_with_stockcode(df, look_back=60):
    """
    多股票数据预处理：添加股票编码作为特征 + 标准化 + 滑动窗口
    返回：X, y, stock_codes, scaler
    """
    df = df.copy()
    df['stock_code_id'] = LabelEncoder().fit_transform(df['ts_code'])

    features = df.drop(columns=['trade_date', 'label', 'ts_code']).values
    labels = df['label'].values
    stock_ids = df['stock_code_id'].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X, y, stock_codes = [], [], []
    for i in range(len(features_scaled) - look_back):
        X.append(features_scaled[i:i+look_back])
        y.append(labels[i+look_back])
        stock_codes.append(stock_ids[i+look_back])  # 提取预测目标那一行的股票id

    return np.array(X), np.array(y), np.array(stock_codes), scaler
