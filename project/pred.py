import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from tensorflow.keras.models import load_model
import talib
from sklearn.preprocessing import MinMaxScaler
import joblib
from pandas.tseries.offsets import BDay
import pandas as pd

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统字体
plt.rcParams['axes.unicode_minus'] = False

# 获取数据（复用训练时的特征列）
def get_data(symbol):
    df = ts.pro_api().daily(ts_code=symbol, start_date='20110101', end_date='20250219')
    df.index = pd.to_datetime(df['trade_date'])
    # 确保数据框中包含 'volume' 列，如果列名不同，改为 'vol'
    df.rename(columns={'vol': 'volume'}, inplace=True)  # 如果是 'vol'，改为 'volume'
    print(df.columns)  # 调试时查看列名
    return df[['open', 'high', 'low', 'close', 'volume']]

# 技术指标计算（与训练时一致）
def add_indicators(df):
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], _, _ = talib.MACD(df['close'])
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    return df.dropna()

# 预测主函数
def predict_stock(symbol, look_back=180, predict_days=5):
    # 加载历史数据
    df = add_indicators(get_data(symbol))

    # 加载训练好的模型和归一化器
    model = load_model('stock_predictor_model.h5')
    scaler = joblib.load('scaler.save')  # 需在训练时保存

    # 准备输入数据，使用模型所需的特征列
    features = ['close', 'rsi', 'macd', 'ma5', 'ma20', 'volume']
    input_data = scaler.transform(df[features].iloc[-look_back:])

    # 多步预测
    predictions = []
    current_batch = input_data.reshape(1, look_back, input_data.shape[1])

    for _ in range(predict_days):
        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred[0])

        # 更新输入数据（假设只预测收盘价）
        new_features = np.zeros(input_data.shape[1])
        new_features[0] = pred  # 收盘价位置
        current_batch = np.append(current_batch[:, 1:, :],
                                  np.array([new_features]).reshape(1, 1, -1),
                                  axis=1)

    # 反归一化
    pred_prices = scaler.inverse_transform(
        np.concatenate([np.array(predictions).reshape(-1, 1),
                        np.zeros((len(predictions), input_data.shape[1] - 1))],
                       axis=1)
    )[:, 0]

    # 生成未来日期
    last_date = df.index[-1]
    future_dates = [last_date + BDay(i + 1) for i in range(predict_days)]

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(df['close'].iloc[-30:], label='Historical Price')
    plt.plot(future_dates, pred_prices, 'ro-', label='Predicted Price')
    plt.title(f'{symbol} Price Prediction', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return dict(zip(
        [d.strftime('%Y-%m-%d') for d in future_dates],
        np.round(pred_prices, 2)
    ))

# 执行预测
if __name__ == "__main__":
    predictions = predict_stock('601996.SH', predict_days=5)
    print("未来5个交易日预测价格：")
    for date, price in predictions.items():
        print(f"{date}: {price}")