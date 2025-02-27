import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers.legacy import RMSprop  # 使用 legacy RMSprop 优化器
from tensorflow.keras.callbacks import EarlyStopping
import talib
import matplotlib.pyplot as plt
from matplotlib import rcParams  # 导入rcParams，用于配置字体
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置Tushare Token
ts.set_token('2bf25750566da626ec1d3c56edec5c63546a685b4dfbed78e109917a')
pro = ts.pro_api()

# 设置 matplotlib 使用支持中文的字体
rcParams['font.family'] = 'SimSong'  # 如果 'SimHei' 字体不可用，可以尝试其他中文字体，如 'PingFang'

# 获取股票历史数据
def get_stock_data(symbol, start_date, end_date):
    df = pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
    df.rename(columns={'vol': 'volume'}, inplace=True)  # 重命名成交量列
    return df

# 计算技术指标（包括 MFI, OBV, CCI 等）
def add_technical_indicators(df):
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['signal'], df['hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
    df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['dpo'] = df['close'] - talib.MA(df['close'], timeperiod=20)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['chaikin_oscillator'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
    df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
    df['obv'] = talib.OBV(df['close'], df['volume'])
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df = df.fillna(0)
    return df

# 数据预处理：去除缺失值并进行归一化处理
def preprocess_data(df):
    df = df.dropna()  # 删除缺失值
    df = df.select_dtypes(include=[np.number])  # 只保留数值型列
    scaler = MinMaxScaler(feature_range=(0, 1))  # 初始化 MinMaxScaler
    data_scaled = scaler.fit_transform(df)  # 归一化数据至 [0, 1]
    return df, data_scaled, scaler

# 创建时间序列数据集
def create_dataset(data_scaled, look_back=180):
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i - look_back:i, :])
        y.append(data_scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    return X, y

# 创建LSTM模型
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(LSTM(units=50, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error')
    return model

# 反归一化收盘价
def inverse_transform_close(scaler, scaled_data):
    close_min = scaler.data_min_[0]
    close_max = scaler.data_max_[0]
    return scaled_data * (close_max - close_min) + close_min

# 主流程
symbol = '601996.SH'
df = get_stock_data(symbol, '20110101', '20250219')
df = add_technical_indicators(df)
df, data_scaled, scaler = preprocess_data(df)

look_back = 180
X, y_normalized = create_dataset(data_scaled, look_back)
real_close = df['close'].iloc[look_back:].values

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y_normalized[:train_size], y_normalized[train_size:]
real_close_train, real_close_test = real_close[:train_size], real_close[train_size:]

# 创建LSTM模型
model = create_lstm_model((X_train.shape[1], X_train.shape[2]))

# 使用早停机制监控验证损失
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 保存模型
model.save('stock_predictor_model.h5')  # 保存为 .h5 格式

# 预测
predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)

# 反归一化
predicted_train_price = inverse_transform_close(scaler, predicted_train.flatten())
predicted_test_price = inverse_transform_close(scaler, predicted_test.flatten())

# 评估模型
rmse = np.sqrt(mean_squared_error(real_close_test, predicted_test_price))
mae = mean_absolute_error(real_close_test, predicted_test_price)
r2 = r2_score(real_close_test, predicted_test_price)
print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")
print(f"Test R²: {r2}")

# 可视化
plt.plot(real_close_test, label='真实价格')
plt.plot(predicted_test_price, label='预测价格')
plt.legend()
plt.show()