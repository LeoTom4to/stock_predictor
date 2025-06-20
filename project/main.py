from project.config import STOCK_LIST, START_DATE, END_DATE, LOOK_BACK, MODEL_SAVE_PATH

from project.data.data_fetch import get_multi_stock_data
from project.data.feature_engineering import add_technical_indicators
from project.data.target_generator import generate_classification_target
from project.data.preprocess import prepare_lstm_data_with_stockcode

from project.model.lstm_with_stock_embedding import create_lstm_with_stock_embedding
from project.model.train import train_model
from project.model.predict import load_trained_model, predict_probabilities
from project.model.evaluate import evaluate_classifier

from project.strategy.signal_generator import generate_signals
from project.strategy.backtest import backtest_strategy
from project.strategy.metrics import compute_strategy_metrics
from project.strategy.threshold_experiment import run_threshold_experiment, plot_threshold_performance

from project.utils.visualization import plot_signals, plot_returns

import numpy as np
from sklearn.model_selection import train_test_split

# ====== Step 1: 获取数据 ======
df = get_multi_stock_data(STOCK_LIST, START_DATE, END_DATE)
df = add_technical_indicators(df)
df = generate_classification_target(df, horizon=3)

# ====== Step 2: 数据预处理（含股票编码） ======
X, y, stock_codes, scaler = prepare_lstm_data_with_stockcode(df, look_back=LOOK_BACK)
print(f"Samples: {X.shape[0]}, Features: {X.shape[2]}")
num_stocks = int(np.max(stock_codes)) + 1

# ====== Step 3: 划分训练集 / 测试集（不打乱） ======
X_train, X_test, y_train, y_test, sc_train, sc_test = train_test_split(
    X, y, stock_codes, test_size=0.2, shuffle=False
)

# ====== Step 4: 构建带嵌入结构的 LSTM 模型 ======
model = create_lstm_with_stock_embedding(
    input_shape=(X.shape[1], X.shape[2]),
    num_stocks=num_stocks
)

# ====== Step 5: 训练模型 ======
train_model(model, X_train, sc_train, y_train)

# ====== Step 6: 加载模型，预测测试集概率 ======
model = load_trained_model()
probs = predict_probabilities(model, X_test, sc_test)

# ====== Step 7: 分类评估 ======
evaluate_classifier(y_test, probs, threshold=0.5)

# ====== Step 8: 策略信号与回测收益 ======
test_df = df.iloc[-len(probs):]
real_close = test_df['close'].values
signals = generate_signals(probs, threshold=0.5)
returns = backtest_strategy(signals, real_close)

# ====== Step 9: 输出策略评估结果 ======
metrics = compute_strategy_metrics(returns)
print("Strategy Evaluation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

plot_signals(real_close[1:], signals, title="Close Price with Buy Signals")
benchmark_returns = (real_close[1:] - real_close[:-1]) / real_close[:-1]
plot_returns(returns, benchmark_returns, title="Cumulative Returns Comparison")

# ====== Step 10: 多阈值对比实验 ======
thresholds, win_rates, profits, sharpes = run_threshold_experiment(probs, real_close)
plot_threshold_performance(thresholds, win_rates, profits, sharpes)
