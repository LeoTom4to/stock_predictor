# config.py

# Tushare Token（请勿泄露给他人）
TUSHARE_TOKEN = '2bf25750566da626ec1d3c56edec5c63546a685b4dfbed78e109917a'

# 多股票列表（可自行替换为沪深300或其他A股代码）
STOCK_LIST = [
    '600519.SH',  # 贵州茅台
    '000001.SZ',  # 平安银行
    '000651.SZ',  # 格力电器
    '601318.SH',  # 中国平安
    '300750.SZ',  # 宁德时代
]

# 通用数据设置
START_DATE = '20180101'
END_DATE = '20250101'
LOOK_BACK = 60             # LSTM输入序列长度
PREDICT_HORIZON = 1        # 预测未来几天内的上涨概率
TEST_RATIO = 0.2           # 测试集比例

# 训练参数
EPOCHS = 50
BATCH_SIZE = 32

# 模型保存路径
MODEL_SAVE_PATH = 'model/lstm_stock_model.keras'
