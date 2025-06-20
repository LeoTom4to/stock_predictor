# strategy/signal_generator.py

import numpy as np

def generate_signals(probabilities, threshold=0.6):
    """
    将预测概率转为操作信号：
    1 表示预测未来上涨（可买入），0 表示观望/空仓
    """
    return (probabilities > threshold).astype(int)
