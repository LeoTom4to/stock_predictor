# model/predict.py

import numpy as np
from tensorflow.keras.models import load_model
from project.config import MODEL_SAVE_PATH

def load_trained_model():
    return load_model(MODEL_SAVE_PATH)

def predict_probabilities(model, X, stock_codes):
    """
    输入时间序列 + 股票编码，返回上涨概率
    """
    probs = model.predict([X, stock_codes])
    return probs.flatten()
