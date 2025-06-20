# model/train.py

import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from project.config import EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH

def train_model(model, X_train, stock_train, y_train):
    """
    训练带股票编码输入的 LSTM 分类模型
    """
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 自动类别权重
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print("Auto class weights:", class_weight_dict)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        [X_train, stock_train], y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=1
    )

    model.save(MODEL_SAVE_PATH)
    return history
