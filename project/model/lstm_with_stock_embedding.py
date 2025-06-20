# model/lstm_with_stock_embedding.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

def create_lstm_with_stock_embedding(input_shape, num_stocks, embedding_dim=4):
    """
    创建带有股票嵌入的 LSTM 模型
    """
    time_input = Input(shape=input_shape, name='features')
    stock_input = Input(shape=(1,), name='stock_code')  # 单个股票编码

    emb = Embedding(input_dim=num_stocks, output_dim=embedding_dim)(stock_input)
    emb = Flatten()(emb)
    emb_repeated = Dense(input_shape[0])(emb)  # 与时间步对齐
    emb_repeated = Reshape((input_shape[0], 1))(emb_repeated)

    merged = Concatenate(axis=-1)([time_input, emb_repeated])

    x = LSTM(64, return_sequences=False)(merged)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[time_input, stock_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
