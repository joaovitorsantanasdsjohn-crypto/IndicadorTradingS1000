import tensorflow as tf
import numpy as np

class SignalFilter:
    def __init__(self):
        # Modelo leve MLP para Render gratuito
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(6,)),  # 3 EMAs + RSI + 2 Bollinger
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Se tiver pesos treinados offline, pode carregar aqui:
        # self.model.load_weights('ml_weights.h5')

    def predict(self, features):
        """Recebe array (1,6) e retorna probabilidade"""
        features = np.array(features).reshape(1, -1)
        return self.model.predict(features, verbose=0)[0][0]
