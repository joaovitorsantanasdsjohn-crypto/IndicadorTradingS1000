import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class SignalFilter:
    def __init__(self):
        # Simula um modelo leve de Machine Learning
        self.scaler = MinMaxScaler()
        self.model = self._build_model()
        self._train_dummy_model()

    def _build_model(self):
        """Cria um modelo neural leve."""
        model = Sequential([
            Dense(16, input_dim=6, activation='relu'),
            Dropout(0.1),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _train_dummy_model(self):
        """Treina o modelo com dados fictícios só pra permitir previsões funcionais."""
        X = np.random.rand(200, 6)
        y = np.random.randint(0, 2, 200)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, epochs=10, batch_size=8, verbose=0)

    def predict(self, features):
        """Retorna uma probabilidade simulada de alta ou baixa."""
        try:
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            prob = float(self.model.predict(X_scaled, verbose=0)[0][0])
            return prob
        except Exception as e:
            print("Erro na predição do modelo:", e)
            return 0.5
