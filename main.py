import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset (replace with actual path)
data = pd.read_csv('historical-hourly-weather-data.csv')

# Preprocess data
def preprocess_data(data):
    data.fillna(method='ffill', inplace=True)
    data['date'] = pd.to_datetime(data['datetime_column'])
    data.set_index('date', inplace=True)
    
    # Feature engineering: Use past 3 days' data to predict tomorrow
    for i in range(1, 4):
        data[f'temp_lag_{i}'] = data['temperature'].shift(i)
        data[f'wind_lag_{i}'] = data['wind_speed'].shift(i)
    
    data['wind_tomorrow'] = (data['wind_speed'].shift(-1) >= 6).astype(int)
    data.dropna(inplace=True)
    return data

processed_data = preprocess_data(data)

# Split features and labels
features = ['temp_lag_1', 'temp_lag_2', 'temp_lag_3', 'wind_lag_1', 'wind_lag_2', 'wind_lag_3']
X = processed_data[features].values
y_temp = processed_data['temperature'].values.reshape(-1, 1)
y_wind = processed_data['wind_tomorrow'].values.reshape(-1, 1)

# Train-test split
X_train, X_test, y_temp_train, y_temp_test, y_wind_train, y_wind_test = train_test_split(
    X, y_temp, y_wind, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Define the custom MLP class
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', learning_rate=0.01):
        self.learning_rate = learning_rate
        self.activation = relu if activation == 'relu' else sigmoid
        self.activation_derivative = relu_derivative if activation == 'relu' else sigmoid_derivative

        # Initialize weights and biases
        self.weights = [
            np.random.randn(input_size, hidden_sizes[0]),
            np.random.randn(hidden_sizes[0], hidden_sizes[1]),
            np.random.randn(hidden_sizes[1], output_size)
        ]
        self.biases = [
            np.zeros((1, hidden_sizes[0])),
            np.zeros((1, hidden_sizes[1])),
            np.zeros((1, output_size))
        ]

    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]

        for w, b in zip(self.weights, self.biases):
            X = np.dot(X, w) + b
            self.layer_inputs.append(X)
            X = self.activation(X)
            self.layer_outputs.append(X)
        
        return X

    def backward(self, X, y):
        m = y.shape[0]
        error = self.layer_outputs[-1] - y
        deltas = [error * self.activation_derivative(self.layer_outputs[-1])]

        for i in range(len(self.weights) - 1, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            deltas.append(error * self.activation_derivative(self.layer_outputs[i]))

        deltas.reverse()

        # Update weights and biases using SGD
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.layer_outputs[i].T.dot(deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m

    def train(self, X, y, epochs=500):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if epoch % 50 == 0:
                loss = np.mean((self.layer_outputs[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return self.forward(X)

# Train and evaluate MLP for temperature prediction
mlp_temp = MLP(input_size=X_train.shape[1], hidden_sizes=[50, 50], output_size=1, activation='relu', learning_rate=0.01)
mlp_temp.train(X_train, y_temp_train, epochs=500)

y_temp_pred = mlp_temp.predict(X_test)
temp_mae = mean_absolute_error(y_temp_test, y_temp_pred)
print(f"Temperature Prediction MAE: {temp_mae:.2f}")

# Train and evaluate MLP for wind prediction (classification)
mlp_wind = MLP(input_size=X_train.shape[1], hidden_sizes=[50, 50], output_size=1, activation='sigmoid', learning_rate=0.01)
mlp_wind.train(X_train, y_wind_train, epochs=500)

y_wind_pred = mlp_wind.predict(X_test)
wind_auc = roc_auc_score(y_wind_test, y_wind_pred)
print(f"Wind Prediction AUC: {wind_auc:.2f}")
