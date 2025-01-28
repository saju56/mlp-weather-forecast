import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score

def load_and_preprocess(city):
    # Load data for a specific city
    def load_csv(file, col_name):
        df = pd.read_csv(f'{file}.csv', parse_dates=['datetime'])
        df = df[['datetime', city]].resample('D', on='datetime').mean().rename(columns={city: col_name})
        return df
    
    # Load and process numerical features
    temp = load_csv('temperature', 'temp_mean')
    humidity = load_csv('humidity', 'humidity_mean')
    pressure = load_csv('pressure', 'pressure_mean')
    wind_speed = pd.read_csv('wind_speed.csv', parse_dates=['datetime'])
    wind_speed_max = wind_speed[['datetime', city]].resample('D', on='datetime').max().rename(columns={city: 'wind_speed_max'})
    wind_speed_mean = wind_speed[['datetime', city]].resample('D', on='datetime').mean().rename(columns={city: 'wind_speed_mean'})
    
    # Wind direction (sin/cos)
    wind_dir = pd.read_csv('wind_direction.csv', parse_dates=['datetime'])
    wind_dir['sin'] = np.sin(np.deg2rad(wind_dir[city]))
    wind_dir['cos'] = np.cos(np.deg2rad(wind_dir[city]))
    wind_dir_sin = wind_dir.resample('D', on='datetime')['sin'].mean().rename('wind_dir_sin')
    wind_dir_cos = wind_dir.resample('D', on='datetime')['cos'].mean().rename('wind_dir_cos')
    
    # Weather descriptions (binary flags)
    weather = pd.read_csv('weather_description.csv', parse_dates=['datetime'])
    keywords = ['rain', 'snow', 'cloud', 'clear', 'thunderstorm']
    weather[city] = weather[city].fillna('').astype(str)  # Convert all values to strings
    weather['date'] = weather['datetime'].dt.date

    weather_daily = weather.groupby('date')[city].agg(' '.join).reset_index()  # Now safe to join

    for kw in keywords:
        weather_daily[kw] = weather_daily[city].str.contains(kw, case=False).astype(int)
    weather_daily['date'] = pd.to_datetime(weather_daily['date'])
    weather_daily = weather_daily.set_index('date').drop(columns=[city])
    
    # Combine features
    features = temp.join(humidity).join(pressure).join(wind_speed_max).join(wind_speed_mean)
    features = features.join(wind_dir_sin).join(wind_dir_cos).join(weather_daily)
    
    # Add rolling mean for temperature
    #features['temp_rolling_mean'] = features['temp_mean'].rolling(window=3).mean()

    # Create labels (next day's temperature and wind)
    features['temp_target'] = features['temp_mean'].shift(-1)
    features['wind_target'] = (features['wind_speed_max'].shift(-1) >= 10).astype(int)
    features.dropna(inplace=True)
    
    return features


def create_sequences(data, window_size=3, stride=1):
    X, y_temp, y_wind = [], [], []
    for i in range(0, len(data) - window_size, stride):
        window = data.iloc[i:i+window_size]
        X.append(window.values.flatten())  # Flatten the window into a single row
        y_temp.append(data.iloc[i+window_size]['temp_target'])  # Target for the next day
        y_wind.append(data.iloc[i+window_size]['wind_target'])  # Binary wind target
    return np.array(X), np.array(y_temp), np.array(y_wind)

# Example for New York
data = load_and_preprocess('New York')
X, y_temp, y_wind = create_sequences(data)

# Split into train, validation, test
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_temp_train, y_temp_val, y_temp_test = y_temp[:train_size], y_temp[train_size:train_size+val_size], y_temp[train_size+val_size:]
y_wind_train, y_wind_val, y_wind_test = y_wind[:train_size], y_wind[train_size:train_size+val_size], y_wind[train_size+val_size:]

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        self.biases = []
        self.val_loss_history = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes)-1):
            # He initialization modified for stability
            #self.weights.append(np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2. / sizes[i]) * 0.1)
            self.weights.append(np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(1. / sizes[i]))
            #self.weights.append(np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2. / sizes[i]))  # He initialization
            self.biases.append(np.zeros(sizes[i+1]))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def stable_sigmoid(self, z):
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)), 
                        np.exp(z) / (1 + np.exp(z)))
    
    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights)-1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.activations.append(self.stable_sigmoid(z))
        # Output layer (linear for temp, sigmoid for wind)
        z_output = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        temp_pred = z_output[:, 0]
        wind_pred = self.stable_sigmoid(z_output[:, 1])  # Use stable version
        return temp_pred, wind_pred
    
    def compute_loss(self, temp_pred, wind_pred, y_temp, y_wind):
        mse = np.mean((temp_pred - y_temp) ** 2)
        bce = -np.mean(y_wind * np.log(wind_pred + 1e-8) + (1 - y_wind) * np.log(1 - wind_pred + 1e-8))
        return mse + bce, mse, bce

    def backward(self, X, temp_pred, wind_pred, y_temp, y_wind, lr):
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        
        # Output gradients
        delta_temp = 2 * (temp_pred - y_temp) / len(y_temp)
        delta_wind = (wind_pred - y_wind) / len(y_wind)
        delta_output = np.column_stack((delta_temp[:, None], delta_wind[:, None]))
        clip_value = 1.0
        # Backpropagate
        delta = delta_output
        for i in reversed(range(len(self.weights))):
            a = self.activations[i]
            grads_w[i] = a.T @ delta
            grads_b[i] = np.sum(delta, axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * a * (1 - a)  # sigmoid derivative
                delta = np.clip(delta, -clip_value, clip_value)

        
        # for i in range(len(grads_w)):
        #     grads_w[i] = np.clip(grads_w[i], -clip_value, clip_value)
        #     grads_b[i] = np.clip(grads_b[i], -clip_value, clip_value)
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]
    
    def train(self, X_train, y_temp, y_wind, epochs=100, lr=0.001, batch_size=32):
        self.val_loss_history = []
        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch = indices[i:i+batch_size]
                X_batch = X_train[batch]
                y_temp_batch = y_temp[batch]
                y_wind_batch = y_wind[batch]
                
                temp_pred, wind_pred = self.forward(X_batch)
                self.backward(X_batch, temp_pred, wind_pred, y_temp_batch, y_wind_batch, lr)
            
            # Validation loss
            temp_val, wind_val = self.forward(X_val)
            loss, mse, bce = self.compute_loss(temp_val, wind_val, y_temp_val, y_wind_val)
            self.val_loss_history.append(loss)
            print(f'Epoch {epoch+1}, Val Loss: {loss:.4f}')
        return self.val_loss_history

def plot_and_save_loss(loss_history, filename='validation_loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_log_loss(loss_history, filename='validation_loss_log.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Validation Loss During Training (Log Scale)')
    plt.grid(True, which='both')
    plt.savefig(filename)
    plt.close()

def plot_predictions(y_true, y_pred, filename='predictions_vs_actual.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Temperature (°C)')
    plt.title('Predicted vs Actual Temperatures')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Initialize and train
input_size = X_train.shape[1]
model = MLP(input_size, hidden_sizes=[256, 128], output_size=2)
loss_history = model.train(X_train, y_temp_train, y_wind_train, epochs=1000, lr=0.0001)

# Plot and save
plot_and_save_loss(loss_history)
plot_log_loss(loss_history)       # Additional log plot
# Predict on test set
temp_pred, wind_pred = model.forward(X_test)

# Calculate metrics
mae = mean_absolute_error(y_temp_test, temp_pred)
auc = roc_auc_score(y_wind_test, wind_pred)
absolute_errors = np.abs(temp_pred - y_temp_test)
accuracy_within_2deg = (absolute_errors <= 2).mean() * 100

print(f'Temperature MAE: {mae:.2f}°C')
print(f'Temperature Accuracy (±2°C): {accuracy_within_2deg:.2f}%')
print(f'Wind AUC: {auc:.2f}')

plot_predictions(y_temp_test, temp_pred)