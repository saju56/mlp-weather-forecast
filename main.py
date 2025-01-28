import os
import kagglehub
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Download dataset
dataset_path = kagglehub.dataset_download("selfishgene/historical-hourly-weather-data")
print(f"Dataset downloaded to: {dataset_path}")


# Load and preprocess temperature data
def load_temperature_data(file_path):
    df = pd.read_csv(os.path.join(file_path, "temperature.csv"))

    # Melt to long format
    melted = df.melt(id_vars=['datetime'], var_name='city', value_name='temperature')

    # Convert datetime and filter valid data
    melted['datetime'] = pd.to_datetime(melted['datetime'])
    melted = melted[melted['datetime'] >= '2012-10-01 13:00:00']  # Remove initial NaNs

    # Convert from Kelvin to Celsius and clean data
    melted['temperature'] = melted['temperature'] - 273.15
    melted = melted.dropna(subset=['temperature'])  # Remove rows with missing temperatures

    return melted


# Load your temperature data
temp_df = load_temperature_data(dataset_path)

# Create daily aggregates
daily_data = temp_df.groupby(['city', pd.Grouper(key='datetime', freq='D')]) \
    .agg({'temperature': 'mean'}) \
    .reset_index()


# Create sequences with proper IIIXO window
def create_sequences(data, window_size=3, gap=1):
    sequences = []
    targets = []

    encoder = OneHotEncoder(sparse_output=False)
    cities = data['city'].unique()
    encoder.fit(data[['city']])

    for city in cities:
        city_df = data[data['city'] == city].sort_values('datetime')
        city_dates = city_df['datetime'].unique()

        for i in range(len(city_dates) - window_size - gap - 1):
            # Input: 3 consecutive days
            input_days = city_df[city_df['datetime'].isin(city_dates[i:i + window_size])]

            # Target: day after gap + window (IIIXO pattern)
            target_day = city_df[city_df['datetime'] == city_dates[i + window_size + gap]]

            if len(input_days) != window_size or len(target_day) == 0:
                continue  # Skip incomplete sequences

            # Encode city
            city_code = encoder.transform([[city]]).flatten()

            # Create features (3 days of temperatures)
            features = input_days['temperature'].values
            full_features = np.concatenate([city_code, features])

            sequences.append(full_features)
            targets.append(target_day['temperature'].values[0])

    return np.array(sequences), np.array(targets)


# Create sequences and targets
# X, y = create_sequences(daily_data)
# np.save('X_cache.npy', X)
# np.save('y_cache.npy', y)

X = np.load('X_cache.npy')
y = np.load('y_cache.npy')

#
cities = daily_data['city'].unique()
plt.figure(figsize=(12, 6))

for city in cities:
    city_data = daily_data[daily_data['city'] == city].sort_values('datetime')
    plt.plot(city_data['datetime'], city_data['temperature'], label=city, marker='o')

plt.title('Temperature Over Time for Multiple Cities')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#

# Train-test split with temporal preservation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Normalization (excluding city encoding)
num_cities = X.shape[1] - 3  # 3 days of temperature data
scaler = StandardScaler()
X_train[:, num_cities:] = scaler.fit_transform(X_train[:, num_cities:])
X_val[:, num_cities:] = scaler.transform(X_val[:, num_cities:])
X_test[:, num_cities:] = scaler.transform(X_test[:, num_cities:])

# Convert to PyTorch tensors
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))


# MLP Model
class TemperaturePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = TemperaturePredictor(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# Training loop
def train_model(model, train_loader, val_loader, epochs=100):
    best_mae = float('inf')
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_loader.dataset.tensors[0]).squeeze()
            val_mae = mean_absolute_error(val_loader.dataset.tensors[1], val_preds)

            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch + 1}/{epochs} | Val MAE: {val_mae:.2f}°C")


# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Train the model
train_model(model, train_loader, val_loader)

# Final evaluation
model.load_state_dict(torch.load('best_model.pth'))
with torch.no_grad():
    test_preds = model(test_loader.dataset.tensors[0]).squeeze()
    test_mae = mean_absolute_error(test_loader.dataset.tensors[1], test_preds)
    print(f"\nFinal Test MAE: {test_mae:.2f}°C")
    print(
        f"Accuracy: {sum(np.abs(test_preds.numpy() - test_loader.dataset.tensors[1].numpy()) <= 2) / len(test_preds):.1%} of predictions within ±2°C")