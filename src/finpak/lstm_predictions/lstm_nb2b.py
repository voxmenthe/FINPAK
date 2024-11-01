import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.preprocessing import StandardScaler
from finpak.data.fetchers.yahoo import download_multiple_tickers


def get_device():
  if torch.backends.mps.is_available():
      return torch.device("mps")
  elif torch.cuda.is_available():
      return torch.device("cuda")
  else:
      return torch.device("cpu")

# Use this device throughout your code
device = get_device()
print(f"Using device: {device}")


tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2000-04-01'
end_date = '2024-08-31'

# Download historical data for the tickers
data_df = download_multiple_tickers(tickers, start_date, end_date)
data_df = data_df.loc[:,'Adj Close'] # Extract from multi-index dataframe

# Compute percentage changes
prices = data_df['AAPL']
data_df['pct_change_1d'] = prices.pct_change(periods=1)
data_df['pct_change_2d'] = prices.pct_change(periods=2)
data_df['target'] = data_df['pct_change_1d'].shift(-1)  # Next day's 1-day % change

# Drop NaN values resulting from percentage change calculations
data_df = data_df.dropna()

# Extract features and target
features = data_df[['pct_change_1d', 'pct_change_2d']].values
y = data_df['target'].values

# Standardize features and target
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(features)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

"""
- **Create Sequences:**
- Decide on a sequence length (e.g., 10).
- Convert the percentage changes into input-output pairs where each input is a sequence of `sequence_length` steps, and the output is the next value.
"""
sequence_length = 10
X_sequences = []
y_sequences = []

for i in range(len(X_scaled) - sequence_length):
  X_sequences.append(X_scaled[i:i + sequence_length])
  y_sequences.append(y_scaled[i + sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

class StockDataset(Dataset):
  def __init__(self, sequences, targets):
      self.sequences = sequences
      self.targets = targets

  def __len__(self):
      return len(self.sequences)

  def __getitem__(self, idx):
      sequence = torch.tensor(self.sequences[idx], dtype=torch.float32).to(device)
      target = torch.tensor(self.targets[idx], dtype=torch.float32).to(device)
      return sequence, target

class SwishActivation(nn.Module):
  def forward(self, x):
      return x * torch.sigmoid(x)

class LSTMLayer(nn.Module):
  def __init__(self, input_size, hidden_size):
      super(LSTMLayer, self).__init__()
      self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
      self.layer_norm = nn.LayerNorm(hidden_size)
      self.swish = SwishActivation()

  def forward(self, x, hidden=None):
      out, hidden = self.lstm(x, hidden)
      out = self.layer_norm(out)
      out = self.swish(out)
      return out, hidden

class EnhancedLSTMModel(nn.Module):
  def __init__(self, input_size=2, hidden_size=50, num_layers=2):
      super(EnhancedLSTMModel, self).__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      
      self.layers = nn.ModuleList([
          LSTMLayer(input_size if i == 0 else hidden_size, hidden_size)
          for i in range(num_layers)
      ])
      
      self.fc = nn.Linear(hidden_size, 1)
      self.layer_norm_final = nn.LayerNorm(hidden_size)
      self.swish = SwishActivation()

  def forward(self, x):
      batch_size = x.size(0)
      h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
      c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
      hidden = (h0, c0)

      residual = None
      for i, layer in enumerate(self.layers):
          if i == 0:
              out, hidden = layer(x, hidden)
          else:
              layer_input = out if residual is None else out + residual
              out, hidden = layer(layer_input, hidden)
          
          if i > 0:  # Apply residual connection for all layers except the first
              residual = out

      out = self.layer_norm_final(out)
      out = self.swish(out)
      out = self.fc(out[:, -1, :])
      return out.squeeze()

# Create the full dataset
full_dataset = StockDataset(X_sequences, y_sequences)

# Define split ratios
train_ratio = 0.88
val_ratio = 0.12

# Calculate split indices
train_size = int(len(y_sequences) * train_ratio)
val_size = int(len(y_sequences) * val_ratio)

# Split the dataset sequentially
train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Adjust input_size in the model to match the feature size (2 in this case)
input_size = X_sequences.shape[2]
model = EnhancedLSTMModel(input_size=input_size, hidden_size=2048, num_layers=3).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00075)
best_val_loss = float('inf')

num_epochs = 112
patience = 18  # For early stopping
epochs_without_improvement = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
  model.train()
  epoch_train_loss = 0
  for sequences, targets in train_loader:
      sequences, targets = sequences.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = model(sequences)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
      epoch_train_loss += loss.item()
  
  avg_train_loss = epoch_train_loss / len(train_loader)
  train_losses.append(avg_train_loss)
  
  # Validation phase
  model.eval()
  epoch_val_loss = 0
  with torch.no_grad():
      for sequences, targets in val_loader:
          sequences, targets = sequences.to(device), targets.to(device)
          outputs = model(sequences)
          val_loss = criterion(outputs, targets)
          epoch_val_loss += val_loss.item()
  
  avg_val_loss = epoch_val_loss / len(val_loader)
  val_losses.append(avg_val_loss)
  
  # Print statistics
  print(f'Epoch [{epoch+1}/{num_epochs}]')
  print(f'Train Loss: {avg_train_loss:.4f}')
  print(f'Validation Loss: {avg_val_loss:.4f}')
  
  # Early stopping check
  if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      epochs_without_improvement = 0
      # When saving the model, move it to CPU first
      torch.save(model.to('cpu').state_dict(), 'best_lstm_model.pth')
      model.to(device)  # Move it back to the device if you continue training
      print("New best model saved!")
  else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= patience:
          print(f"Early stopping triggered after {epoch+1} epochs")
          break
  
  print('-' * 50)

"""
### **5. Model Inference and Graphing**

Note: Generating 2-day percentage changes in the inference loop is challenging without actual future prices. For simplicity, we'll adjust the inference to use only existing features.
"""

# Load the trained model
input_size = X_sequences.shape[2]
model = EnhancedLSTMModel(input_size=input_size, hidden_size=2048, num_layers=3).to(device)
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval()

# Prepare data for plotting
# Use the validation set for predictions
val_features = X_sequences[train_size:train_size + val_size]
val_targets = y_sequences[train_size:train_size + val_size]

# Get predictions
model_predictions = []
with torch.no_grad():
  for sequence in val_features:
      sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
      prediction = model(sequence)
      model_predictions.append(prediction.item())

# Inverse transform the predictions and targets
model_predictions = scaler_y.inverse_transform(np.array(model_predictions).reshape(-1, 1)).flatten()
actual_targets = scaler_y.inverse_transform(val_targets.reshape(-1, 1)).flatten()

# Since the target is the next day's percentage change, reconstruct the predicted prices
# Start from the last price in the training set
last_train_price = prices[train_size + sequence_length - 1]
predicted_prices = [last_train_price]
actual_prices = [last_train_price]

for pred_pct_change in model_predictions:
  next_price = predicted_prices[-1] * (1 + pred_pct_change)
  predicted_prices.append(next_price)

for actual_pct_change in actual_targets:
  next_price = actual_prices[-1] * (1 + actual_pct_change)
  actual_prices.append(next_price)

# Remove the first element as it was the initial price
predicted_prices = predicted_prices[1:]
actual_prices = actual_prices[1:]

# Prepare dates for plotting
val_dates = data_df.index[train_size + sequence_length:train_size + sequence_length + val_size]

# Plotting
plt.figure(figsize=(15, 8))
plt.plot(val_dates, actual_prices, label='Actual Stock Price', color='blue', linewidth=2)
plt.plot(val_dates, predicted_prices, label='Predicted Stock Price', color='red', linewidth=2)
plt.title('LSTM Model Predictions vs Actual Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()