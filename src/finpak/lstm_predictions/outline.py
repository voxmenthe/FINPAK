### **1. Data Processing**

"""
**Objective:** Convert a vector of raw stock prices into percentage changes, optionally adjusted by volatility (standard deviation), and prepare sequence data for the model.

#### **Steps:**

- **Load Stock Price Data:**
  - Read your stock price data into a NumPy array or Pandas DataFrame.
  - Ensure the data is in chronological order (oldest to newest).
"""
import pandas as pd

df = pd.read_csv('stock_prices.csv')  # Replace with your file
prices = df['Close'].values  # Assuming 'Close' column has the prices

"""
- **Calculate Percentage Changes:**
  - Compute the percentage change between consecutive prices.
  - Optionally adjust the percentage changes by dividing by the volatility (standard deviation).
"""

import numpy as np

# Calculate percentage changes
pct_changes = np.diff(prices) / prices[:-1]

# Optionally adjust by volatility
adjust_by_volatility = True
if adjust_by_volatility:
    volatility = np.std(pct_changes)
    pct_changes = pct_changes / volatility

"""
- **Normalize Data (Optional):**
  - Normalize the percentage changes to have zero mean and unit variance.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pct_changes = scaler.fit_transform(pct_changes.reshape(-1, 1)).flatten()

"""
- **Create Sequences:**
  - Decide on a sequence length (e.g., 50).
  - Convert the percentage changes into input-output pairs where each input is a sequence of `sequence_length` steps, and the output is the next value.
"""

sequence_length = 50
X = []
y = []

for i in range(len(pct_changes) - sequence_length):
    X.append(pct_changes[i:i + sequence_length])
    y.append(pct_changes[i + sequence_length])

X = np.array(X)
y = np.array(y)

"""

### **2. Dataset and DataLoader Setup**

**Objective:** Create a PyTorch Dataset and DataLoader to efficiently feed data into the model during training.

#### **Steps:**

- **Create a Custom Dataset:**
"""

import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sequence, target

"""
- **Instantiate the Dataset and DataLoader:**
"""

dataset = StockDataset(X, y)
from torch.utils.data import DataLoader

batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

"""
### **3. Model Definition**

**Objective:** Define an LSTM-based neural network model using PyTorch.

#### **Steps:**

- **Define the Model Architecture:**
"""

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # LSTM forward pass
        out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))  # x.unsqueeze(-1) to add input_size dimension

        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out.squeeze()

"""
### **4. Model Training and Storage**

**Objective:** Train the model using the DataLoader and save the trained model for future inference.

#### **Steps:**

- **Set Up Training Components:**
"""

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for sequences, targets in dataloader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'lstm_model.pth')

"""

### **5. Model Inference and Graphing**

**Objective:** Use the trained model to make predictions, compare them with actual stock prices, and visualize the results.

#### **Steps:**

- **Load the Trained Model:**
"""

model = LSTMModel()
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

# Select a starting point for prediction
start_idx = 0  # Change as needed
input_seq = pct_changes[start_idx:start_idx + sequence_length]
input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

"""
- **Autoregressive Prediction:**
"""

prediction_length = 100  # Number of future steps to predict
predictions = []

with torch.no_grad():
    for _ in range(prediction_length):
        # Get the model prediction
        pred = model(input_seq)
        predictions.append(pred.item())

        # Update the input sequence by appending the prediction and removing the first element
        input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(-1)), dim=1)

"""
- **Denormalize Predictions (If Data Was Normalized):**
"""

# If you used StandardScaler
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actual_values = scaler.inverse_transform(pct_changes[start_idx + sequence_length:start_idx + sequence_length + prediction_length].reshape(-1, 1)).flatten()

# Reconstruct prices from percentage changes
last_known_price = prices[start_idx + sequence_length]
predicted_prices = [last_known_price]

for pct_change in predictions:
    next_price = predicted_prices[-1] * (1 + pct_change)
    predicted_prices.append(next_price)

actual_prices = prices[start_idx + sequence_length:start_idx + sequence_length + prediction_length + 1]


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(range(len(predicted_prices)), predicted_prices, label='Predicted Prices')
plt.plot(range(len(actual_prices)), actual_prices, label='Actual Prices')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.show()

### **Additional Implementation Hints:**

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# In training loop, move sequences and targets to the device
sequences, targets = sequences.to(device), targets.to(device)

"""
  - Normalize your data appropriately; sometimes Min-Max scaling works better.

- **Validation and Testing:**
  - Split your data into training and testing sets to evaluate model performance on unseen data.
  - Use metrics like Mean Absolute Error (MAE) in addition to MSE for evaluation.

- **Logging and Checkpointing:**
  - Use TensorBoard or other logging tools to monitor training progress.
  - Save model checkpoints periodically in case you need to resume training.

"""