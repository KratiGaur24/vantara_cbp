import torch
import torch.nn as nn

# =====================================================================
# 1. ARTIFICIAL NEURAL NETWORK (ANN) FOR CHURN CLASSIFICATION
# =====================================================================
class CustomerChurnANN(nn.Module):
    """Multi-Layer Perceptron with Batch Normalization and Dropout."""
    def __init__(self, input_dim=5):
        super(CustomerChurnANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        return self.sigmoid(self.fc2(x))

# =====================================================================
# 2. RECURRENT NEURAL NETWORK (LSTM) FOR TIMING SEQUENCES
# =====================================================================
class CustomerLSTM(nn.Module):
    """Sequential LSTM for analyzing multi-step purchase timing."""
    def __init__(self, input_dim=5, hidden_dim=16):
        super(CustomerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # Extract final time step
        return self.sigmoid(self.fc(last_step))

# =====================================================================
# 3. AUTOENCODER FOR ANOMALY DETECTION
# =====================================================================
class CustomerAutoencoder(nn.Module):
    """Self-supervised bottleneck Autoencoder for reconstruction error."""
    def __init__(self, input_dim=5):
        super(CustomerAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))