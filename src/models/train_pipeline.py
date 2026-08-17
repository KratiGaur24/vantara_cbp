# =====================================================================
# SYSTEM LAYER CONFIGURATION & PIPELINE IMMUTABILITY ENVIRONMENT
# =====================================================================
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def enforce_pipeline_reproducibility(seed: int = 42):
    """Locks framework stochasticity across all engine abstractions to guarantee deterministic execution."""
    random.seed(seed)                                  # Seed native Python random generator
    np.random.seed(seed)                              # Seed NumPy linear algebra vector engine
    torch.manual_seed(seed)                            # Seed PyTorch CPU calculation graphs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)               # Seed all available local GPU graphics cores

# Execute immutability anchor before initializing network graphs
enforce_pipeline_reproducibility(seed=42)

# Select compute backend dynamically based on physical hardware availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Execution hardware baseline initialized to target backend device: {device}")

# =====================================================================
# DATA INGESTION, MULTI-SPLIT PARTITIONING & TENSOR TRANSFORMATION
# =====================================================================
# 1. Load the processed customer feature matrix
data_path = r"D:\Personal\Vantara\vantara_cbp\data\processed\customer_features.csv"
df = pd.read_csv(data_path)

# 2. Separate clues (X) from target answer (y)
feature_cols = ['Recency', 'Frequency', 'Total_Spend', 'Avg_Basket_Size', 'Engagement_Score']
X = df[feature_cols].values
y = df['Churn_Target'].values

# 3. First Split: Isolate 70% for Training, 30% for Validation + Testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# 4. Second Split: Partition the 30% into 15% Validation and 15% Testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# 5. Fit StandardScaler ONLY on Training data, then transform Val and Test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. Convert scaled NumPy arrays to PyTorch 32-bit Float Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

print(f"📊 Data Scaffolding Complete:")
print(f"   ↳ Training Tensor Shape   : {X_train_tensor.shape}")
print(f"   ↳ Validation Tensor Shape : {X_val_tensor.shape}")
print(f"   ↳ Testing Tensor Shape    : {X_test_tensor.shape}")

# =====================================================================
# PYTORCH NEURAL NETWORK ARCHITECTURE & EARLY STOPPING REGULATOR
# =====================================================================
class CustomerChurnANN(nn.Module):
    """Custom Multi-Layer Perceptron for Tabular Churn Classification."""
    def __init__(self, input_dim=5):
        super(CustomerChurnANN, self).__init__()
        # Hidden Layer 1: Expand inputs to 32 nodes
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
        # Output Layer: Compress 32 hidden representations down to 1 scalar logit
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Passes input tensors sequentially through hidden layers and activations."""
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return self.sigmoid(out)

class EarlyStopping:
    """Monitors validation loss and halts training when improvement stalls."""
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# =====================================================================
# MINI-BATCH DATALOADERS, LOSS FUNCTION & ADAM OPTIMIZER
# =====================================================================
# 1. Package tensors into mini-batch DataLoaders (Batch size = 64)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. Instantiate Model, Loss Criteria, Optimizer, and Early Stopping
model = CustomerChurnANN(input_dim=5).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
early_stopping = EarlyStopping(patience=7, min_delta=0.001)

# 3. Model Training Loop
print("⏳ Initiating PyTorch Deep Neural Network Training...")
model.train()
for epoch in range(100):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()                  # Clear old gradients
        predictions = model(batch_X)           # Forward pass
        loss = criterion(predictions, batch_y) # Calculate loss
        loss.backward()                        # Backward pass (Gradients)
        optimizer.step()                       # Update weights

print("✅ Preliminary Training Epochs Complete!")
# =====================================================================
# MODEL EVALUATION, SCORECARD METRICS & WEIGHTS SERIALIZATION
# =====================================================================
# 1. Switch model to evaluation mode and disable gradient tracking
model.eval()
with torch.no_grad():
    # Generate raw churn probability predictions for the unseen Test set
    raw_probabilities = model(X_test_tensor)
    
    # Apply a 0.50 decision threshold to classify as Churned (1) or Active (0)
    test_predictions = (raw_probabilities >= 0.50).float()

# 2. Convert PyTorch tensors back to NumPy arrays for scikit-learn evaluation
y_true = y_test_tensor.cpu().numpy()
y_pred = test_predictions.cpu().numpy()

# 3. Compute evaluation metrics
from sklearn.metrics import accuracy_score, classification_report

final_accuracy = accuracy_score(y_true, y_pred) * 100
print("\n=======================================================")
print(f"🏆 PYTORCH DEEP NEURAL NETWORK ACCURACY: {final_accuracy:.2f}%")
print("=======================================================")
print("\n📋 DETAILED DEEP LEARNING CLASSIFICATION SCORECARD:")
print(classification_report(y_true, y_pred, target_names=['Active (0)', 'Churned (1)']))

# 4. Save trained PyTorch network state weights to disk
models_dir = r"D:\Personal\Vantara\vantara_cbp\models"
weights_path = os.path.join(models_dir, "pytorch_neural_net.pt")
torch.save(model.state_dict(), weights_path)
print(f"💾 Production PyTorch weights saved successfully to:\n   ↳ {weights_path}\n")

# =====================================================================
# DAY 12: LSTM TIME-SERIES SEQUENCE TENSOR CONSTRUCTION
# =====================================================================
import torch
import torch.nn as nn
import numpy as np

# Define sequence dimensions
sequence_length = 4   # 4 historical lookback intervals
feature_dim = 5       # 5 behavioral feature metrics

# Reshape static 2D scaled train array [3696, 5] into 3D sequence array [3696, 4, 5]
num_samples = X_train_scaled.shape[0]
X_train_seq = np.zeros((num_samples, sequence_length, feature_dim))

for step in range(sequence_length):
    # Simulate past temporal shifts leading up to current snapshot
    X_train_seq[:, step, :] = X_train_scaled * (1 - (sequence_length - 1 - step) * 0.05)

# Convert 3D NumPy array into PyTorch Float Tensor
X_train_seq_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
print(f"⏱️ Sequential 3D Training Tensor Shape: {X_train_seq_tensor.shape}")

# =====================================================================
# PYTORCH RECURRENT LSTM ARCHITECTURE
# =====================================================================
class CustomerLSTM(nn.Module):
    """Recurrent LSTM Architecture for Sequential Churn Prediction."""
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=1):
        super(CustomerLSTM, self).__init__()
        # 1. Recurrent LSTM Layer (batch_first=True expects [batch, seq, feature])
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 2. Linear classification layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # lstm_out shape: [batch_size, sequence_length, hidden_dim]
        lstm_out, (hn, cn) = self.lstm(x)
        # Extract representation from the final time step
        last_step_out = lstm_out[:, -1, :]
        return self.sigmoid(self.fc(last_step_out))
      
# =====================================================================
# LSTM MODEL INSTANTIATION & TRAINING LOOP
# =====================================================================
# 1. Instantiate the recurrent model
lstm_model = CustomerLSTM(input_dim=5, hidden_dim=32, num_layers=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

# 2. Sequential training loop
print("⏳ Initiating Sequential LSTM Training...")
lstm_model.train()
for epoch in range(20):
  optimizer.zero_grad()  # Reset accumulated gradients
  predictions = lstm_model(X_train_seq_tensor)  # Forward pass through sequence
  loss = criterion(predictions, y_train_tensor)  # Calculate loss
  loss.backward()  # Backpropagation Through Time (BPTT)
  optimizer.step()  # Update weights

print(f"✅ LSTM Training Complete! Final Training Loss: {loss.item():.4f}") 

# =====================================================================
# DAY 13: AUTOENCODER ARCHITECTURE FOR ANOMALY DETECTION
# =====================================================================
class CustomerAutoencoder(nn.Module):
    """Self-supervised Autoencoder for compressing and reconstructing features."""
    def __init__(self, input_dim=5):
        super(CustomerAutoencoder, self).__init__()
        # Encoder: Compress 5 features -> 2 hidden bottleneck dimensions
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
        # Decoder: Reconstruct 2 bottleneck dimensions -> 5 original features
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, input_dim)
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction

# =====================================================================
# DAY 13: AUTOENCODER TRAINING & ANOMALY THRESHOLD CALCULATION
# =====================================================================
# 1. Instantiate Autoencoder, MSE Loss, and Adam Optimizer
autoencoder = CustomerAutoencoder(input_dim=5).to(device)
criterion_mse = nn.MSELoss()
optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=0.005)

# 2. Train Autoencoder to reconstruct normal customer features
autoencoder.train()
for epoch in range(25):
    optimizer_ae.zero_grad()
    reconstructions = autoencoder(X_train_tensor)
    loss = criterion_mse(reconstructions, X_train_tensor)
    loss.backward()
    optimizer_ae.step()

# 3. Compute Reconstruction Error (MSE) per customer on Validation data
autoencoder.eval()
with torch.no_grad():
    val_recon = autoencoder(X_val_tensor)
    recon_errors = torch.mean((X_val_tensor - val_recon) ** 2, dim=1).cpu().numpy()

# 4. Calculate 95th Percentile Anomaly Cutoff Threshold
anomaly_threshold = np.percentile(recon_errors, 95)
print(f"✅ Autoencoder Training Complete | Final MSE Loss: {loss.item():.4f}")
print(f"📏 Anomaly Cutoff Threshold (95th Percentile): {anomaly_threshold:.4f}")

# 5. Save Model Weights to Disk
torch.save(autoencoder.state_dict(), r"D:\Personal\Vantara\vantara_cbp\models\autoencoder.pt")

# =====================================================================
# DAY 14: SHAP GLOBAL & INDIVIDUAL EXPLANATIONS
# =====================================================================
import joblib
import pandas as pd
import shap

# 1. Load trained champion tree model
model_path = r"D:\Personal\Vantara\vantara_cbp\models\random_forest_model.pkl"
rf_model = joblib.load(model_path)

# 2. Compute SHAP values on scaled test data
feature_names = [
    'Recency',
    'Frequency',
    'Total_Spend',
    'Avg_Basket_Size',
    'Engagement_Score',
]
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_df)

# Handle multi-class output shape (select Churned class)
shap_churn = (
    shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
)

# 3. Print Global Feature Importance Scorecard
global_imp = pd.DataFrame({
    'Feature': feature_names,
    'Mean_SHAP': np.abs(shap_churn).mean(axis=0),
}).sort_values(by='Mean_SHAP', ascending=False)

print("\n🔍 SHAP GLOBAL FEATURE IMPORTANCE RANKINGS:")
print(global_imp.to_string(index=False))

# =====================================================================
# DAY 15: LIME LOCAL EXPLANATION ENGINE
# =====================================================================
from lime.lime_tabular import LimeTabularExplainer

# 1. Initialize LIME Tabular Explainer on training data
lime_explainer = LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    class_names=['Active (0)', 'Churned (1)'],
    mode='classification',
    random_state=42
)

# 2. Generate local rule explanation for Customer #0
customer_sample = X_test_scaled[0]
lime_exp = lime_explainer.explain_instance(
    data_row=customer_sample,
    predict_fn=rf_model.predict_proba,
    num_features=5
)

print("=======================================================")
print("📋 LIME LOCAL RULE BREAKDOWN (Customer #0):")
print("=======================================================")
for feature_rule, weight in lime_exp.as_list():
    print(f"   ↳ {feature_rule:35s}: {weight:+.4f}")

print("\n🎉 WEEK 3 (DAYS 11–15) DEEP LEARNING & XAI PIPELINE COMPLETE!")