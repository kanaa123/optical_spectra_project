import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import SpectralCNN

# -------------------------------
# Configuration parameters
# -------------------------------
DATA_PATH = "data/processed/spectra_preprocessed.npz"
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
LR = 5e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# History variables
# ---------------------------
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# ---------------------------
# Load dataset 
# ---------------------------
data = np.load(DATA_PATH)

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.long)
X_val = torch.tensor(data["X_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------
# Initialize model, loss, optimizer
# -----------------------------------
num_classes = len(np.unique(data["y_train"]))
model = SpectralCNN(num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------------------------
# Training Loop with early stopping
# --------------------------------------
best_val_loss = float("inf")
epochs_no_improve = 0
best_model_path = os.path.join(SAVE_DIR, "baseline_cnn_best.path")

for epoch in range(1, NUM_EPOCHS + 1):
    # ----- Train -----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(output, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = running_loss / total
    train_acc = correct / total 
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    # ---- Validation -----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    val_loss /= total
    val_acc = correct / total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
          f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

    # ---- Early stopping chack ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  --> Best model saved.")
    else: 
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

# ------------------------
# Save training history
# ------------------------
np.savez(
    "models/logs/cnn_training_history.npz",
    train_loss=train_loss_history,
    val_loss=val_loss_history,
    train_acc=train_acc_history,
    val_acc=val_acc_history
)

print("training completed. Best model saved at: ", best_model_path)
