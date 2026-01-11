import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import SpectralCNN, SpectralTransformer   # import model classes
from utils.metrics import (
    evaluate_model, 
    compute_confusion_matrix, 
    compute_multiclass_auc, 
    compute_pca
)
from utils.plots import (
    confusion_plot, 
    plot_roc_multiclass, 
    plot_pca, 
    plot_confidence_histogram,
    plot_training_curves
)

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASSES = ["long-pass", "short-pass", "band-pass", "neutral-density"]

# Loading test data -------
data = np.load("data/processed/spectra_preprocessed.npz")
X = data["X_test"]
y = data["y_test"]
X_train = data["X_train"]
y_train = data["y_train"]

# Convert to PyTorch tensors ---------
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
test_dataset = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# Evaluate  
# -----------------------
# CNN -----------------
cnn_model = SpectralCNN(input_length=401,num_classes=NUM_CLASSES).to(DEVICE)
cnn_acc, cnn_report, cnn_labels, cnn_probs = evaluate_model(cnn_model, "models/baseline_cnn_best.path", test_loader, DEVICE, CLASSES)
cnn_model.eval()

print("==== CNN Evaluation ====")
print(f"Test Accuracy: {cnn_acc*100:.2f}%")
print(cnn_report)

# Transformer ------------
transformer_model = SpectralTransformer(input_length=401, num_classes=NUM_CLASSES).to(DEVICE)
trans_acc, trans_report, trans_labels, trans_probs = evaluate_model(transformer_model, "models/transformer_best.pth", test_loader, DEVICE, CLASSES)
transformer_model.eval()

print("==== Transformer Evaluation ====")
print(f"Test Accuracy: {trans_acc*100:.2f}%")
print(trans_report)

# -----------------------------
# Confusion Matrix 
# -----------------------------
# CNN ------------------
cm_cnn = compute_confusion_matrix(cnn_model, test_loader, DEVICE)
class_names = ["long-pass", "short-pass", "band-pass", "neutral-density"] 

confusion_plot(
    cm_cnn, 
    class_names=class_names,
    normalize=True,
    title="CNN Confusion Matrix"
)

# Transformer -------------
cm_transformer = compute_confusion_matrix(transformer_model, test_loader, DEVICE)

confusion_plot(
    cm_transformer,
    class_names=class_names,
    normalize=True,
    title="Transformer Confuson Matrix"
)

# -------------------------
# ROC curve and AUC
# -------------------------
# CNN ---------------
cnn_auc_per_class, cnn_macro_auc = compute_multiclass_auc(cnn_labels, cnn_probs, num_classes=NUM_CLASSES)
plot_roc_multiclass(
    cnn_labels, cnn_probs, num_classes=NUM_CLASSES
)
# Transformer ----------------
trans_auc_per_class, trans_macro_auc = compute_multiclass_auc(trans_labels, trans_probs, num_classes=NUM_CLASSES)
plot_roc_multiclass(
    trans_labels, trans_probs, num_classes=NUM_CLASSES
)

# ------------------------
# Compute and plot PCA
# ------------------------
# Combine train + test for visualization 
X_all = np.concatenate([X_train, X])
y_all = np.concatenate([y_train, y])

# Compute PCA
X_pca, pca_model = compute_pca(X_all, n_components=2)
print(f"Explained variance ratios: {pca_model.explained_variance_ratio_}")

# Plot PCA
plot_pca(X_pca, y_all, title="2D PCA of Spectra Dataset")

# ------------------------------
# Plot confidence histogram
# ------------------------------
# CNN ------------
cnn_preds = np.argmax(cnn_probs, axis=1)
plot_confidence_histogram(cnn_probs, cnn_preds, cnn_labels, model_name="CNN")

# Transformer ----------
trans_preds = np.argmax(trans_probs, axis=1)
plot_confidence_histogram(trans_probs, trans_preds, trans_labels, model_name="Transformer")

# ------------------
# Training curve 
# ------------------
cnn_history = np.load("models/logs/cnn_training_history.npz")
plot_training_curves(
    cnn_history["train_loss"],
    cnn_history["val_loss"],
    cnn_history["train_acc"],
    cnn_history["val_acc"],
    model_name="CNN"
)

transformer_history = np.load("models/logs/trans_training_history.npz")
plot_training_curves(
    transformer_history["train_loss"],
    transformer_history["val_loss"],
    transformer_history["train_acc"],
    transformer_history["val_acc"],
    model_name="Transformer"
)