import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_model(model, checkpoint_path, test_loader, device, class_name):
    # Load model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.numpy())
            all_probs.append(probs.cpu().numpy())

    # Combine all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Accuracy 
    acc = accuracy_score(all_labels, all_preds)

    # Recall and F1-score (per class)
    report = classification_report(all_labels, all_preds, target_names=class_name)

    return acc, report, all_labels, all_probs

# --------------------------
# Confusion matrix function
# --------------------------
def compute_confusion_matrix(model, dataloader, device):
    """
    Computes the confusion matrix for a trained model. 
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN or Transformer model.
    dataloader : torch.utils.data.DataLoader
        Dataloader for the test dataset.
    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix of shape (num_classes, num_classes).
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return confusion_matrix(all_labels, all_preds) 

# ------------------------
# AUC calculation
# ------------------------
def compute_multiclass_auc(y_true, y_probs, num_classes):
    """
    Returns per-class AUC and macro-average AUC
    """
    y_bin = label_binarize(y_true ,classes=np.arange(num_classes))

    auc_per_class = {}

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc_per_class[i] = auc(fpr, tpr)

    macro_auc = np.mean(list(auc_per_class.values()))

    return auc_per_class, macro_auc

# ---------------------------
# PCA computation function 
# ---------------------------
def compute_pca(X, n_components=2):
    """
    DCompute PCA transformation for the given data.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix (num_samples x num_features)
    n_components : int
        Number of PCA components (default 2)
        
    Returns
    -------
    X_pca : np.ndarray
        Transformed data in PCA space (num_samples x n_components)
    pca : PCA object
        Fitted sklearn PCA object
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca
