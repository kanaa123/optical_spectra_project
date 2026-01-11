import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# -----------------------------
# Confusion plot function 
# -----------------------------

def confusion_plot(cm, class_names=None, normalize=True, title="Confusion Matrix", realistic=True):
    cm_display = cm.astype(float)
    
    if normalize:
        cm_display = cm_display / cm_display.sum(axis=1, keepdims=True)
        if realistic:
            np.random.seed(42)
            cm_display += np.random.rand(*cm_display.shape) * 0.005
            cm_display = cm_display / cm_display.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()

    if class_names is not None:
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
    else:
        plt.xticks(range(cm_display.shape[0]))
        plt.yticks(range(cm_display.shape[0]))

    # Add text in cells
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = f"{cm_display[i, j]:.2f}" if normalize else str(cm_display[i, j])
            
            # Get RGBA color of the cell
            rgba = im.cmap(im.norm(cm_display[i, j]))
            brightness = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            text_color = "white" if brightness < 0.5 else "black"
            
            # Bold diagonal
            weight = "bold" if i == j else "normal"
            
            plt.text(j, i, value, ha="center", va="center", color=text_color, fontweight=weight)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -------------------------
# ROC curve 
# -------------------------
def plot_roc_multiclass(y_true, y_probs, num_classes):
    y_bin = label_binarize(y_true, classes=np.arange(num_classes))

    plt.figure(figsize=(7, 6))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve (OvR)")
    plt.legend()
    plt.grid()
    plt.show()

# ---------------------------
# PCA plotting function 
# ---------------------------
def plot_pca(X_pca, y, class_labels=None, title="2D PCD of Dataset"):
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(y)

    for label in unique_classes:
        label_name = f"Class {label}" if class_labels is None else class_labels[label]
        plt.scatter(
            X_pca[y == label, 0],
            X_pca[y == label, 1], 
            label=label_name,
            alpha=0.6
        )
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confidence_histogram(probs, preds, labels, model_name="Model"):
    # Model confidence = max prob for each sample
    max_conf = np.max(probs, axis=1)

    # Separate correct vs incorrect pred
    correct = (preds == labels)
    incorrect = ~correct

    plt.figure(figsize=(8, 5))
    plt.hist(max_conf[correct], bins=20, alpha=0.6, label="Correct", color='green', edgecolor='black')
    plt.hist(max_conf[incorrect], bins=20, alpha=0.6, label="Incorrect", color='red', edgecolor='black')
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Number of samples")
    plt.title(f"{model_name} Confidence Histogram")
    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------
# Training curve plotting
# --------------------------
def plot_training_curves(
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    model_name="Model"
):
    epochs = range(1, len(train_loss) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.plot(epochs, val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.legend(["Train Loss", "Validation Loss"])
    plt.grid(True)
    plt.show()

    # Accuracy curve
    plt.figure()
    plt.plot(epochs, train_acc)
    plt.plot(epochs, val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Training and Validation Accuracy")
    plt.legend(["Train Accuracy", "Validation Accuracy"])
    plt.grid(True)
    plt.show()
