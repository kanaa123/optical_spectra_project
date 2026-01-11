import numpy as np

# -----------------------------
# Load your datasets
# -----------------------------
# Replace these with your actual dataset paths or variables
# Example: you might have saved them as .npz or .npy
data = np.load("data/processed/spectra_preprocessed.npz")
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# -----------------------------
# Step 1: Check number of samples
# -----------------------------
print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# -----------------------------
# Step 2: Check for sample overlap
# -----------------------------
# Convert each sample to a tuple so it can be added to a set
train_samples = set(map(tuple, X_train))
test_samples = set(map(tuple, X_test))

overlap = train_samples.intersection(test_samples)
print("Number of overlapping samples:", len(overlap))

if len(overlap) > 0:
    print("Warning: Data leakage detected! Some samples exist in both train and test.")
else:
    print("No overlap detected — train and test sets are separate.")

# -----------------------------
# Optional Step 3: Quick visual check
# -----------------------------
import matplotlib.pyplot as plt

plt.plot(X_train[0], label='Train sample 0')
plt.plot(X_test[0], label='Test sample 0')
plt.legend()
plt.title("Visual check of first train/test sample")
plt.show()

