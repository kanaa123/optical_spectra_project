import os
import numpy as np
from sklearn.model_selection import train_test_split

# -------------------------------
# Configuration parameters
# --------------------------------

DATA_PATH = "data/dataset/synthetic_spectra_dataset.npz"

# Noise augmentation 
ADD_NOISE = True
NOISE_MIN = 0.002
NOISE_MAX = 0.01

# Dataset split ratio 
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# -----------------------
# Utility functions
# -----------------------

def normalize_spectra(spectra):
    """
    Normalize each spectrum to [0, 1]
    """

    spectra_min = spectra.min(axis = 1, keepdims = True)
    spectra_max = spectra.max(axis = 1, keepdims = True)
    normalized = (spectra - spectra_min) / (spectra_max - spectra_min + 1e-8)
    return normalized

def add_gaussian_noise(spectra, min_std = 0.002, max_std = 0.01):
    """
    Add Gaussian noise with random sigma for each spectrum
    """
    noisy = np.copy(spectra)
    for i in range(noisy.shape[0]):
        sigma = np.random.uniform(min_std, max_std)
        noisy[i] += np.random.normal(0, sigma, size = noisy.shape[1])
        return np.clip(noisy, 0.0, 0.1) #keep in valid range 

# ----------------------------
# Load dataset
# ----------------------------
print("Loading dataset...")
data = np.load(DATA_PATH)
X = data["X"]    #spectra
y = data["y"]    #Labels
wavelength = data["wavelength"]

print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")

# ----------------------------------------
# Remove duplicates to prevent overlap
# ----------------------------------------
_, unique_idx = np.unique(X, axis=0, return_index=True)
X = X[unique_idx]
y = y[unique_idx]

print(f"Dataset after removing duplicates: {X.shape}")

# --------------------------
# Normalize
# --------------------------
X = normalize_spectra(X)
print("Normalization done.")

# -------------------------
# Add noise
# -------------------------
if ADD_NOISE:
    X = add_gaussian_noise(X, NOISE_MIN, NOISE_MAX)
    print(f"Added Gaussian noise (σ in [{NOISE_MIN}, {NOISE_MAX}]).")

# -------------------------
# Split dataset
# -------------------------
# train + temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED, stratify=y
)

# val and test from temp
val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(1 - val_ratio_adjusted), random_state=RANDOM_SEED, stratify=y_temp
)

print("Datase split")
print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# ---------------------------------
# Find and delete overlaping data
# ---------------------------------
def find_overlaps(X_train, X_test, tol=1e-6):
    """
    Returns indices of test samples that are already in train
    tol: tolerance for floating-point comparison
    """
    overlap_idx = []
    for i, x_test in enumerate(X_test):
        # Compare with all train samples
        if np.any(np.all(np.isclose(X_train, x_test, atol=tol), axis=1)):
            overlap_idx.append(i)
    return overlap_idx

# Check overlaps
overlap_idx = find_overlaps(X_train, X_test, tol=1e-6)
print("Number of overlapping samples:", len(overlap_idx))

# Remoce overlapping test samples
if len(overlap_idx) > 0:
    print("Removing overlapping samples from test set...")
    X_test = np.delete(X_test, overlap_idx, axis=0)
    y_test = np.delete(y_test, overlap_idx, axis=0)

print("New test set shape:", X_test.shape)

# -----------------------
# Verify no overlaps 
# -----------------------
train_samples = set(map(tuple, X_train))
test_samples = set(map(tuple, X_test))
overlap = train_samples.intersection(test_samples)
print("Number of overlapping samples between train and test:", len(overlap))
if len(overlap) > 0:
    print("Warning! Overlap detected. Please check dataset.")
else:
    print("No overlap detected. Safe to proceed.")

# -------------------------------
# Save preprocessed dataset
# -------------------------------
SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

np.savez(
    os.path.join(SAVE_DIR, "spectra_preprocessed.npz"),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test = y_test,
    wavelength=wavelength
)

print("Preprocessed dataset saved to: ", os.path.join(SAVE_DIR, "spectra_preprocessed.npz"))