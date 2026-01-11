import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# Configuration 
# ---------------------
DATA_PATH = "data/processed/spectra_preprocessed.npz"
NUM_SMAPLES_PER_CLASS = 10   # for visualization only
FILTER_LABELS = ["LP", "SP", "BP", "ND"]

# ----------------------
# Load dataset 
# ----------------------
data = np.load(DATA_PATH)
X = data["X_train"]      # (num_samples, 401)
y = data["y_train"]      # (nume_samples,)
wavelength = data["wavelength"]

# -----------------------
# Helper function 
# -----------------------
def plot_class_spectra(X, y, class_id, title): 
    plt.figure(figsize=(7, 4))
    idx = np.where(y == class_id)[0][:NUM_SMAPLES_PER_CLASS]

    for i in idx: 
        plt.plot(wavelength, X[i], alpha=0.7)
        
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Absorption")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------
# 1. Visualize spectra per filter
# ----------------------------------
for class_id, label in enumerate(FILTER_LABELS):
    plot_class_spectra(
        X, y, 
        class_id, 
        title=f"{label} Filter: Example Spectra"
    )

# ---------------------------------
# 2. Compare mean spectra
# ----------------------------------
plt.figure(figsize=(7, 4))
for class_id, label in enumerate(FILTER_LABELS):
    mean_spectrum = X[y == class_id].mean(axis=0)
    plt.plot(wavelength, mean_spectrum, label=label)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Absorption")
plt.title("Mean Spectrum per Filter Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------
# 3. Inspect intra-class variability
# -------------------------------------
plt.figure(figsize=(7, 4))
for class_id, label in enumerate(FILTER_LABELS):
    std_spectrum = X[y == class_id].std(axis=0)
    plt.plot(wavelength, std_spectrum, label=label)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Absorption")
plt.title("Intra-Class Spectral Variability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Visualization complete.")
