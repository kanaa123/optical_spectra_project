import numpy as np
import json
import os

# -------------------------------------------------
# Configuration
# -------------------------------------------------
OUTPUT_DIR = "data/dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES_PER_CLASS = 3000
WAVELENGTH_START = 400 #nm
WAVELENGTH_END = 800 #nm
N = 401 #number of wavelength points
NOISE_STD_RANGE = (0.002, 0.01)       # Gaussian noise
BASELINE_OFFSET_RANGE = (-0.02, 0.02)
BASELINE_SLOPE_RANGE = (-1e-4, 1e-4)

CLASSES = ["long-pass", "short-pass", "band-pass", "neutral-density"]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------------------------------
# Wavelength Grid
# -------------------------------------------------
wavelength = np.linspace(WAVELENGTH_START, WAVELENGTH_END, N)

# =========================
# Beer–Lambert Law
# A = ε(λ) · c · l
# =========================
def beer_lambert(epsilon, concentration, path_length):
    return epsilon * concentration * path_length

# -------------------------------------------------
# Spectral models
# -------------------------------------------------
def long_pass(wl, cutoff):
    return 1 / (1 + np.exp(-(wl - cutoff)/10))

def short_pass(wl, cutoff):
    return 1 / (1 + np.exp((wl - cutoff)/10))

def band_pass(wl, center, width):
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def neutral_density(wl):
    base_level = np.random.uniform(0.3, 0.9)
    ripple = np.random.normal(0, 0.01, size=wl.shape)
    return np.clip(base_level + ripple, 0, None)

# =========================
# Noise & Baseline
# =========================
def add_noise_and_baseline(absorbance):
    # Gaussian noise
    noise_std = np.random.uniform(*NOISE_STD_RANGE)
    noise = np.random.normal(0, noise_std, size=absorbance.shape)

    # Baseline offset
    offset = np.random.uniform(*BASELINE_OFFSET_RANGE)

    # Baseline slope
    slope = np.random.uniform(*BASELINE_SLOPE_RANGE)
    baseline = slope * (wavelength - wavelength.mean())

    return absorbance + noise + offset + baseline

# =========================
# Normalization
# =========================
def normalize_spectrum(spectrum):
    min_val = np.min(spectrum)
    max_val = np.max(spectrum)
    return (spectrum - min_val) / (max_val - min_val + 1e-8)

# -------------------------------------------------
# Dataset generation
# -------------------------------------------------
def generate_dataset():
    X = [] #input data
    y = [] #labels
    label_map = {cls: idx for idx, cls in enumerate(CLASSES)}

    for cls in CLASSES:
        for _ in range(NUM_SAMPLES_PER_CLASS):
            
            # Random physical parameters
            concentration = np.random.uniform(0.5, 2.0)
            path_length = np.random.uniform(0.5, 2.0)
            
            if cls == "long-pass":
                cutoff = np.random.uniform(500, 650)
                epsilon = long_pass(wavelength, cutoff)

            elif cls == "short-pass":
                cutoff = np.random.uniform(500, 650)
                epsilon = short_pass(wavelength, cutoff)

            elif cls == "band-pass":
                center = np.random.uniform(500, 700)
                width = np.random.uniform(20, 60)
                epsilon = band_pass(wavelength, center, width)

            else:  # neutral-density
                epsilon = neutral_density(wavelength)

            # Beer–Lambert absorbance
            absorbance = beer_lambert(epsilon, concentration, path_length)

            # Add noise & baseline
            absorbance = add_noise_and_baseline(absorbance)

            # Normalize
            absorbance = normalize_spectrum(absorbance)

            X.append(absorbance)
            y.append(label_map[cls])

    return np.array(X), np.array(y)

# =========================
# Run & Save
# =========================
if __name__ == "__main__":
    X, y = generate_dataset()

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    np.savez(
        os.path.join(OUTPUT_DIR, "synthetic_spectra_dataset.npz"),
        X=X,
        y=y,
        wavelength=wavelength
    )

    print("Dataset saved to:", OUTPUT_DIR)

# -------------------------------------------------
# Save metadata
# -------------------------------------------------
os.makedirs("data/metadata", exist_ok=True)

with open("data/metadata/class_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {i: CLASSES[i] for i in range(len(CLASSES))},
        f,
        indent=2
    )

with open("data/metadata/dataset_description.txt", "w", encoding="utf-8") as f:
    f.write(
        f"Wavelength range: {WAVELENGTH_START}–{WAVELENGTH_END} nm\n"
        f"Number of wavelength points: {N}\n"
        f"Classes: {CLASSES}\n"
        f"Samples per class: {NUM_SAMPLES_PER_CLASS}\n"
        f"Gaussian noise σ range: {NOISE_STD_RANGE}\n"
    )

# -------------------------------------------------
# Save small TXT preview for visualization
# -------------------------------------------------
preview_dir = "data/preview"
os.makedirs(preview_dir, exist_ok=True)

preview_file = os.path.join(preview_dir, "spectra_preview.txt")

with open(preview_file, "w", encoding="utf-8") as f:
    f.write("Preview of generated absorbance spectra\n")
    f.write("Format: wavelength (nm), absorbance\n\n")

    for i in range(2):  # ONLY 2 samples (keep it small)
        f.write(
            f"Spectrum {i}, Class {y[i]} "
            f"({CLASSES[y[i]]})\n"
        )
        for wl, val in zip(wavelength, X[i]):
            f.write(f"{wl:.2f}, {val:.6f}\n")
        f.write("\n")

# -------------------------------------------------
# Done
# -------------------------------------------------
print("Dataset created successfully.")
print("Main dataset: data/dataset/synthetic_spectra_dataset.npz")
print("Preview TXT : data/preview/spectra_preview.txt")

