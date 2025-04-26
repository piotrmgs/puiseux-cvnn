# Copyright (c) 2025 Piotr Migus
# This code is licensed under the MIT License.
# See the LICENSE file in the repository root for full license information.
"""
This script performs the following steps:
- Bandpass filtering
- Segmentation and vectorization
- Normalization
- Train-test splitting
- A variety of signal and data visualizations, including:
    * Class distribution
    * Segment comparison
    * Hilbert transform
    * Spectrogram analysis
    * Correlation matrix
    * t-SNE projection
"""

import os
import wfdb
import wfdb.processing  # optional: for resampling operations
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import torch

# ---------------------------
# CONFIGURATION AND CONSTANTS
# ---------------------------

# Signal and segmentation parameters
FS = 360                # Sampling frequency [Hz]
WINDOW_SIZE = 128       # Segment length in samples
PRE_SAMPLES = 50        # Number of samples before the R peak (R peak is at PRE_SAMPLES within each segment)

# Bandpass filter parameters (low- and high-pass cutoff)
LOWCUT = 0.5            # Lower cutoff frequency [Hz] (removes baseline wander)
HIGHCUT = 40.0          # Upper cutoff frequency [Hz] (removes high-frequency noise)
ORDER = 2               # Filter order

# Paths – adjust to your local environment
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'mit-bih')
model_path = os.path.abspath(model_path)
DATA_FOLDER = os.path.abspath(model_path)  # Folder containing raw MIT-BIH records (no file extensions)
RECORD_NAMES = ["100", "101", "102", "103", "104", "105", "106", "107"]        # List of record IDs to be processed
OUTPUT_FOLDER = os.path.join(current_dir)       # Folder for saving visualizations

# Data split parameters
TEST_SIZE = 0.2         # Proportion of data to be used as the test set

# Ensure the output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# ---------------------------
# FILTERING FUNCTIONS
# ---------------------------

def bandpass_filter(signal: np.ndarray, fs: int, lowcut: float, highcut: float, order: int = 2) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter to a 1D signal.

    The filter removes components below `lowcut` (e.g., baseline wander)
    and above `highcut` (e.g., high-frequency noise).

    Parameters:
        signal (np.ndarray): 1D input signal
        fs (int): Sampling frequency in Hz
        lowcut (float): Lower cutoff frequency in Hz
        highcut (float): Upper cutoff frequency in Hz
        order (int): Filter order

    Returns:
        np.ndarray: Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


# ---------------------------
# RECORD PROCESSING
# ---------------------------

def process_record(record_name: str, folder: str, window_size: int = WINDOW_SIZE, pre_samples: int = PRE_SAMPLES, fs: int = FS):
    """
    Loads an MIT-BIH record and corresponding annotations. Applies bandpass filtering
    to all channels. For each heartbeat labeled 'N' (normal) or 'V' (premature ventricular contraction),
    a fixed-length segment is extracted and transformed using the Hilbert transform.
    
    The resulting features include both the real and imaginary parts of the analytic signal.

    Parameters:
        record_name (str): Record ID (e.g., "100")
        folder (str): Path to the folder containing the MIT-BIH records
        window_size (int): Length of each segment in samples
        pre_samples (int): Number of samples preceding the R peak (R peak is at position `pre_samples` in the segment)
        fs (int): Sampling frequency in Hz

    Returns:
        samples (List[np.ndarray]): List of feature vectors (each has shape [num_channels * 2 * window_size])
        labels (List[int]): Corresponding class labels (0 for 'N', 1 for 'V')
    """
    record_path = os.path.join(folder, record_name)
    
    # Check for existence of the required data files (.dat, .hea, .atr)
    if not os.path.exists(record_path + ".dat"):
        logging.error("Missing data file for record %s", record_name)
        return [], []
    
    try:
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, extension='atr')
    except Exception as e:
        logging.error("Failed to load record %s: %s", record_name, e)
        return [], []
    
    # Signal shape: (num_samples, num_channels)
    signals = record.p_signal

    # Apply bandpass filtering to each channel (vectorized)
    try:
        filtered_signals = np.apply_along_axis(
            lambda x: bandpass_filter(x, fs=fs, lowcut=LOWCUT, highcut=HIGHCUT, order=ORDER),
            axis=0, arr=signals
        )
    except Exception as e:
        logging.error("Filtering error for record %s: %s", record_name, e)
        return [], []
    
    samples = []
    labels = []
    
    # Process beat annotations; only 'N' (normal) and 'V' (PVC) are used
    for i, beat_idx in enumerate(ann.sample):
        symbol = ann.symbol[i]
        if symbol not in ['N', 'V']:
            continue  # skip other beat types
        
        # Ensure the segment is within signal bounds
        if beat_idx - pre_samples < 0 or (beat_idx - pre_samples + window_size) > filtered_signals.shape[0]:
            continue
        
        # Extract segment for all channels
        seg = filtered_signals[beat_idx - pre_samples: beat_idx - pre_samples + window_size, :]
        
        # Compute Hilbert transform for each channel and concatenate real + imaginary parts
        channel_features = []
        for ch in range(seg.shape[1]):
            analytic_signal = hilbert(seg[:, ch])
            channel_features.append(analytic_signal.real)
            channel_features.append(analytic_signal.imag)
        
        sample = np.concatenate(channel_features)
        samples.append(sample)
        
        # Assign label: 0 = normal ('N'), 1 = PVC ('V')
        label = 0 if symbol == 'N' else 1
        labels.append(label)
    
    return samples, labels

def load_mitbih_data(folder: str, record_names: list, window_size: int = WINDOW_SIZE, pre_samples: int = PRE_SAMPLES, fs: int = FS):
    """
    Loads and aggregates data from multiple MIT-BIH records.
    Each record is processed independently and progress is tracked.

    Parameters:
        folder (str): Path to the folder containing the MIT-BIH data
        record_names (List[str]): List of record IDs (e.g., ["100", "101", ...])
        window_size (int): Segment length in samples
        pre_samples (int): Number of samples preceding the R peak
        fs (int): Sampling frequency in Hz

    Returns:
        X (np.ndarray): Feature matrix of shape (num_samples, num_features)
        y (np.ndarray): Label vector of shape (num_samples,)
    """
    all_samples = []
    all_labels = []
    
    for rec in tqdm(record_names, desc="Processing records"):
        logging.info("Processing record %s...", rec)
        samples, labels = process_record(rec, folder, window_size, pre_samples, fs)
        all_samples.extend(samples)
        all_labels.extend(labels)
    
    X = np.array(all_samples)
    y = np.array(all_labels)
    return X, y


# ---------------------------
# VISUALIZATION FUNCTIONS
# ---------------------------

def visualize_class_distribution(labels: np.ndarray, output_folder: str):
    """
    Plots and saves a histogram showing the class distribution of 'N' and 'V' beats.

    Parameters:
        labels (np.ndarray): Vector of class labels
        output_folder (str): Directory where the plot will be saved
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_names = {0: 'Normal (N)', 1: 'PVC (V)'}
    labels_str = [class_names.get(x, str(x)) for x in unique]
    
    plt.figure(figsize=(6, 4))
    plt.bar(labels_str, counts, color=['skyblue', 'salmon'])
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution in the Dataset")
    plt.tight_layout()
    output_path = os.path.join(output_folder, "class_distribution.png")
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved class distribution plot to: %s", output_path)


def visualize_example_segments(record_name: str, folder: str, pre_samples: int, window_size: int, fs: int, output_folder: str):
    """
    Generates comparative visualizations of example signal segments for classes 'N' and 'V', including:
        - Original vs. filtered signal with R-peak marker
        - Hilbert transform envelope
        - Spectrogram of the filtered signal

    Parameters:
        record_name (str): MIT-BIH record ID (e.g., "100")
        folder (str): Path to the dataset folder
        pre_samples (int): Number of samples before the R-peak
        window_size (int): Length of the segment in samples
        fs (int): Sampling frequency in Hz
        output_folder (str): Directory to save visualizations
    """
    record_path = os.path.join(folder, record_name)
    try:
        record = wfdb.rdrecord(record_path)
        ann = wfdb.rdann(record_path, extension='atr')
    except Exception as e:
        logging.error("Failed to load record %s for visualization: %s", record_name, e)
        return

    signals = record.p_signal  # Raw (unfiltered) signals
    
    # Apply bandpass filtering to all channels
    filtered_signals = np.apply_along_axis(
        lambda x: bandpass_filter(x, fs=fs, lowcut=LOWCUT, highcut=HIGHCUT, order=ORDER),
        axis=0, arr=signals
    )
    
    # Identify example indices for 'N' and 'V' classes
    example_indices = {'N': None, 'V': None}
    for i, beat_idx in enumerate(ann.sample):
        symbol = ann.symbol[i]
        if symbol in example_indices and example_indices[symbol] is None:
            if beat_idx - pre_samples >= 0 and (beat_idx - pre_samples + window_size) <= len(filtered_signals):
                example_indices[symbol] = beat_idx
        if all(v is not None for v in example_indices.values()):
            break

    for cls, beat_idx in example_indices.items():
        if beat_idx is None:
            logging.warning("No representative beat found for class %s in record %s", cls, record_name)
            continue
        
        # Extract raw and filtered segments (first channel only)
        raw_segment = signals[beat_idx - pre_samples: beat_idx - pre_samples + window_size, 0]
        filt_segment = filtered_signals[beat_idx - pre_samples: beat_idx - pre_samples + window_size, 0]
        analytic_signal = hilbert(filt_segment)
        envelope = np.abs(analytic_signal)
        
        # Plot 1: Raw vs. Filtered signal with R-peak marker
        plt.figure(figsize=(10, 5))
        plt.plot(raw_segment, label="Raw Signal", alpha=0.7)
        plt.plot(filt_segment, label="Filtered Signal", alpha=0.7)
        plt.axvline(pre_samples, color='red', linestyle='--', label="R Peak")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.title(f"Raw vs. Filtered Signal – Class {cls} (Record {record_name})")
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_folder, f"signal_comparison_{record_name}_class_{cls}.png")
        plt.savefig(output_path)
        plt.close()
        logging.info("Saved signal comparison plot to: %s", output_path)
        
        # Plot 2: Hilbert transform envelope
        plt.figure(figsize=(10, 5))
        plt.plot(filt_segment, label="Filtered Signal", alpha=0.7)
        plt.plot(envelope, label="Hilbert Envelope", alpha=0.7)
        plt.axvline(pre_samples, color='red', linestyle='--', label="R Peak")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.title(f"Hilbert Transform – Class {cls} (Record {record_name})")
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_folder, f"hilbert_transform_{record_name}_class_{cls}.png")
        plt.savefig(output_path)
        plt.close()
        logging.info("Saved Hilbert transform plot to: %s", output_path)
        
        # Plot 3: Spectrogram of the filtered segment
        plt.figure(figsize=(10, 5))
        plt.specgram(filt_segment, Fs=fs, cmap='viridis')
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title(f"Spectrogram – Class {cls} (Record {record_name})")
        plt.colorbar(label="Intensity")
        plt.tight_layout()
        output_path = os.path.join(output_folder, f"spectrogram_{record_name}_class_{cls}.png")
        plt.savefig(output_path)
        plt.close()
        logging.info("Saved spectrogram to: %s", output_path)


def visualize_correlation_matrix(X: np.ndarray, output_folder: str, num_features: int = 50):
    """
    Computes and visualizes the correlation matrix of the first `num_features` extracted features.
    This helps to identify potential redundancy or high correlation between features.

    Parameters:
        X (np.ndarray): Feature matrix of shape (num_samples, num_features)
        output_folder (str): Directory where the plot will be saved
        num_features (int): Number of leading features to include in the correlation analysis
    """
    X_subset = X[:, :num_features] if X.shape[1] > num_features else X
    corr_matrix = np.corrcoef(X_subset.T)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_path = os.path.join(output_folder, "correlation_matrix.png")
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved correlation matrix to: %s", output_path)


def visualize_tsne(X: np.ndarray, y: np.ndarray, output_folder: str):
    """
    Applies t-SNE to reduce the dimensionality of the feature space to 2D
    and visualizes the embedded points, colored by class labels.

    Parameters:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Corresponding label vector
        output_folder (str): Directory where the t-SNE plot will be saved
    """
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], alpha=0.7, label='Normal (N)', color='skyblue')
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], alpha=0.7, label='PVC (V)', color='salmon')
    plt.title("t-SNE Visualization of Feature Space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_folder, "tsne_visualization.png")
    plt.savefig(output_path)
    plt.close()
    logging.info("Saved t-SNE visualization to: %s", output_path)


# ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":
    # 1. Load MIT-BIH data
    logging.info("Starting data loading...")
    X, y = load_mitbih_data(DATA_FOLDER, RECORD_NAMES, WINDOW_SIZE, PRE_SAMPLES, FS)
    if X.size == 0 or y.size == 0:
        logging.error("No data was loaded. Please check your configuration and file paths.")
        exit(1)
    
    logging.info("Data loading complete. Total samples: %d", X.shape[0])
    
    # 2. Normalize the data
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    logging.info("Feature normalization complete.")
    
    # 3. Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=TEST_SIZE, stratify=y, random_state=42
    )
    logging.info("Data split: %d training samples, %d test samples", X_train.shape[0], X_test.shape[0])
    
    # 4. Convert to PyTorch tensors (optional, for ML pipeline integration)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    logging.info("Data successfully converted to PyTorch tensors.")
    
    # 5. Visualizations:
    # a) Class distribution
    visualize_class_distribution(y, OUTPUT_FOLDER)
    
    # b) Signal segment comparison, Hilbert transform, and spectrogram
    #    Default visualization performed on record "100" (you may repeat for others)
    visualize_example_segments("100", DATA_FOLDER, PRE_SAMPLES, WINDOW_SIZE, FS, OUTPUT_FOLDER)
    
    # c) Correlation matrix (for first 50 features)
    visualize_correlation_matrix(X, OUTPUT_FOLDER, num_features=50)
    
    # d) t-SNE feature space projection
    visualize_tsne(X, y, OUTPUT_FOLDER)
    
    logging.info("All visualizations saved to folder: %s", OUTPUT_FOLDER)
