import os

import matplotlib.pyplot as plt
import numpy as np
import pywt
import tensorflow as tf
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

WAVELET_NAME = "db2"
DECOMPOSITION_LEVEL = 2
DATA_FOLDER = "signal_files"
SEGMENT_SIZE = 1024

# Scalable
CLASS_MAPPING = {
    "fan": 0,
    "gear": 1, 
}

def extract_features_from_segment(segment):
    """
    Engineers raw signal into features using Wavelet Transform.
    Returns: [L1_norm, L2_norm, Max_norm] of the a2 coefficient.
    """

    # DECOMPOSITION
    coeffs = pywt.wavedec(segment, WAVELET_NAME, level=DECOMPOSITION_LEVEL)

    a2_coeffs = coeffs[0]

    # NORMS

    # Norm L1 (Manhattan) - sum of absolute values
    # Total error cost, all errors are treated linearly
    norm_L1 = np.linalg.norm(a2_coeffs, ord=1)

    # Norm L2 (Euclidean) - square root of the sum of squares
    # Error Energy, reactive to big devations
    norm_L2 = np.linalg.norm(a2_coeffs, ord=2)

    # Norm L-inf - maximum error
    # Worst-case scenario
    norm_Linf = np.linalg.norm(a2_coeffs, ord=np.inf)

    return [norm_L1, norm_L2, norm_Linf]


# DATA PREPARATION

features = []
labels = []

if not os.path.exists(DATA_FOLDER):
    print(f"Error: Folder '{DATA_FOLDER}' not found.")

    print("Generating synthetic dummy data for demonstration...")
    for _ in range(50):
        # Fake 'Fan' (lower energy)
        dummy_sig = np.random.normal(0, 0.1, SEGMENT_SIZE)
        features.append(extract_features_from_segment(dummy_sig))
        labels.append(0)
        # Fake 'Gear' (higher energy/transients)
        dummy_sig = np.random.normal(0, 0.5, SEGMENT_SIZE)
        features.append(extract_features_from_segment(dummy_sig))
        labels.append(1)
else:
    print(f"Loading files from {DATA_FOLDER}...")
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".wav"):
            filepath = os.path.join(DATA_FOLDER, filename)
            try:
                fs, signal = wavfile.read(filepath)
                
                if len(signal.shape) > 1:
                    signal = signal[:, 0] # change stereo to mono 

                found_label = None

                for keyword, class_id in CLASS_MAPPING.items():
                    if keyword in filename.lower():
                        found_label = class_id
                        break # stop looking after keyword match
                
                if found_label is None:
                    continue # skip unknow file
                
                num_segments = len(signal) // SEGMENT_SIZE
                for i in range(num_segments):
                    seg = signal[i*SEGMENT_SIZE:(i+1)*SEGMENT_SIZE] # slicing excludes end index
                    feats = extract_features_from_segment(seg)
                    features.append(feats)
                    labels.append(found_label)


            except Exception as e:
                print(f"Error processing {filename}: {e}")

X = np.array(features)
y = np.array(labels)

print(f"Dataset created. Samples: {X.shape[0]}, Features per sample: {X.shape[1]}")

