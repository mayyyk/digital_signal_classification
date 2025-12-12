# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
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

    # Norm L2 (Euclidean) - square root of the sum of squares (each number in a2_coeffs in squared, then all are summed up and we take the square root of this sum)
    # Signal energy in low-frequency band
    # Reactive to big devations
    norm_L2 = np.linalg.norm(a2_coeffs, ord=2)

    # Norm L-inf - maximum error
    # Worst-case scenario
    norm_Linf = np.linalg.norm(a2_coeffs, ord=np.inf)

    return [norm_L1, norm_L2, norm_Linf]

# %%

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
                    print(f"File '{filename}' does not match the classification pattern.")
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

# %%

# DATA VISUALIZATION

df = pd.DataFrame(X, columns=['L1_Norm', 'L2_Norm', 'Linf_Norm'])
df['Class'] = ['Gear' if label == 1 else 'Fan' for label in y]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='L2_Norm', hue='Class', element="step", stat="density", common_norm=False)
plt.title('Does L2 Norm differ between classes?')
plt.xlabel('L2 Norm Value (Energy a2)')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='L2_Norm', y='Linf_Norm', hue='Class', style='Class', s=100, alpha=0.7)
plt.title('Feature Space: Energy vs. Maximum Peak')
plt.xlabel('L2 Norm (Energy)')
plt.ylabel('Linf Norm (Max Amplitude)')

plt.tight_layout()
# plt.show()

print("\n--- Sample Data (Features + Label) ---")
print(df.head(5))

# %%
# NEURAL NETWORK

# Scaling data to normal distribution, mean = 0, std = 1, otherwise "gradient descent" doesn't work well

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Training and test separation, random_state guarantees repetitive randomness
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=16)

# Building model
model = Sequential()

# Hidden layer
model.add(Dense(10, activation='relu', input_shape=(3,), name="Hidden_Layer")) # 10 neurons, 3 input features for each

# Output layer
model.add(Dense(1, activation='sigmoid', name="Probability_Output")) # 1 output neuron because of binary classification: 1 or 0

# Model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# adam - actualizes weights, adjusts learning pace
# binary_crossentropy - calculates error
# accuracy metrics - counts % of correct answers

# Training
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# %%

plt.figure(figsize=(12, 5))

# Loss Plot - should decrease
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curve: Error (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Error (Binary Crossentropy)')
plt.legend()
plt.grid(True)

# Accuracy Plot - should increase
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Acc')
plt.plot(history.history['val_accuracy'], label='Validation Acc')
plt.title('Learning Curve: Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (0-1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# %%

# FINAL EVALUATION

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Model predictions

predictions = model.predict(X_test) 
print(f"{'True Class':<20} | {'Network Prediction':<20} | {'Conclusion'}")
print("-" * 60)

for i in range(max(10, len(X_test))):
    true_label = "Gear (1)" if y_test[i] == 1 else "Fan (0)"
    
    prob = predictions[i][0]
    
    # Sigmoid returns 0-1. Threshold is set to 0.5
    predicted_label = "Gear (1)" if prob > 0.5 else "Fan (0)"
    
    match = "OK" if (y_test[i] == (1 if prob > 0.5 else 0)) else "ERROR"
    
    print(f"{true_label:<20} | {prob:.4f} ({predicted_label}) | {match}")# %%
