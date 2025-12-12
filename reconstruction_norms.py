import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, signal = wavfile.read("./signal_files/gear1.wav")
wavelet_name = 'db2'
level = 4

# DECOMPOSITION

coeffs = pywt.wavedec(signal, wavelet_name, level=level)

cA4 = coeffs[0]
cD4 = coeffs[1]
cD3 = coeffs[2]
cD2 = coeffs[3]
cD1 = coeffs[4]

print(len(signal))
print(len(cD1), len(cD2)) # Each next coefficient level gets halfed in size

# RECONSTRUCTION

signal_rec = pywt.waverec(coeffs, wavelet_name)

# NORMS

error_vector = signal - signal_rec[:len(signal)] # equal sizes

# Norm L1 (Manhattan) - sum of absolute values
# Total error cost, all errors are treated linearly
norm_L1 = np.linalg.norm(error_vector, ord=1)

# Norm L2 (Euclidean) - square root of the sum of squares
# Error Energy, reactive to big devations
norm_L2 = np.linalg.norm(error_vector, ord=2)

# Norm L-inf - maximum error
# Worst-case scenario
norm_Linf = np.linalg.norm(error_vector, ord=np.inf)

print(f"Norm L1:   {norm_L1:.4f}")
print(f"Norm L2: {norm_L2:.4f}")
print(f"Norm Linf:    {norm_Linf:.4f}")

# Wizualizacja dla zrozumienia detali
plt.figure(figsize=(10, 6))
plt.subplot(4, 1, 1)
plt.plot(signal)
plt.title("Original signal")
plt.subplot(4, 1, 2)
plt.plot(cA4)
plt.title(f"A4 approximation ({wavelet_name}) - feature extraction")
plt.subplot(4, 1, 3)
plt.plot(cD1)
plt.title("D1 details - noise")
plt.subplot(4, 1, 4)
plt.plot(error_vector)
plt.title("Error vector")
plt.tight_layout()
plt.show()


