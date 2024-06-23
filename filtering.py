import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
import pickle

from scipy.fft import fft, rfft
from scipy.fft import fftfreq, rfftfreq

# Load the saved probabilities
with open('./prob_arrays/97s.pkl', 'rb') as f:
    probs_array = pickle.load(f)

# Extract probs
sitting_probs = probs_array[:, 0]
standing_probs = probs_array[:, 1]
frame_numbers = np.arange(len(sitting_probs)) # x axis

# Fourier
def fft_smooth(data, cutoff):
    ft = fft(data)
    frequencies = np.fft.fftfreq(len(data), d=1)  # d=1 means we assume a unit sampling rate
    ft[np.abs(frequencies) > cutoff] = 0
    smoothed_data = np.fft.ifft(ft).real
    return frequencies

sitting_fft = fft_smooth(sitting_probs, cutoff=0.05)

# Plot the original and filtered signals
frames = range(len(probs_array))
plt.figure(figsize=(15, 5))

plt.plot(frames, sitting_probs, label="'sitting' original", alpha=0.5)
plt.plot(frames, np.abs(sitting_fft), label="Fourier", color='blue')

plt.xlabel('Frame Number')
plt.ylabel('Probability')
plt.title('Probabilities of Captions Over Frames (Filtered and Smoothed)')
plt.legend()
plt.grid(True)
plt.show()



#plt.plot(frames, probs_array[:, 1], label="'standing' original", alpha=0.5)
#plt.plot(frames, smoothed_standing, label="'standing' smoothed", linewidth=2)
#plt.plot(frames, smoothed_sitting, label="'sitting' smoothed", linewidth=2)



# # Filter settings
# cutoff = 0.05  # desired cutoff frequency of the filter, Hz
# fs = 1.0  # sample rate, Hz
# order = 2  # filter order

# # Apply inverse fourier
# sitting_fft = 

# # Apply low-pass filter
# filtered_sitting = butter_lowpass_filter(probs_array[:, 0], cutoff, fs, order)
# filtered_standing = butter_lowpass_filter(probs_array[:, 1], cutoff, fs, order)

# # Apply Savitzky-Golay filter for smoothing
# smoothed_sitting = savgol_filter(filtered_sitting, window_length=51, polyorder=3)
# smoothed_standing = savgol_filter(filtered_standing, window_length=51, polyorder=3)

# # Butterworth low-pass filter
# def butter_lowpass_filter(data, cutoff, fs, order=5):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y