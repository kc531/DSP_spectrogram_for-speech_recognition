import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import librosa
import numpy as np
file_path = "sample_audio.wav"
samples, sampling_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)
f, t, Zxx = signal.stft(samples, fs=sampling_rate)
plt.pcolormesh(t, f, np.abs(Zxx))
plt.xlabel("Time (seconds) -->")
plt.ylabel("Frequency")
plt.show()