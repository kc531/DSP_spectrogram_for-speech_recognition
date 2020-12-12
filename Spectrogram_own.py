import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pandas import DataFrame
import seaborn

#############################################################################################################

def fft_plot(audio, sampling_rate) :
    n = len(audio)
    T = 1/sampling_rate
    yf = scipy.fft.fft(audio)
    xf = np.linspace(0.0, 1.0/(2.0*T), n//2)
    plt.figure()
    plt.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    plt.xlabel("Frequency -->")
    plt.ylabel("Magnitude")
    plt.show()
    
#############################################################################################################

def spectrogram(samples, sample_rate, stride_ms = 10.0, window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram
    
#############################################################################################################

file_path = "sample_audio.wav"
samples, sampling_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)

print("Number of Samples : ", len(samples))
print("Sampling Rate : ", sampling_rate)

duration_of_audio = len(samples)/sampling_rate
print("Duration of audio : ", duration_of_audio, "seconds");

plt.figure()
librosa.display.waveplot(y = samples, sr = sampling_rate)
plt.xlabel("Time (seconds) -->")
plt.ylabel("Amplitude")
plt.show()

fft_plot(samples, sampling_rate)

maximum_frequency = 8000

spectrogram_values = spectrogram(samples, sampling_rate, max_freq=maximum_frequency)

dimensions_of_spectrogram = np.shape(spectrogram_values)
print("dimensions of spectrogram_values : ",dimensions_of_spectrogram)

df = DataFrame(np.flipud(spectrogram_values), np.linspace(maximum_frequency, 0,dimensions_of_spectrogram[0]), np.linspace(0,(dimensions_of_spectrogram[1]-1)*10,dimensions_of_spectrogram[1]))
plt.figure()
seaborn.heatmap(df)
plt.show()