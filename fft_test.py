from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

def sin_generator(amp, vertical_shift, phase_shift, sec, f, fs = 50):
    # time = np.linspace(0, sec, sec * fs, endpoint=False)
    time = np.arange(0, sec * fs) * (1/fs)
    signal = amp * np.sin(2 * np.pi * time * f + phase_shift) + vertical_shift
    return time, signal

def plot_signal(signal, time, sec, fs = 50):
    print(time.shape)
    time = np.arange(0, sec * fs) * (1/fs)
    print(time.shape)

    # Perform Fast Fourier Transform (FFT)
    fft_result = fft(signal)

    # Calculate frequencies
    freq = fftfreq(fs * sec, 1 / fs)
    # Plot the signal and its FFT
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(time, signal)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Signal')

    axs[1].plot(freq[:(fs*sec)//2], np.abs(fft_result[:(fs*sec)//2]))
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')

    plt.show()
# Generate a sample signal
# Define signal parameters
# amp, vertical_shift, phase_shift, sec, f, fs = 50
time_1, sin_1 = sin_generator(1, 0, 0, 10, 10)
plot_signal(sin_1, time_1, 10)

time_2, sin_2 = sin_generator(2.3, 0, 0, 4, 7)
plot_signal(sin_2, time_2, 4)
sin_3 = np.hstack((sin_1, sin_2))
print(sin_3.shape)
plot_signal(sin_3, time_1, 10 + 4)