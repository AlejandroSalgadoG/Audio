import matplotlib.pyplot as plt
import numpy as np
import os

from Fourier import Time2Freq
from Wav.WavReader import WavReader
from Wav.WavWriter import WavWriter

max_freq = 4000

omega = np.arange(max_freq)

def norm_rows(matrix):
    row_min, row_max =  matrix.min(axis=1, keepdims=True), matrix.max(axis=1, keepdims=True)
    return (matrix - row_min) / (row_max - row_min)

t_data = [WavReader(f"Data/vowels/a/{file_name}").get_data() for file_name in os.listdir("Data/vowels/a")]
freq_data = [Time2Freq.transform(sample) for sample in t_data]
amps = np.array([f_data.get_amplitudes()[:max_freq] for f_data in freq_data])
norm_amps = norm_rows(amps)

mean_amp = norm_amps.mean(axis=0)
std_amp = norm_amps.std(axis=0)

up_amp = mean_amp + std_amp
down_amp = mean_amp - std_amp

fig, ax = plt.subplots()

ax.plot(mean_amp, linewidth=0.25, color="blue")
ax.plot(norm_amps[0,:], linewidth=1, color="red")

ax.fill_between(omega, mean_amp, up_amp, color="blue", alpha=0.2)
ax.fill_between(omega, mean_amp, down_amp, color="blue", alpha=0.2)

ax.set_ylim([0, 1])

plt.show()

print("done")