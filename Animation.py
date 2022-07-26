import matplotlib.pyplot as plt
import numpy as np

from Fourier import FrequencyDomain
from Wav.WavReader import WavReader, batch_data

max_freq = 8000
omega = np.arange(max_freq)

def norm(data):
    min_s, max_s =  data.min(), data.max()
    return (data - min_s) / (max_s - min_s)

reader = WavReader(f"Data/vowels/all/all.wav")

plt.ion()
fig, (ax1, ax2) = plt.subplots(2)

for data, start, end in batch_data(reader.get_data(), size=max_freq*2):
    freq_data = FrequencyDomain(data)
    amps = freq_data.get_amplitudes()[:max_freq]
    norm_amps = norm(amps)

    ax1.plot(data, linewidth=1, color="red")
    ax2.plot(norm_amps, linewidth=1, color="red")

    ax1.set_xlim([0, max_freq * 2])
    ax1.set_ylim([-5e3, 5e3])

    ax2.set_xlim([0, max_freq])
    ax2.set_ylim([0, 1])

    plt.draw()
    plt.pause(1)

    ax1.cla()
    ax2.cla()

plt.show(block=True)

print("done")