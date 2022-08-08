import os
import numpy as np
import matplotlib.pyplot as plt

from Fourier import MultiSample
from Wav.WavReader import WavReader


def plot_amps(axis, omega, amps_data):
    mean_amps = amps_data.mean(axis=0)
    std_amps = amps_data.std(axis=0)

    up_amps = mean_amps + std_amps
    down_amps = np.maximum(mean_amps - std_amps, 0)

    axis.plot(mean_amps, linewidth=0.5, color="blue")
    axis.fill_between(omega, mean_amps, up_amps, color="blue", alpha=0.2)
    axis.fill_between(omega, mean_amps, down_amps, color="blue", alpha=0.2)

    axis.set_xlim([0, omega.size])


fig, axis = plt.subplot_mosaic([
    ["a"],
    ["e"],
    ["i"],
    ["o"],
    ["u"],
])

max_freq = 2000
omega = np.arange(max_freq)

samples = MultiSample()

for vowel in ["a", "e", "i", "o", "u"]:

    for file_name in os.listdir(f"Data/vowels/{vowel}"):
        samples.add_from_time_data( vowel, WavReader(f"Data/vowels/{vowel}/{file_name}").get_data() )

    plot_amps( axis[vowel], omega, samples.get_amplitudes(vowel, min_freq=0, max_freq=max_freq) )

plt.show()

print("done")