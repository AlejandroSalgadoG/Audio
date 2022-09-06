import matplotlib.pyplot as plt
import numpy as np

from Fourier import BatchData, Time2Freq
from Wav.WavReader import WavReader

window_size = 500

window = np.hanning(window_size)
t_data = WavReader(f"Data/vowels/a/a.wav").get_data()

plt.ion()

fig, axis = plt.subplot_mosaic([
    ["signal", "signal"],
    ["batch", "freq_batch"],
    ["window", "freq_window"],
])

axis["signal"].set_xlim([0, t_data.n_samples])
axis["signal"].set_ylim([-5e3, 6e3])

axis["signal"].plot(t_data.data, color="blue", linewidth=1)

for batch_data in t_data.batch(size=window_size, increment=window_size//4):
    window_data = BatchData(batch_data.data * window, batch_data.start, batch_data.end)

    freq_orig = Time2Freq.transform(batch_data)
    freq_wind = Time2Freq.transform(window_data)

    axis["signal"].plot([batch_data.start, batch_data.end], [5.5e3, 5.5e3], color="red", linewidth=1)

    axis["batch"].plot(batch_data.data, linewidth=1, color="red")
    axis["window"].plot(window_data.data, linewidth=1, color="red")

    axis["freq_batch"].plot(freq_orig.get_amplitudes(), linewidth=1, color="red")
    axis["freq_window"].plot(freq_wind.get_amplitudes(), linewidth=1, color="red")

    axis["batch"].set_xlim([0, window_size-1])
    axis["batch"].set_ylim([-5e3, 6e3])

    axis["window"].set_xlim([0, window_size-1])
    axis["window"].set_ylim([-5e3, 6e3])

    axis["freq_batch"].set_xlim([0, freq_orig.nyquist])
    axis["freq_window"].set_xlim([0, freq_wind.nyquist])

    plt.draw()
    plt.waitforbuttonpress()

    axis["signal"].plot([batch_data.start, batch_data.end], [5.5e3, 5.5e3], color="white", linewidth=1)

    axis["batch"].cla()
    axis["window"].cla()
    axis["freq_batch"].cla()
    axis["freq_window"].cla()

plt.show(block=True)

print("done")