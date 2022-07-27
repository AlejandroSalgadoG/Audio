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
data = reader.get_data()

split1 = batch_data(data, size=max_freq*2)
split2 = batch_data(data, size=max_freq*2, offset=max_freq)

plt.ion()

fig, axis = plt.subplot_mosaic([
    ["signal",   "signal"  ],
    ["sub1",     "sub2"    ],
    ["fourier1", "fourier2"],
])

axis["signal"].plot(data, linewidth=1, color="blue")
axis["signal"].set_xlim([0, reader.n_data_samples])
axis["signal"].set_ylim([-6e3, 6e3])

for (batch1, start1, end1), (batch2, start2, end2) in zip(split1, split2):
    freq_data1 = FrequencyDomain(batch1)
    amps1 = freq_data1.get_amplitudes()[:max_freq]
    norm_amps1 = norm(amps1)

    freq_data2 = FrequencyDomain(batch2)
    amps2 = freq_data2.get_amplitudes()[:max_freq]
    norm_amps2 = norm(amps2)

    axis["signal"].plot([start1, end1], [5e3, 5e3], color="red", linewidth=1)
    axis["signal"].plot([start2, end2], [4.5e3, 4.5e3], color="green", linewidth=1)

    axis["sub1"].plot(batch1, linewidth=1, color="red")
    axis["fourier1"].plot(norm_amps1, linewidth=1, color="red")

    axis["sub2"].plot(batch2, linewidth=1, color="green")
    axis["fourier2"].plot(norm_amps2, linewidth=1, color="green")

    axis["sub1"].set_xlim([0, max_freq * 2])
    axis["sub1"].set_ylim([-5e3, 5e3])

    axis["sub2"].set_xlim([0, max_freq * 2])
    axis["sub2"].set_ylim([-5e3, 5e3])

    axis["fourier1"].set_xlim([0, max_freq])
    axis["fourier1"].set_ylim([0, 1])

    axis["fourier2"].set_xlim([0, max_freq])
    axis["fourier2"].set_ylim([0, 1])

    plt.draw()
    #plt.pause(1)
    plt.waitforbuttonpress()

    axis["signal"].plot([start1, end1], [5e3, 5e3], color="white", linewidth=1)
    axis["signal"].plot([start2, end2], [4.5e3, 4.5e3], color="white", linewidth=1)

    axis["sub1"].cla()
    axis["fourier1"].cla()

    axis["sub2"].cla()
    axis["fourier2"].cla()

plt.show(block=True)

print("done")