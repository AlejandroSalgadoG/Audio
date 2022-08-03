import matplotlib.pyplot as plt

from Fourier import Time2Freq
from Wav.WavReader import WavReader

window_size = 450
n_times = 28

def norm(data):
    min_s, max_s =  data.min(), data.max()
    return (data - min_s) / (max_s - min_s)

reader = WavReader(f"Data/vowels/all/all.wav")
t_data = reader.get_data()

plt.ion()

fig, axis = plt.subplot_mosaic([
    ["signal",   "signal",   "signal"  ],
    ["sub1",     "sub2",     "sub3"    ],
    ["fourier1", "fourier2", "fourier3"],
])

axis["signal"].plot(t_data.data, linewidth=1, color="blue")
axis["signal"].set_xlim([0, reader.n_data_samples])
axis["signal"].set_ylim([-6e3, 6e3])

batch1 = t_data.batch(size=window_size)
batch2 = t_data.batch(size=window_size, offset=window_size // 3)
batch3 = t_data.batch(size=window_size, offset=window_size // 3 * 2)

for window1, window2, window3 in zip(batch1, batch2, batch3):
    freq_data1 = Time2Freq.transform(window1.repeat(n_times))
    amps1 = freq_data1.get_amplitudes()
    norm_amps1 = norm(amps1)

    freq_data2 = Time2Freq.transform(window2.repeat(n_times))
    amps2 = freq_data2.get_amplitudes()
    norm_amps2 = norm(amps2)

    freq_data3 = Time2Freq.transform(window3.repeat(n_times))
    amps3 = freq_data3.get_amplitudes()
    norm_amps3 = norm(amps3)

    axis["signal"].plot([window1.start, window1.end], [5e3, 5e3], color="red", linewidth=1)
    axis["signal"].plot([window2.start, window2.end], [4.5e3, 4.5e3], color="green", linewidth=1)
    axis["signal"].plot([window3.start, window3.end], [4e3, 4e3], color="purple", linewidth=1)

    axis["sub1"].plot(window1.data, linewidth=1, color="red")
    axis["fourier1"].plot(norm_amps1, linewidth=1, color="red")

    axis["sub2"].plot(window2.data, linewidth=1, color="green")
    axis["fourier2"].plot(norm_amps2, linewidth=1, color="green")

    axis["sub3"].plot(window3.data, linewidth=1, color="purple")
    axis["fourier3"].plot(norm_amps3, linewidth=1, color="purple")

    axis["sub1"].set_xlim([0, window_size])
    axis["sub1"].set_ylim([-5e3, 5e3])

    axis["sub2"].set_xlim([0, window_size])
    axis["sub2"].set_ylim([-5e3, 5e3])

    axis["sub3"].set_xlim([0, window_size])
    axis["sub3"].set_ylim([-5e3, 5e3])

    axis["fourier1"].set_xlim([0, freq_data1.nyquist])
    axis["fourier1"].set_ylim([0, 1])

    axis["fourier2"].set_xlim([0, freq_data2.nyquist])
    axis["fourier2"].set_ylim([0, 1])

    axis["fourier3"].set_xlim([0, freq_data3.nyquist])
    axis["fourier3"].set_ylim([0, 1])

    plt.draw()
    #plt.pause(1)
    plt.waitforbuttonpress()

    axis["signal"].plot([window1.start, window1.end], [5e3, 5e3], color="white", linewidth=1)
    axis["signal"].plot([window2.start, window2.end], [4.5e3, 4.5e3], color="white", linewidth=1)
    axis["signal"].plot([window3.start, window3.end], [4e3, 4e3], color="white", linewidth=1)

    axis["sub1"].cla()
    axis["fourier1"].cla()

    axis["sub2"].cla()
    axis["fourier2"].cla()

    axis["sub3"].cla()
    axis["fourier3"].cla()

plt.show(block=True)

print("done")