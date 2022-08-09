import matplotlib.pyplot as plt

from Fourier import Time2Freq
from Wav.WavReader import WavReader
from Wav.WavReproduce import WavReproducer

window_size = 450
n_times = 28
max_freq = 2000

def norm(data):
    min_s, max_s =  data.min(), data.max()
    return (data - min_s) / (max_s - min_s)

def error(x1, x2):
    return sum( (x1 - x2)**2 ) / x1.size

a_data = WavReader(f"Data/vowels/perfect/a/a.wav").get_data()
e_data = WavReader(f"Data/vowels/perfect/e/e.wav").get_data()
i_data = WavReader(f"Data/vowels/perfect/i/i.wav").get_data()
o_data = WavReader(f"Data/vowels/perfect/o/o.wav").get_data()
u_data = WavReader(f"Data/vowels/perfect/u/u.wav").get_data()

a_freq = Time2Freq.transform(a_data)
e_freq = Time2Freq.transform(e_data)
i_freq = Time2Freq.transform(i_data)
o_freq = Time2Freq.transform(o_data)
u_freq = Time2Freq.transform(u_data)

a_amps = a_freq.get_amplitudes()[:max_freq]
e_amps = e_freq.get_amplitudes()[:max_freq]
i_amps = i_freq.get_amplitudes()[:max_freq]
o_amps = o_freq.get_amplitudes()[:max_freq]
u_amps = u_freq.get_amplitudes()[:max_freq]

norm_a_amps = norm(a_amps)
norm_e_amps = norm(e_amps)
norm_i_amps = norm(i_amps)
norm_o_amps = norm(o_amps)
norm_u_amps = norm(u_amps)

wav_reproducer = WavReproducer()
reader = WavReader(f"Data/vowels/a/a.wav")
t_data = reader.get_data()

plt.ion()

fig, axis = plt.subplot_mosaic([
    ["signal",     "signal",    "signal"   ],
    ["sub_signal", "fourier_a", "fourier_e"],
    ["fourier_i",  "fourier_o", "fourier_u"],
])

axis["signal"].plot(t_data.data, linewidth=1, color="blue")
axis["signal"].set_xlim([0, reader.n_data_samples])
axis["signal"].set_ylim([-6e3, 6e3])

batch1 = t_data.batch(size=window_size)

for window1 in batch1:
    window1_repeat = window1.repeat(n_times)

    wav_reproducer.reproduce( window1_repeat )

    freq_data = Time2Freq.transform(window1_repeat)
    amps_s = freq_data.get_amplitudes(max_freq=max_freq)
    norm_s_amps = norm(amps_s)

    error_a = error(norm_s_amps, norm_a_amps)
    error_e = error(norm_s_amps, norm_e_amps)
    error_i = error(norm_s_amps, norm_i_amps)
    error_o = error(norm_s_amps, norm_o_amps)
    error_u = error(norm_s_amps, norm_u_amps)

    error_min = min(error_a, error_e, error_i, error_o, error_u)

    axis["signal"].plot([window1.start, window1.end], [5e3, 5e3], color="red", linewidth=1)

    axis["sub_signal"].plot(window1.data, linewidth=1, color="red")

    axis["fourier_a"].plot(norm_s_amps, linewidth=1, color="blue")
    axis["fourier_a"].plot(norm_a_amps, linewidth=1, color="black", alpha=0.85)
    axis["fourier_a"].text(1200, 0.8, "%.4f" % error_a, color="black" if error_a != error_min else "red")

    axis["fourier_e"].plot(norm_s_amps, linewidth=1, color="blue")
    axis["fourier_e"].plot(norm_e_amps, linewidth=1, color="black", alpha=0.85)
    axis["fourier_e"].text(1200, 0.8, "%.4f" % error_e, color="black" if error_e != error_min else "red")

    axis["fourier_i"].plot(norm_s_amps, linewidth=1, color="blue")
    axis["fourier_i"].plot(norm_i_amps, linewidth=1, color="black", alpha=0.85)
    axis["fourier_i"].text(1200, 0.8, "%.4f" % error_i, color="black" if error_i != error_min else "red")

    axis["fourier_o"].plot(norm_s_amps, linewidth=1, color="blue")
    axis["fourier_o"].plot(norm_o_amps, linewidth=1, color="black", alpha=0.85)
    axis["fourier_o"].text(1200, 0.8, "%.4f" % error_o, color="black" if error_o != error_min else "red")

    axis["fourier_u"].plot(norm_s_amps, linewidth=1, color="blue")
    axis["fourier_u"].plot(norm_u_amps, linewidth=1, color="black", alpha=0.85)
    axis["fourier_u"].text(1200, 0.8, "%.4f" % error_u, color="black" if error_u != error_min else "red")

    axis["sub_signal"].set_xlim([0, window_size])
    axis["sub_signal"].set_ylim([-5e3, 5e3])

    axis["fourier_a"].set_xlim([0, max_freq])
    axis["fourier_e"].set_xlim([0, max_freq])
    axis["fourier_i"].set_xlim([0, max_freq])
    axis["fourier_o"].set_xlim([0, max_freq])
    axis["fourier_u"].set_xlim([0, max_freq])

    axis["fourier_a"].set_ylim([0, 1])
    axis["fourier_e"].set_ylim([0, 1])
    axis["fourier_i"].set_ylim([0, 1])
    axis["fourier_o"].set_ylim([0, 1])
    axis["fourier_u"].set_ylim([0, 1])

    axis["sub_signal"].set_xticklabels([])
    axis["sub_signal"].set_yticklabels([])

    axis["fourier_a"].set_yticklabels([])
    axis["fourier_e"].set_yticklabels([])
    axis["fourier_i"].set_yticklabels([])
    axis["fourier_o"].set_yticklabels([])
    axis["fourier_u"].set_yticklabels([])

    plt.draw()
    #plt.pause(1)
    plt.waitforbuttonpress()

    axis["signal"].plot([window1.start, window1.end], [5e3, 5e3], color="white", linewidth=1)

    axis["sub_signal"].cla()

    axis["fourier_a"].cla()
    axis["fourier_e"].cla()
    axis["fourier_i"].cla()
    axis["fourier_o"].cla()
    axis["fourier_u"].cla()

plt.show(block=True)

print("done")