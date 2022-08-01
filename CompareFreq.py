import matplotlib.pyplot as plt

from Fourier import Time2Freq
from Wav.WavReader import WavReader

max_freq = 2000

a_data = WavReader(f"Data/vowels/a/a.wav").get_data()
e_data = WavReader(f"Data/vowels/e/e.wav").get_data()
i_data = WavReader(f"Data/vowels/i/i.wav").get_data()
o_data = WavReader(f"Data/vowels/o/o.wav").get_data()
u_data = WavReader(f"Data/vowels/u/u.wav").get_data()

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

fig, axis = plt.subplot_mosaic([
    ["fourier1"],
    ["fourier2"],
    ["fourier3"],
    ["fourier4"],
    ["fourier5"],
])

axis["fourier1"].plot(a_amps, linewidth=1, color="blue")
axis["fourier2"].plot(e_amps, linewidth=1, color="blue")
axis["fourier3"].plot(i_amps, linewidth=1, color="blue")
axis["fourier4"].plot(o_amps, linewidth=1, color="blue")
axis["fourier5"].plot(u_amps, linewidth=1, color="blue")

axis["fourier1"].set_xlim([0, max_freq])
axis["fourier2"].set_xlim([0, max_freq])
axis["fourier3"].set_xlim([0, max_freq])
axis["fourier4"].set_xlim([0, max_freq])
axis["fourier5"].set_xlim([0, max_freq])

plt.show()

print("done")