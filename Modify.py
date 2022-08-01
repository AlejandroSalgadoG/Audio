import matplotlib.pyplot as plt

from Fourier import Freq2Time, Time2Freq
from Wav.WavReader import WavReader
from Wav.WavWriter import WavWriter


reader = WavReader(f"Data/vowels/a/a.wav")
t_data1 = reader.get_data()

f_data1 = Time2Freq.transform(t_data1)
amps1 = f_data1.get_amplitudes()

f_data1.pos_coeff[1500:] = 0

t_data2 = Freq2Time.transform(f_data1)
f_data2 = Time2Freq.transform(t_data2)
amps2 = f_data2.get_amplitudes()

fig, axis = plt.subplot_mosaic([
    ["signal", "signal"],
    ["fourier1", "fourier2"],
])

axis["signal"].plot(t_data1.data, linewidth=1, color="blue")
axis["signal"].plot(t_data2.data, linewidth=1, color="red")

axis["fourier1"].plot(amps1, linewidth=1, color="blue")
axis["fourier2"].plot(amps2, linewidth=1, color="red")

axis["signal"].set_xlim([0, reader.n_data_samples])
axis["fourier1"].set_xlim([0, f_data1.nyquist])
axis["fourier2"].set_xlim([0, f_data2.nyquist])

plt.show()

WavWriter(t_data2).write_data("Output/a.wav")

print("done")