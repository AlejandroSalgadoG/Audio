import matplotlib.pyplot as plt
import numpy as np

from Fourier import FrequencyDomain, TimeDomain
from Wav.WavReader import WavReader
from Wav.WavWriter import WavWriter


reader = WavReader(f"Data/vowels/a/a.wav")
data = reader.get_data()

f_data1 = FrequencyDomain(data)
amps1 = f_data1.get_amplitudes()[:f_data1.nyquist]

new_coeff = np.copy(f_data1.coeff)
new_coeff[f_data1.nyquist - 4500 : f_data1.nyquist + 4500] = 0
t_data = TimeDomain(new_coeff)

f_data2 = FrequencyDomain(t_data.signal)
amps2 = f_data2.get_amplitudes()[:f_data2.nyquist]

fig, axis = plt.subplot_mosaic([
    ["signal", "signal"],
    ["fourier1", "fourier2"],
])

axis["signal"].plot(data, linewidth=1, color="blue")
axis["signal"].plot(t_data.signal, linewidth=0.5, color="red")

axis["fourier1"].plot(amps1, linewidth=1, color="blue")
axis["fourier2"].plot(amps2, linewidth=1, color="red")

axis["signal"].set_xlim([0, reader.n_data_samples])
axis["signal"].set_xlim([0, reader.n_data_samples])
axis["fourier1"].set_xlim([0, f_data1.nyquist])
axis["fourier2"].set_xlim([0, f_data2.nyquist])

plt.show()

WavWriter(t_data.signal).write_data("Output/a.wav")

print("done")