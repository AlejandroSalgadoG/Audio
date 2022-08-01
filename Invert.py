import matplotlib.pyplot as plt

from Fourier import FrequencyDomain, TimeDomain
from Wav.WavReader import WavReader

reader = WavReader(f"Data/vowels/a/a.wav")
data = reader.get_data()

f_data = FrequencyDomain(data)
t_data = TimeDomain(f_data.coeff)

fig, axis = plt.subplots()

axis.plot(data, linewidth=1, color="blue")
axis.plot(t_data.signal, linewidth=0.5, color="red")

axis.set_xlim([0, reader.n_data_samples])

plt.show()

print("done")