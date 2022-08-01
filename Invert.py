import matplotlib.pyplot as plt

from Fourier import Freq2Time, Time2Freq
from Wav.WavReader import WavReader

reader = WavReader(f"Data/vowels/a/a.wav")

t_data_original = reader.get_data()
t_data_reconstruct = Freq2Time.transform(Time2Freq.transform(t_data_original))

fig, axis = plt.subplots()

axis.plot(t_data_original.data, linewidth=1, color="blue")
axis.plot(t_data_reconstruct.data, linewidth=0.5, color="red")

axis.set_xlim([0, reader.n_data_samples])

plt.show()

print("done")