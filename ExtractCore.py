import numpy as np
import matplotlib.pyplot as plt

from Fourier import TimeData
from Wav.WavReader import WavReader
from Wav.WavWriter import WavWriter

# a -> 5272 - 5707 - 435 - x28
# e -> 6254 - 6686 - 432 - x28
# i -> 6738 - 7112 - 374 - x32
# o -> 5314 - 5739 - 425 - x28
# u -> 4328 - 4720 - 392 - x31

t_data = WavReader(f"Data/vowels/u/u.wav").get_data().extract_portion(4328, 4720)
data = TimeData( np.concatenate([t_data.data] * 31) )

fig, axis = plt.subplot_mosaic([["time"]])
axis["time"].plot(data.data, linewidth=1, color="blue")
axis["time"].set_xlim([0, data.n_samples])

plt.show()

writer = WavWriter(data)
writer.write_data("Data/vowels/u/perfect_u.wav")

print("done")