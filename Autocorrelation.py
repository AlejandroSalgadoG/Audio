import numpy as np
import matplotlib.pyplot as plt

from Wav.WavReader import WavReader

vowel = "u"
min_search = 0
max_search = 550
step = 5

start = 32000
end = start + 2000

all_data = WavReader(f"Data/vowels/long/{vowel}/{vowel}.wav").get_data()
t_data = all_data.extract_portion(start, end)

plt.ion()

fig, axis = plt.subplot_mosaic([
    ["signal", "auto"],
    ["corr",   "auto"],
])

axis["signal"].plot(t_data.data[min_search:max_search])

axis["signal"].set_xlim([min_search, max_search])
axis["corr"].set_ylim([-1.1, 1.1])
axis["corr"].set_xlim([min_search, max_search])

for lag in range(min_search, max_search, step):
    t_data_lag = all_data.extract_portion(start + lag, end + lag)

    corr = np.corrcoef(t_data.data, t_data_lag.data)[0,1]

    axis["auto"].set_title(f"Lag = {lag}, corr = {corr:.2f}")

    axis["auto"].scatter(t_data.data, t_data_lag.data)
    axis["signal"].plot(t_data.data[:lag + 1], color="red")
    axis["corr"].scatter(lag, corr, s=5, color="blue")

    plt.draw()
    plt.pause(0.01)

    axis["auto"].cla()

plt.show(block=True)

print("done")