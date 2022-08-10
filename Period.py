import numpy as np
import matplotlib.pyplot as plt

from Wav.WavReader import WavReader

vowel = "u"
min_search = 0
max_search = 550
step = 1

start = 32000
end = start + 2000

all_data = WavReader(f"Data/vowels/long/{vowel}/{vowel}.wav").get_data()
t_data = all_data.extract_portion(start, end)

fig, axis = plt.subplot_mosaic([
    ["corr"],
    ["diff"],
])

axis["corr"].set_ylim([-1.1, 1.1])
axis["corr"].set_xlim([min_search, max_search])
axis["diff"].set_xlim([min_search, max_search])

lags = np.arange(min_search, max_search, step)

corr = np.array([np.corrcoef(t_data.data, all_data.extract_portion(start+lag, end+lag).data)[0,1] for lag in lags])

diff = np.diff(corr)
peaks_pos, *_ = np.where( np.diff( np.sign(diff) ) )

period_pos = peaks_pos[ np.argmax( np.abs(corr[peaks_pos]) ) ]

for peak_pos in peaks_pos + 1:
    axis["corr"].axvline(peak_pos)
    axis["diff"].axvline(peak_pos)

axis["corr"].axvline(period_pos, color="red")

axis["corr"].scatter(lags, corr, s=5, color="blue")
axis["diff"].scatter(lags[1:], diff, s=5, color="blue")
axis["diff"].axhline(0)

plt.show()

print("done")