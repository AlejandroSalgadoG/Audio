import numpy as np
import matplotlib.pyplot as plt

from Wav.WavReader import WavReader

vowel = "u"
period = 368
start = 32000
end = 37000

t_data = WavReader(f"Data/vowels/long/{vowel}/{vowel}.wav").get_data().extract_portion(start, end)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

canonical_theta = np.linspace(0, 2*np.pi, period)
theta = np.resize(canonical_theta, t_data.data.size)

ax.plot(theta, t_data.data)

#ax.set_rmax(2)
ax.set_rticks([])
ax.set_xticks([])
#ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

plt.show()

print("done")