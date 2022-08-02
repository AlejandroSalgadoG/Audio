import matplotlib.pyplot as plt

from Wav.WavReader import WavReader

a_data = WavReader(f"Data/vowels/a/a.wav").get_data()
e_data = WavReader(f"Data/vowels/e/e.wav").get_data()
i_data = WavReader(f"Data/vowels/i/i.wav").get_data()
o_data = WavReader(f"Data/vowels/o/o.wav").get_data()
u_data = WavReader(f"Data/vowels/u/u.wav").get_data()

fig, axis = plt.subplot_mosaic([
    ["time1"],
    ["time2"],
    ["time3"],
    ["time4"],
    ["time5"],
])

axis["time1"].plot(a_data.data, linewidth=1, color="blue")
axis["time2"].plot(e_data.data, linewidth=1, color="blue")
axis["time3"].plot(i_data.data, linewidth=1, color="blue")
axis["time4"].plot(o_data.data, linewidth=1, color="blue")
axis["time5"].plot(u_data.data, linewidth=1, color="blue")

plt.show()

print("done")