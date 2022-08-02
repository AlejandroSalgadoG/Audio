import matplotlib.pyplot as plt

from Wav.WavReader import WavReader

a_data = WavReader(f"Data/vowels/perfect/a.wav").get_data()
e_data = WavReader(f"Data/vowels/perfect/e.wav").get_data()
i_data = WavReader(f"Data/vowels/perfect/i.wav").get_data()
o_data = WavReader(f"Data/vowels/perfect/o.wav").get_data()
u_data = WavReader(f"Data/vowels/perfect/u.wav").get_data()

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

axis["time1"].set_xlim([0, 12000])
axis["time2"].set_xlim([0, 12000])
axis["time3"].set_xlim([0, 12000])
axis["time4"].set_xlim([0, 12000])
axis["time5"].set_xlim([0, 12000])

plt.show()

print("done")