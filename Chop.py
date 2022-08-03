import matplotlib.pyplot as plt

from Wav.WavReader import WavReader
from Wav.WavWriter import WavWriter

data = WavReader(f"Data/vowels/all/all.wav").get_data()

a = data.extract_portion(start=34000, end=51000)
e = data.extract_portion(start=120000, end=133500)
i = data.extract_portion(start=204000, end=222000)
o = data.extract_portion(start=291800, end=304500)
u = data.extract_portion(start=367000, end=375600)

new_data = a.concat(e).concat(i).concat(o).concat(u)

fig, axis = plt.subplot_mosaic([
    ["time1"],
    ["time2"],
    ["time3"],
    ["time4"],
    ["time5"],
    ["time6"],
    ["time7"],
])

axis["time1"].plot(data.data, linewidth=1, color="blue")
axis["time2"].plot(a.data, linewidth=1, color="blue")
axis["time3"].plot(e.data, linewidth=1, color="blue")
axis["time4"].plot(i.data, linewidth=1, color="blue")
axis["time5"].plot(o.data, linewidth=1, color="blue")
axis["time6"].plot(u.data, linewidth=1, color="blue")
axis["time7"].plot(new_data.data, linewidth=1, color="blue")

plt.show()

writer = WavWriter(new_data)
writer.write_data("Data/vowels/all/all_chop.wav")

print("done")