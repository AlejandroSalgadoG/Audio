import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from Wav.WavReader import WavReader

vowel = "u"
period = 368
start = 32000
end = 37000

t_data = WavReader(f"Data/vowels/long/{vowel}/{vowel}.wav").get_data().extract_portion(start, end)

data = t_data.data

n = 368

x1 = data[:-n]
x2 = data[n:]

x = x1.reshape(-1, 1)
y = x2

reg = LinearRegression().fit(x, y)
y_hat = reg.predict(x)

fig, axis = plt.subplot_mosaic([
    ["data"],
])

axis["data"].plot(y)
axis["data"].plot(y_hat)

plt.show()

print("done")