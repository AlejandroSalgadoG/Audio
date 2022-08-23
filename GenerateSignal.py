import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from Fourier import TimeData
from Wav.WavReader import WavReader
from Wav.WavWriter import WavWriter

a_data = WavReader(f"Data/vowels/long/a/a.wav").get_data().extract_portion(32000, 37000)
e_data = WavReader(f"Data/vowels/long/e/e.wav").get_data().extract_portion(32000, 37000)
i_data = WavReader(f"Data/vowels/long/i/i.wav").get_data().extract_portion(32000, 37000)
o_data = WavReader(f"Data/vowels/long/o/o.wav").get_data().extract_portion(32000, 37000)
u_data = WavReader(f"Data/vowels/long/u/u.wav").get_data().extract_portion(32000, 37000)

def lag_data(data, lags):
    n = data.size - max(lags)
    return np.concatenate( [data[lag:n+lag].reshape(-1, 1) for lag in lags], axis=1 ) if n > 0 else np.array([[]])

def feature_target_split(X):
    return X[:, :-1], X[:,  -1]

data = u_data.data
lags = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

x, y = feature_target_split( lag_data(data, lags) )
reg = LinearRegression().fit(x, y)

def predict(sample):
    return reg.predict([sample])[0]

def extract_sample(data):
    return [data[lag] for lag in lags]

y_hat = data[:500].tolist()
for i in range(20000):
    sample = [y_hat[lag + i] for lag in lags[:-1]]
    y_hat.append( predict(sample) )

y_hat = y_hat[500:] # remove initial data

fig, axis = plt.subplot_mosaic([
    ["result"],
])

axis["result"].plot(data[:500].tolist() + y.tolist())
axis["result"].plot(data[:500].tolist() + y_hat)

plt.show()

t_data = TimeData( np.array(y_hat).astype(int) )
writer = WavWriter(t_data)
writer.write_data("Data/vowels/generated/u.wav")

print("done")