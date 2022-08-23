import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from Wav.WavReader import WavReader

start, end = 32000, 37000
a_data = WavReader(f"Data/vowels/long/a/a.wav").get_data().extract_portion(start, end)
e_data = WavReader(f"Data/vowels/long/e/e.wav").get_data().extract_portion(start, end)
i_data = WavReader(f"Data/vowels/long/i/i.wav").get_data().extract_portion(start, end)
o_data = WavReader(f"Data/vowels/long/o/o.wav").get_data().extract_portion(start, end)
u_data = WavReader(f"Data/vowels/long/u/u.wav").get_data().extract_portion(start, end)

def lag_data(data, lags):
    n = data.size - max(lags)
    return np.concatenate( [data[lag:n+lag].reshape(-1, 1) for lag in lags], axis=1 ) if n > 0 else np.array([[]])

def feature_target_split(X):
    return X[:, :-1], X[:,  -1]

def mse(y, y_hat):
    return np.sum( np.square(y - y_hat) )

lags = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
data = a_data.data
x, y = feature_target_split( lag_data(data, lags) )

reg = LinearRegression().fit(x, y)
m = reg.coef_
b = reg.intercept_

fig, axis = plt.subplot_mosaic([
    ["a", "e"],
    ["i", "o"],
    ["u", "r"],
])

x_a, y_a = feature_target_split( lag_data(a_data.data, lags) )
x_e, y_e = feature_target_split( lag_data(e_data.data, lags) )
x_i, y_i = feature_target_split( lag_data(i_data.data, lags) )
x_o, y_o = feature_target_split( lag_data(o_data.data, lags) )
x_u, y_u = feature_target_split( lag_data(u_data.data, lags) )
x_r, y_r = feature_target_split( lag_data(np.random.uniform(low=-4000, high=2000, size=(5000, 1)), lags) )

y_hat_a = reg.predict(x_a)
y_hat_e = reg.predict(x_e)
y_hat_i = reg.predict(x_i)
y_hat_o = reg.predict(x_o)
y_hat_u = reg.predict(x_u)
y_hat_r = reg.predict(x_r)

mse_a = mse(y_a, y_hat_a)
mse_e = mse(y_e, y_hat_e)
mse_i = mse(y_i, y_hat_i)
mse_o = mse(y_o, y_hat_o)
mse_u = mse(y_u, y_hat_u)
mse_r = mse(y_r, y_hat_r)

mse_total = mse_a + mse_e + mse_i + mse_o + mse_u

mse_prop_a = 100 - mse_a * 100 / mse_total
mse_prop_e = 100 - mse_e * 100 / mse_total
mse_prop_i = 100 - mse_i * 100 / mse_total
mse_prop_o = 100 - mse_o * 100 / mse_total
mse_prop_u = 100 - mse_u * 100 / mse_total
mse_prop_r = 100 - mse_r * 100 / mse_total

mse_prop_max = max(mse_prop_a, mse_prop_e, mse_prop_i, mse_prop_o, mse_prop_u)

axis["a"].plot(y_a)
axis["a"].plot(y_hat_a)
axis["a"].set_title( "a = %.2f" % mse_prop_a, color="red" if mse_prop_a == mse_prop_max else "black" )
axis["a"].set_xticks([])

axis["e"].plot(y_e)
axis["e"].plot(y_hat_e)
axis["e"].set_title( "e = %.2f" % mse_prop_e, color="red" if mse_prop_e == mse_prop_max else "black" )
axis["e"].set_xticks([])

axis["i"].plot(y_i)
axis["i"].plot(y_hat_i)
axis["i"].set_title( "i = %.2f" % mse_prop_i, color="red" if mse_prop_i == mse_prop_max else "black" )
axis["i"].set_xticks([])

axis["o"].plot(y_o)
axis["o"].plot(y_hat_o)
axis["o"].set_title( "o = %.2f" % mse_prop_o, color="red" if mse_prop_o == mse_prop_max else "black" )
axis["o"].set_xticks([])

axis["u"].plot(y_u)
axis["u"].plot(y_hat_u)
axis["u"].set_title( "u = %.2f" % mse_prop_u, color="red" if mse_prop_u == mse_prop_max else "black" )
axis["u"].set_xticks([])

axis["r"].plot(y_r)
axis["r"].plot(y_hat_r)
axis["r"].set_title( "r = %.2f" % mse_prop_r )
axis["r"].set_xticks([])

plt.show()

print("done")