import numpy as np
from sklearn.linear_model import LinearRegression

from Wav.WavReader import WavReader

class LinModel:
    def __init__(self, lags):
        self.lags = lags
        self.model = LinearRegression()

    def mse(self, y, y_hat):
        return np.sum( np.square(y - y_hat) )

    def apply_lag(self, signal):
        n = signal.size - max(self.lags)
        if n <= 0: return np.array([[]])
        return np.concatenate( [signal[lag:n+lag].reshape(-1, 1) for lag in self.lags], axis=1 )

    def extract_features_and_target(self, signal):
        lag_matrix = self.apply_lag(signal)
        x, y = lag_matrix[:, :-1], lag_matrix[:, -1]
        return x, y

    def fit(self, signal):
        x, y = self.extract_features_and_target(signal)
        self.model.fit(x, y)
        return self

    def predict(self, signal):
        x, _ = self.extract_features_and_target(signal)
        return self.model.predict(x)

    def test(self, signal):
        x, y = self.extract_features_and_target(signal)
        y_hat = self.model.predict(x)
        return self.mse(y, y_hat)


class TemporalClassificator:
    def __init__(self, model_a, model_e, model_i, model_o, model_u):
        self.models = {
            "a": model_a,
            "e": model_e,
            "i": model_i,
            "o": model_o,
            "u": model_u,
        }

    def get_labels(self):
        return list(self.models.keys())

    def get_percent_certainty(self, signal):
        errors = np.array([ model.test(signal) for model in self.models.values() ])
        return errors * 100 / np.sum(errors), self.get_labels(),

    def classify(self, signal):
        percent_errors, labels = self.get_percent_certainty(signal)
        return percent_errors, labels[ np.argmin(percent_errors) ]


start, end = 32000, 37000
a_data = WavReader(f"Data/vowels/long/a/a.wav").get_data()
e_data = WavReader(f"Data/vowels/long/e/e.wav").get_data()
i_data = WavReader(f"Data/vowels/long/i/i.wav").get_data()
o_data = WavReader(f"Data/vowels/long/o/o.wav").get_data()
u_data = WavReader(f"Data/vowels/long/u/u.wav").get_data()

lags = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450]

model_a = LinModel(lags).fit(a_data.extract_portion(start, end).data)
model_e = LinModel(lags).fit(e_data.extract_portion(start, end).data)
model_i = LinModel(lags).fit(i_data.extract_portion(start, end).data)
model_o = LinModel(lags).fit(o_data.extract_portion(start, end).data)
model_u = LinModel(lags).fit(u_data.extract_portion(start, end).data)

classifier = TemporalClassificator(model_a, model_e, model_i, model_o, model_u)

for batch in u_data.batch(5000, 13000):
    percent_errors, result = classifier.classify( batch.data )
    print(percent_errors, result)