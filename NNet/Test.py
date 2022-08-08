import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(2,),
    max_iter=2000,
    tol=1e-5,
    alpha=0.3,
    learning_rate="adaptive",
    random_state=1,
    solver="lbfgs",
    activation="tanh",
    early_stopping=True,
)

x_train = np.random.rand(10000, 2)
y_train = np.zeros( shape=(10000, 1) )

f1 = x_train[:, 0]
f2 = x_train[:, 1]

c1, c2 = (0.75, 0.5)
dist = np.sqrt((f1 - c1)**2 + (f2 - c2)**2)

y_train[dist >= 0.5] = 0
y_train[dist < 0.5] = 1

#y_train = (f2 >= f1).astype(int)

mlp.fit(x_train, y_train)

_x = np.linspace(0, 1, 100)
_y = np.linspace(0, 1, 100)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

x_test = np.stack([x, y]).T
y_hat = mlp.predict(x_test)

fig, axis = plt.subplot_mosaic([
    ["data",  "space"],
])

axis["data"].scatter(f1, f2, c=y_train, s=2)
axis["data"].set_xlim([0, 1])
axis["data"].set_ylim([0, 1])

axis["space"].scatter(x, y, c=y_hat, s=2)
axis["space"].set_xlim([0, 1])
axis["space"].set_ylim([0, 1])

plt.show()

print("done")