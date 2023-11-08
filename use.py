import numpy as np

from Network import Network
from Layer import FCLayer, ActivationLayer
from Loss import mse, mse_prime
from Activation import tanh, tanh_prime

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
x_evaluate = np.array([[[1,1]], [[1,1]], [[1,0]], [[1,1]]])
# [0][0][1][0]
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

nn = Network()
nn.add(FCLayer(2, 3))
nn.add(ActivationLayer(tanh, tanh_prime))
nn.add(FCLayer(3, 6))
nn.add(ActivationLayer(tanh, tanh_prime))
nn.add(FCLayer(6, 1))
nn.add(ActivationLayer(tanh, tanh_prime))

nn.use(mse, mse_prime)
nn.fit(x_train, y_train, epochs=10000, learning_rate=0.01)

output = nn.predict(x_evaluate)
print(output)
