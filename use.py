import numpy as np
import pandas as pd

from Network import Network
from Layer import FCLayer, ActivationLayer
from Loss import MSE, CrossEntropy
from Activation import Tanh, ReLu


x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
x_evaluate = np.array([[[0,0]], [[1,1]], [[1,0]], [[1,1]]])
# [0][0][1][0]
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

nn = Network()
nn.add(FCLayer(2, 3))
nn.add(ActivationLayer(Tanh()))
nn.add(FCLayer(3, 1))
nn.add(ActivationLayer(Tanh()))

nn.use(MSE())
nn.fit(x_train, y_train, epochs=10000, learning_rate=0.01)

output = nn.predict(x_evaluate)
print(output)

"""
train = pd.read_csv('/Users/timknudsen/Documents/Dokumente Mac/Pycharm-Projekte/neuralNetwork/archive/mnist_train.csv')
test = pd.read_csv('/Users/timknudsen/Documents/Dokumente Mac/Pycharm-Projekte/neuralNetwork/archive/mnist_test.csv')

def read(data):
    x = np.zeros((len(data), 1, len(data.columns)-1))
    y = np.zeros((len(data), 1, 10))
    for i in range(len(data)):
        x[i] = data.iloc[i,1:].to_numpy()
        y[i, 0, data.iloc[i,0]-1] = 1
    return x, y


x_train, y_train = read(train)
x_test, y_test = read(test)

net = Network()
net.add(FCLayer(784, 392))
net.add(ActivationLayer(Tanh()))
net.add(FCLayer(392, 10))
net.add(ActivationLayer(Tanh()))

net.use(CrossEntropy())
net.fit(x_train, y_train, 10, learning_rate=0.1)

output = net.predict(x_test)
print(output)
"""
