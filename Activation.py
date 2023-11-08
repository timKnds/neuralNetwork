import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2


def relu(x):
    return max(0, x)


def relu_prime(x):
    if x >= 0:
        return 1
    else:
        return 0
