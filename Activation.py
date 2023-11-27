import numpy as np


class Activation:
    @staticmethod
    def activ(x):
        raise NotImplementedError

    @staticmethod
    def activ_grad(x):
        raise NotImplementedError


class Tanh(Activation):
    @staticmethod
    def activ(x):
        return np.tanh(x)

    @staticmethod
    def activ_grad(x):
        return 1-np.tanh(x)**2


class ReLu(Activation):
    @staticmethod
    def activ(x):
        return np.maximum(np.zeros_like(x), x)

    @staticmethod
    def activ_grad(x):
        copy = np.copy(x)
        copy[copy >= 0] = 1
        copy[copy < 0] = 0
        return copy


class Sigmoid(Activation):
    @staticmethod
    def activ(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def activ_grad(x):
        return np.exp(-x)/(1+np.exp(-x))**2