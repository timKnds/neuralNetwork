import numpy as np


class Activation:
    @staticmethod
    def activ(x):
        raise NotImplementedError

    @staticmethod
    def activ_prime(x):
        raise NotImplementedError


class Tanh(Activation):
    @staticmethod
    def activ(x):
        return np.tanh(x)

    @staticmethod
    def activ_prime(x):
        return 1-np.tanh(x)**2


class ReLu(Activation):
    @staticmethod
    def activ(x):
        return max(0, x)

    @staticmethod
    def activ_prime(x):
        if x >= 0:
            return 1
        else:
            return 0
