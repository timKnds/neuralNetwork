import numpy as np


class Loss:
    @staticmethod
    def loss(y_true, y_pred):
        raise NotImplementedError

    @staticmethod
    def loss_prime(y_true, y_pred):
        raise NotImplementedError


class MSE(Loss):
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def loss_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropy(Loss):
    @staticmethod
    def loss(y_true, y_pred):
        return - np.sum(np.log(y_pred)*y_true)

    @staticmethod
    def loss_prime(y_true, y_pred):
        return - np.sum(y_true*1/y_pred)/y_true.size
