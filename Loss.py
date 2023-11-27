import numpy as np


class Loss:
    @staticmethod
    def loss(y, yhat):
        raise NotImplementedError

    @staticmethod
    def loss_prime(y, yhat):
        raise NotImplementedError


class MSE(Loss):
    @staticmethod
    def loss(y, yhat):
        # calculate the mean square error
        return (y - yhat) ** 2

    @staticmethod
    def loss_prime(y, yhat):
        return 2 * (yhat - y)


class CrossEntropy(Loss):
    @staticmethod
    def loss(y, yhat):
        # Calculate the cross-entropy loss
        return - np.sum(y * np.log(yhat)) / y.size


    @staticmethod
    def loss_prime(y, yhat):
        # Calculate the derivative of cross-entropy loss function
        return - (y / yhat) / y.size
