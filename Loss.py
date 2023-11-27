import numpy as np


class Loss:
    @staticmethod
    def loss(y, yhat):
        raise NotImplementedError

    @staticmethod
    def loss_grad(y, yhat):
        raise NotImplementedError


class MSE(Loss):
    @staticmethod
    def loss(y, yhat):
        # calculate the mean square error over numpy arrays y and yhat
        return np.sum((y - yhat) ** 2)    

    @staticmethod
    def loss_grad(y, yhat):
        return [2 * (yhati - yi) for yi, yhati in zip(y, yhat)]


class CrossEntropy(Loss):
    @staticmethod
    def loss(y, yhat):
        # Calculate the cross-entropy loss
        if y == 1:
            return - np.log(yhat)
        elif y == 0:
            return - np.log(1 - yhat)
        else:
            raise ValueError("y must be 0 or 1")

    @staticmethod
    def loss_grad(y, yhat):
        # Calculate the derivative of cross-entropy loss function
        if y == 1:
            return - 1 / yhat
        elif y == 0:
            return 1 / (1 - yhat)
        else:
            raise ValueError("y must be 0 or 1")