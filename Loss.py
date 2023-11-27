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
        assert len(y) == len(yhat)
        # Calculate the cross-entropy loss
        return np.sum(-(y * np.log(yhat))+(1-y)*np.log(1-yhat))

    @staticmethod
    def loss_grad(y, yhat):
        assert len(y) == len(yhat)
        # Calculate the derivative of cross-entropy loss function
        return -(y * 1/yhat) + (1-y) * 1/(1-yhat)