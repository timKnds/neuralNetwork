import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def __call__(self, input_data):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Set weights with random values between -1 and 1
        self.weights = np.random.rand(input_size, output_size) * 2 - 1
        self.biases = np.random.rand(1, output_size) * 2 - 1

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def __call__(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation.activ
        self.activation_prime = activation.activ_grad

    def __call__(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
