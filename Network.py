from Layer import Layer


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_grad = None

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __add__(self, other):
        assert isinstance(other, Layer)
        self.add(other)

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss):
        self.loss = loss.loss
        self.loss_grad = loss.loss_grad

    def fit(self, x_train, y_train, epochs, learning_rate):
        assert len(x_train) == len(y_train)

        samples = len(x_train)
        err = 0
        for j in range(epochs):
            for i in range(samples):
                y = y_train[i]
                yhat = self(x_train[i])
                err += self.loss(y, yhat)
                err_grad = self.loss_grad(y_train[i], yhat)
                for layer in reversed(self.layers):
                    err_grad = layer.backward(err_grad, learning_rate)

            err /= len(x_train)
            print('epoch %d/%d  error=%f' % (j+1, epochs, err))
