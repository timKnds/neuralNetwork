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

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss):
        self.loss = loss.loss
        self.loss_grad = loss.loss_grad

    def fit(self, x_train, y_train, epochs, learning_rate):
        assert len(x_train) == len(y_train)

        for i in range(epochs):
            error = 0
            for xi, yi in zip(x_train, y_train):
            # forward
                output = xi
                for layer in self.layers:
                    output = layer(output)
                error += self.loss(yi, output)
                err_grad = self.loss_grad(yi, output)
                for layer in reversed(self.layers):
                    err_grad = layer.backward(err_grad, learning_rate)
            error /= len(x_train)
            print('Tim epoch %d/%d  error=%f' % (i+1, epochs, error))
