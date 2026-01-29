from layers import Dense, ReLU, Softmax

class NeuralNet:
    def __init__(self):
        self.layers = [
            Dense(784, 128),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, 10),
            Softmax()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                params.append(layer.W)
                params.append(layer.b)
        return params

    def grads(self):
        grads = []
        for layer in self.layers:
            if hasattr(layer, "dW"):
                grads.append(layer.dW)
                grads.append(layer.db)
        return grads
