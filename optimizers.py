import numpy as np

class RMSProp:
    def __init__(self, params, lr=0.001, decay=0.9, eps=1e-8):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {id(p): np.zeros_like(p) for p in params}

    def step(self, params, grads):
        for p, g in zip(params, grads):
            c = self.cache[id(p)]
            c[:] = self.decay * c + (1 - self.decay) * g * g
            p -= self.lr * g / (np.sqrt(c) + self.eps)
