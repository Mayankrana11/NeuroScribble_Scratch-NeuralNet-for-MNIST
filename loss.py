import numpy as np

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-9, 1.0)
        self.y_true = y_true
        return -np.mean(np.sum(y_true * np.log(self.y_pred), axis=1))

    def backward(self):
        return (self.y_pred - self.y_true) / len(self.y_true)
