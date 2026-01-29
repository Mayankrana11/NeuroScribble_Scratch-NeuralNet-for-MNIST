import numpy as np

def accuracy(y_pred, y_true):
    return (y_pred.argmax(axis=1) == y_true.argmax(axis=1)).mean()

def save(model, path="weights.npz"):
    data = {}
    i = 0
    for layer in model.layers:
        if hasattr(layer, "W"):
            data[f"W{i}"] = layer.W
            data[f"b{i}"] = layer.b
            i += 1
    np.savez(path, **data)

def load(model, path="weights.npz"):
    data = np.load(path)
    i = 0
    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W[:] = data[f"W{i}"]
            layer.b[:] = data[f"b{i}"]
            i += 1
