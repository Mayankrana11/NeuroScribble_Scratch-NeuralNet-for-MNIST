import numpy as np
import gzip
import os
import urllib.request

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = "mnist_data"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in MNIST_URLS.items():
        filename = url.split("/")[-1]
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, path)


def load_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def one_hot(y, num_classes=10):
    out = np.zeros((y.size, num_classes))
    out[np.arange(y.size), y] = 1
    return out


def load_mnist():
    download()
    X_train = load_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"))
    y_train = one_hot(load_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")))
    X_test  = load_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"))
    y_test  = one_hot(load_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")))
    return X_train, y_train, X_test, y_test


def batches(X, y, batch_size):
    idx = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        j = idx[i:i + batch_size]
        yield X[j], y[j]
