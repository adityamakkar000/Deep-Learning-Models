import numpy as np


class LR:

    def __init__(self, in_dim, learning_rate):

        self.lr = learning_rate

        self.weights = np.random.randn(in_dim, 1)
        self.bias = np.random.randn(1)
        self.losses = []

    def __call__(self, X):

        result = X @ self.weights + self.bias
        return result

    def backward(self, X, y_true, print_loss=False):

        result = X @ self.weights + self.bias

        batch_size = y_true.shape[0]
        loss = (1 / (2 * batch_size)) * (np.linalg.norm(result - y_true)) ** 2
        self.losses.append(loss)
        if print_loss:
            print(loss)

        dL = (1 / batch_size) * (result - y_true)
        dW = X.T @ dL
        dB = np.sum(dL)

        if np.linalg.norm(dW) > 0.5:
            dW = 1 * (dW / np.linalg.norm(dW))

        self.weights = self.weights - self.lr * dW
        self.bias = self.bias - self.lr * dB
