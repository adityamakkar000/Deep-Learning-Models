import numpy as np


class nMLP:
    def __init__(self, learning_rate, *dims):
        self.numLayers = len(dims) - 1  # numLayers - 1
        self.layers = [i for i in range(self.numLayers + 1)]  # numLayers
        self.dLayers = [i for i in range(self.numLayers)]  # (numLayers - 1)
        self.layerWeights = [
            np.random.randn(dims[i + 1], dims[i]) for i in range(0, self.numLayers)
        ]  # (numLayers - 1)
        self.layerBias = [
            np.random.randn(dims[i], 1) for i in range(1, self.numLayers + 1)
        ]  # (numLayers - 1)

        self.lr = learning_rate

    def get_total_params(self):
        total = np.sum(
            np.array([layer.shape[0] * layer.shape[1] for layer in self.layerWeights])
        )
        total += np.sum(np.array([layer.shape[0] for layer in self.layerBias]))
        return total

    def __call__(self, X):
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        self.layers[0] = X
        for l in range(0, self.numLayers - 1):
            self.layers[l + 1] = np.tanh(
                self.layers[l] @ self.layerWeights[l].T + self.layerBias[l].T
            )
        self.layers[self.numLayers] = softmax(
            self.layers[self.numLayers - 1] @ self.layerWeights[self.numLayers - 1].T
            + self.layerBias[self.numLayers - 1].T
        )

        self.output = self.layers[self.numLayers]

        return self.output

    def backward(self, y_true, print_loss=False, train=True):
        def CCE(predictions, y_true):
            # assuming y_true is in the form of 1 hot embeddings
            loss = -1 * np.sum(y_true * np.log(predictions), axis=(0, 1))
            return loss

        def MSE(predictions, y_true):
            loss = (
                0.5
                * (1 / predictions.shape[0])
                * np.sum(np.sum((predictions - y_true) ** 2, axis=-1), axis=0)
            )
            return loss

        loss = CCE(self.output, y_true)  # scalar 1,

        if print_loss:
            statment = f"Loss {loss}"
            print(statment)

        if train == False:
            return loss

        batch_size = self.output.shape[0]
        self.dW = [i for i in range(len(self.layerWeights))]
        self.dB = [i for i in range(len(self.layerBias))]

        """ Use this for self.dLayers if you have tanh at the end instead of softmax with MSE instead of CCE"""
        # self.dOuput = (1/batch_size) * (y_true * (self.output ** -1)) # batch_size x ouput_dim
        # self.dLayers[self.numLayers - 1] = self.dOuput * (1 - (self.output)**2) # batch_size x output_dim

        self.dLayers[self.numLayers - 1] = self.output - y_true

        for l in range(self.numLayers - 2, -1, -1):
            self.dLayers[l] = (self.dLayers[l + 1] @ self.layerWeights[l + 1]) * (
                1 - (self.layers[l + 1]) ** 2
            )

        for l in range(0, self.numLayers):
            self.dW[l] = (self.dLayers[l].T) @ self.layers[l]
            self.dB[l] = np.sum(self.dLayers[l], axis=0, keepdims=True)

            self.layerWeights[l] -= self.lr * self.dW[l]
            self.layerBias[l] -= self.lr * self.dB[l].T

        return loss
