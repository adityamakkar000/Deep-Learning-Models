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

    def backward(self, y_true, train=True):
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

        self.prev = self.dLayers[0] @ self.layerWeights[0]

        for l in range(0, self.numLayers):
            self.dW[l] = (self.dLayers[l].T) @ self.layers[l]
            self.dB[l] = np.sum(self.dLayers[l], axis=0, keepdims=True)

            self.layerWeights[l] -= self.lr * self.dW[l]
            self.layerBias[l] -= self.lr * self.dB[l].T

        return loss


class ClassifierRNN:

    def __init__(self, input_dim, output_dim, learning_rate, dims):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        self.W_h = np.random.rand(output_dim, output_dim)
        self.W_x = np.random.rand(output_dim, input_dim)
        self.Bias = np.zeros((output_dim, 1))

        self.MLP = nMLP(learning_rate, *dims)

    def __call__(self, x):

        self.T, self.B, self.C = x.shape
        self.X = x

        self.hidden_state = np.zeros(
            (self.T + 1, self.B, self.output_dim)
        )  # for past cell for the intial cell

        for t in range(self.T):

            self.hidden_state[t + 1] = np.tanh(
                x[t] @ self.W_x.T + self.hidden_state[t] @ self.W_h.T + self.Bias.T
            )
        output = self.MLP(self.hidden_state[self.T])
        return self.hidden_state, output

    def backward(self, y_true, train=True):

        self.dW_h = np.zeros_like(self.W_h)
        self.dW_x = np.zeros_like(self.W_x)
        self.dB = np.zeros_like(self.Bias)

        loss = self.MLP.backward(y_true, train=train)

        if train == False:
            return loss

        self.dOutput = self.MLP.prev
        self.W_h_power = np.ones_like(self.W_h)

        # self.dOutput = (
        #     1 / self.B * (y_true - self.hidden_state[self.T])
        # )  # or import from MLP

        for t in range(self.T, 0, -1):

            self.dOutput *= 1 - (self.hidden_state[t]) ** 2
            self.constant = self.W_h_power @ self.dOutput.T
            self.dW_x += self.constant @ self.X[t - 1]
            self.dW_h += self.constant @ self.hidden_state[t]
            self.dB += np.sum(self.constant, axis=1, keepdims=True)
            self.W_h_power = self.W_h_power @ self.W_h

        self.W_x -= self.lr * self.dW_x
        self.W_h -= self.lr * self.dW_h
        self.Bias -= self.lr * self.dB

        return loss

    def getTotalParams(self):
        return self.MLP.get_total_params() + np.sum(
            np.array(
                [
                    self.W_h.shape[0] * self.W_h.shape[1],
                    self.W_x.shape[0] * self.W_x.shape[1],
                ]
            )
        )
