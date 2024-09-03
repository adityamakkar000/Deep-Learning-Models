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
        for l in range(0, self.numLayers ):
            self.layers[l + 1] = np.tanh(
                self.layers[l] @ self.layerWeights[l].T + self.layerBias[l].T
            )
        self.output = self.layers[self.numLayers]

        return self.output

    def backward(self, dPrev):

        batch_size = self.output.shape[0]
        self.dW = [i for i in range(len(self.layerWeights))]
        self.dB = [i for i in range(len(self.layerBias))]

        """ Use this for self.dLayers if you have tanh at the end instead of softmax with MSE instead of CCE"""
        self.dOutput = dPrev
        self.dLayers[self.numLayers - 1] = self.dOuput * (1 - (self.output)**2) # batch_size x output_dim


        for l in range(self.numLayers - 2, -1, -1):
            self.dLayers[l] = (self.dLayers[l + 1] @ self.layerWeights[l + 1]) * (
                1 - (self.layers[l + 1]) ** 2
            )

        self.dPrev = self.dLayer[0] @ self.layersWeights[0]
        for l in range(0, self.numLayers):
            self.dW[l] = (self.dLayers[l].T) @ self.layers[l]
            self.dB[l] = np.sum(self.dLayers[l], axis=0, keepdims=True)

            self.layerWeights[l] -= self.lr * self.dW[l]
            self.layerBias[l] -= self.lr * self.dB[l].T

        return self.dPrev

class encoder:

  def __init__(self, learning_rate, dims) -> None:

    self.lr = learning_rate
    self.encoding_mlp = nMLP(self.lr, *dims[:-1])

    self.mean_weights = np.random.rand(*dims[-1:], *dims[-2:-1])
    self.mean_bias = np.random.rand(*dims[-2:-1], 1)
    self.std_weights = np.random.rand(*dims[-1:], *dims[-2:-1])
    self.std_bias = np.random.rand(*dims[-2:-1], 1)


  def __call__(self, x):


    self.inter1 = self.encoding_mlp(x)


    mean = np.einsum('...ij,...kj,...k->ik', self.inter1, self.mean_weights, self.mean_bias)
    std = np.einsum('...ih, ...kj, ...k->ik', self.inter1, self.std_weights, self.std_bias)

    mean = self.inter1 @ self.mean_weights.T + self.mean_bias.T
    std = self.inter1 @ self.std_weights.T + self.std_bias.T

    return mean, std


  def backward(self, *args: np.Any, **kwds: np.Any) -> np.Any:
    pass



class decoder:

  def __init__(self) -> None:
    pass

  def __call__(self, *args: np.Any, **kwds: np.Any) -> np.Any:
    pass

  def backward(self, *args: np.Any, **kwds: np.Any) -> np.Any:
    pass


class VAE:

  def __init__(self) -> None:
    pass

  def __call__(self, *args: np.Any, **kwds: np.Any) -> np.Any:
    pass

  def backward(self, *args: np.Any, **kwds: np.Any) -> np.Any:
    pass
