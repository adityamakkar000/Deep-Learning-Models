import numpy as np


class MLP:
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

    def backward(self, y_true, prev_grad, train=True):
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

        self.dLayers[self.numLayers - 1] = prev_grad

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


class LayerNorm:

    def __init__(self, last_dim):

        self.gamma = np.random.rand(last_dim)
        self.beta = np.random.rand(last_dim)

    def __call__(self, x):

        mean = np.mean(x, axis=-1)
        std = np.std(x, axis=-1)

        x = (self.gamma / std) * x + self.beta  # broadcast across batch

        return x


# TODO backwards method


class SelfAttention:

    def __init__(self, learning_rate, length, dim):

        self.lr = learning_rate
        self.queries = MLP(self.lr, length, dim)
        self.keys = MLP(self.lr, length, dim)
        self.values = MLP(self.lr, length, dim)

    def __call__(self, x):

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        q = self.queries(x)
        k = self.queries(x)
        v = self.queries(x)

        attention = softmax(np.tril(q @ k.T) / self.queries.shape[1])
        logits = attention @ v

        return logits


class MultiHeadSelfAttention:

    def __init__(self, learning_rate, length, nheads, block_dim):

        self.lr = learning_rate
        head_size = block_dim // nheads
        self.heads = [SelfAttention(self.lr, length, head_size) for _ in range(nheads)]

    def __call__(self, x):

        logits = (head(x) for head in self.heads)
        logits = np.concatenate(*logits, axis=-1)

        return logits


class TransformerBlock:

    def __init__(self, learning_rate, length, nheads, block_dim):

        self.lr = learning_rate
        self.MHA = MultiHeadSelfAttention(
            self.lr, length=length, nheads=nheads, block_dim=block_dim
        )
        self.ln1 = LayerNorm(self.lr, block_dim)
        self.ln2 = LayerNorm(self.lr, block_dim)
        self.feedforward1 = MLP(self.lr, block_dim, 4 * block_dim, block_dim)
        self.feedforward2 = MLP(self.lr, block_dim, 4 * block_dim, block_dim)

    def __call__(self, x):

        x = x + self.feedforward1(self.MHA(self.ln1(x)))
        x = x + self.feedforward2(self.ln2(x))

        return x


class GPT:

    def __init__(self, learning_rate, length, vocab_size, block_dim, nheads, nblocks):

        self.lr = learning_rate

        self.embedding_table = np.random.rand(vocab_size, block_dim)
        self.pos_embedding = np.random.rand(length, block_dim)
        self.blocks = [
            TransformerBlock(self.lr, length, nheads, block_dim) for _ in nblocks
        ]
        self.finalLayerNorm = LayerNorm(self.lr, block_dim)
        self.mlp = MLP(self.lr, block_dim, vocab_size)

    def __call__(self, x):

        def softmax(arr):
            return np.exp(arr) / np.sum(np.exp(arr), axis=1, keepdims=True)

        x = self.embedding_table[x] + self.pos_embedding

        for block in self.blocks:
            x = block(x)

        logits = self.mlp(self.finalLayerNorm(x))

        probs = softmax(logits)

        return probs

