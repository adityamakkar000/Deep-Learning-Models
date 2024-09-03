import numpy as np


class MLP:
    def __init__(
        self, learning_rate, *dims, bias=True, activation=True, last_layer=True
    ):
        self.numLayers = len(dims) - 1  # numLayers - 1
        self.layers = [i for i in range(self.numLayers + 1)]  # numLayers
        self.dLayers = [i for i in range(self.numLayers)]  # (numLayers - 1)
        self.layerWeights = [
            np.random.randn(dims[i + 1], dims[i]) * 0.01
            for i in range(0, self.numLayers)
        ]  # (numLayers - 1)

        self.bias = bias
        self.activation = activation
        self.last_layer = last_layer if activation == True else False
        if bias == True:
            self.layerBias = [
                np.zeros((dims[i], 1)) for i in range(1, self.numLayers + 1)
            ]  # (numLayers - 1)

        self.lr = learning_rate

    def get_total_params(self):
        total = np.sum(
            np.array([layer.shape[0] * layer.shape[1] for layer in self.layerWeights])
        )
        if self.bias == True:
            total += np.sum(np.array([layer.shape[0] for layer in self.layerBias]))
        return total

    def __call__(self, X):

        self.layers[0] = X

        for l in range(0, self.numLayers - 1):
            self.layers[l + 1] = self.layers[l] @ self.layerWeights[l].T + (
                self.layerBias[l].T if self.bias == True else 0
            )
            if self.activation == True:
                self.layers[l + 1] = self.layers[l + 1] * (self.layers[l + 1] > 0)

        self.layers[self.numLayers] = self.layers[
            self.numLayers - 1
        ] @ self.layerWeights[self.numLayers - 1].T + (
            self.layerBias[self.numLayers - 1].T if self.bias == True else 0
        )

        if self.last_layer == True:
            self.layers[self.numLayers] = self.layers[self.numLayers] * (
                self.layers[self.numLayers] > 0
            )

        self.output = self.layers[self.numLayers]

        return self.output

    def backward(self, dPrev):

        batch_size = self.output.shape[0]
        self.dW = [i for i in range(len(self.layerWeights))]
        if self.bias == True:
            self.dB = [i for i in range(len(self.layerBias))]

        """ Use this for self.dLayers if you have tanh at the end instead of softmax with MSE instead of CCE"""
        # self.dOuput = (1/batch_size) * (y_true * (self.output ** -1)) # batch_size x ouput_dim
        # self.dLayers[self.numLayers - 1] = self.dOuput * (1 - (self.output)**2) # batch_size x output_dim

        self.dLayers[self.numLayers - 1] = dPrev

        if self.last_layer == True:
            self.dLayers[self.numLayers - 1] = self.dLayers[self.numLayers - 1] * (
                self.layers[self.numLayers] > 0
            )

        for l in range(self.numLayers - 2, -1, -1):
            self.dLayers[l] = self.dLayers[l + 1] @ self.layerWeights[l + 1]
            if self.activation == True:
                self.dLayers[l] = self.dLayers[l] * ((self.layers[l + 1]) > 0)

        self.dPrev = self.dLayers[0] @ self.layerWeights[0]

        for l in range(0, self.numLayers):

            B, T, C = self.layers[l].shape
            self.dW[l] = self.dLayers[l].reshape(B * T, -1).T @ (
                self.layers[l].reshape(B * T, C)
            )

            self.layerWeights[l] -= self.lr * self.dW[l]
            if self.bias == True:
                self.dB[l] = np.expand_dims(
                    np.sum(self.dLayers[l], axis=(0, 1)), axis=0
                )
                self.layerBias[l] -= self.lr * self.dB[l].T

        return self.dPrev


class LayerNorm:

    def __init__(self, learning_rate, last_dim):

        self.lr = learning_rate
        self.last_dim = last_dim
        self.gamma = np.random.rand(1, 1, last_dim) * 0.01
        self.beta = np.zeros((1, 1, last_dim))

    def __call__(self, x):

        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.std = np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)

        self.x = x

        x = (self.gamma / self.std) * (
            x - self.mean
        ) + self.beta  # broadcast across batch

        return x

    def backward(self, dPrev):

        # prev -> (batch, dim)
        self.dGamma = np.sum(
            dPrev * (self.x - self.mean) / self.std, axis=(0, 1), keepdims=True
        )
        self.dBeta = np.sum(dPrev, axis=(0, 1), keepdims=True)

        N = self.last_dim

        dX = (
            dPrev * self.gamma / self.std
            - np.sum(dPrev * self.gamma, axis=-1, keepdims=True) / (N * self.std)
            - (self.x - self.mean)
            / N
            * np.sum(
                dPrev * (self.std**-3) * self.gamma * (self.x - self.mean),
                axis=-1,
                keepdims=True,
            )
        )

        self.aditya = dX

        self.beta -= self.lr * self.dBeta
        self.gamma -= self.lr * self.dGamma

        return dX

    def get_total_params(self):
        params = 2 * self.last_dim
        return params


class SelfAttention:

    def __init__(self, learning_rate, dim, head_size):

        self.lr = learning_rate
        self.keys = MLP(self.lr, dim, head_size, bias=False, activation="False")
        self.queries = MLP(self.lr, dim, head_size, bias=False, activation="False")
        self.values = MLP(self.lr, dim, head_size, bias=False, activation="False")

    def __call__(self, x):

        def softmax(x):
            x_max = np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        k = self.keys(x)
        q = self.queries(x)
        v = self.values(x)

        dotProduct = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[2])
        attention = softmax(
            (np.tril(dotProduct) - np.triu(np.inf * np.ones_like(dotProduct), k=1))
        )
        logits = attention @ v

        self.attention = attention
        self.q = q
        self.k = k
        self.v = v

        return logits

    def backward(self, dPrev):

        dLogits = dPrev  # batch x l x channel

        dAttention = dLogits @ self.v.transpose(0, 2, 1)
        dDotProduct = (
            dAttention * self.attention
            - np.sum(dAttention * self.attention, axis=-1, keepdims=True)
            * self.attention
        )

        dv = self.attention.transpose(0, 2, 1) @ dLogits
        dk = (1 / np.sqrt(self.q.shape[2])) * dDotProduct.transpose(0, 2, 1) @ self.q
        dq = (1 / np.sqrt(self.q.shape[2])) * dDotProduct @ self.k

        self.dPrev = (
            self.values.backward(dv)
            + self.keys.backward(dk)
            + self.queries.backward(dq)
        )

        return self.dPrev

    def get_total_params(self):

        params = (
            self.queries.get_total_params()
            + self.keys.get_total_params()
            + self.values.get_total_params()
        )
        return params


class MultiHeadSelfAttention:

    def __init__(self, learning_rate, nheads, block_dim):

        self.lr = learning_rate
        head_size = block_dim // nheads
        self.nheads = nheads
        self.heads = [
            SelfAttention(self.lr, block_dim, head_size) for _ in range(nheads)
        ]

    def __call__(self, x):

        logits = np.array([head(x) for head in self.heads])

        logits = np.concatenate(logits, axis=-1)

        return logits

    def backward(self, dPrev):

        dLogits = dPrev
        dHeads = np.array(np.split(dLogits, indices_or_sections=self.nheads, axis=-1))

        self.dPrev = np.sum(
            np.array(
                [
                    self.heads[index].backward(dHeads[index])
                    for index in range(self.nheads)
                ]
            ),
            axis=0,
        )

        return self.dPrev

    def get_total_params(self):

        params = np.sum([head.get_total_params() for head in self.heads])
        return params


class TransformerBlock:

    def __init__(self, learning_rate, nheads, block_dim):

        self.lr = learning_rate
        self.MHA = MultiHeadSelfAttention(self.lr, nheads=nheads, block_dim=block_dim)
        self.ln1 = LayerNorm(self.lr, block_dim)
        self.ln2 = LayerNorm(self.lr, block_dim)
        self.feedforward1 = MLP(self.lr, block_dim, block_dim, activation=False)
        self.feedforward2 = MLP(
            self.lr, block_dim, 4 * block_dim, block_dim, last_layer=False
        )

    def __call__(self, x):

        self.cp0 = x
        self.cp1 = self.ln1(x)
        self.cp2 = self.feedforward1(self.MHA(self.cp1))
        x = x + self.feedforward1(self.MHA(self.ln1(x)))
        self.cp3 = x
        x = x + self.feedforward2(self.ln2(x))
        self.cp4 = x

        return x

    def backward(self, dPrev):

        dLogits = dPrev

        dff2 = self.feedforward2.backward(dLogits)
        dln2 = self.ln2.backward(dff2)

        dInter = dLogits + dln2

        dff1 = self.feedforward1.backward(dInter)
        dAttention = self.MHA.backward(dff1)
        dLn1 = self.ln1.backward(dAttention)

        dX = dLn1 + dInter
        return dX

    def get_total_params(self):
        params = (
            self.MHA.get_total_params()
            + self.ln1.get_total_params()
            + self.ln2.get_total_params()
            + self.feedforward1.get_total_params()
            + self.feedforward2.get_total_params()
        )
        return params


class GPT:

    def __init__(self, learning_rate, length, vocab_size, block_dim, nheads, nblocks):

        self.lr = learning_rate
        self.length = length
        self.vocab_size = vocab_size

        self.embedding_table = np.random.rand(vocab_size, block_dim) * 0.01
        self.pos_embedding = np.random.rand(length, block_dim) * 0.01
        self.blocks = [
            TransformerBlock(self.lr, nheads, block_dim) for _ in range(nblocks)
        ]
        self.finalLayerNorm = LayerNorm(self.lr, block_dim)
        self.mlp = MLP(self.lr, block_dim, vocab_size, last_layer=False)

    def __call__(self, x):

        def softmax(x):
            x_max = np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        self.indices = x
        x = x @ self.embedding_table + self.pos_embedding

        for block in self.blocks:
            x = block(x)

        self.logits = self.mlp(self.finalLayerNorm(x))
        self.predictions = softmax(self.logits)
        return self.logits, self.predictions

    def backward(self, y_true, train=True):

        def CCE(logits, y_true):
            # assuming y_true is in the form of 1 hot embeddings

            ls = (
                logits
                - np.max(logits, axis=-1, keepdims=True)
                - np.log(
                    np.sum(
                        np.exp(logits - np.max(logits, axis=-1, keepdims=True)),
                        axis=-1,
                        keepdims=True,
                    )
                )
            )

            loss = -1 * np.sum(y_true * ls)
            return loss

        def MSE(predictions, y_true):
            loss = (
                0.5
                * (1 / predictions.shape[0])
                * np.sum(np.sum((predictions - y_true) ** 2, axis=-1), axis=0)
            )
            return loss

        loss = CCE(self.logits, y_true)  # scalar 1,

        if train == False:
            return loss

        dLoss = self.predictions - y_true

        dFinalMlp = self.mlp.backward(dLoss)
        dFinalLayerNorm = self.finalLayerNorm.backward(dFinalMlp)

        dBlock = dFinalLayerNorm

        for block in reversed(self.blocks):
            dBlock = block.backward(dBlock)

        dEmbedded = np.sum(self.indices.transpose(0, 2, 1) @ dBlock, axis=0)
        self.embedding_table -= self.lr * dEmbedded
        self.pos_embedding -= self.lr * np.sum(dBlock, axis=0)

        return loss

    def get_total_params(self):

        params = 0
        params += self.embedding_table.shape[0] * self.embedding_table.shape[1]
        params += self.pos_embedding.shape[0] * self.pos_embedding.shape[1]
        params += sum([block.get_total_params() for block in self.blocks])
        params += self.finalLayerNorm.get_total_params()
        params += self.mlp.get_total_params()

        return params

    def inference(self, character, max_tokens):

        def one_hot(array, vocab_size):
            length = array.size
            ans = np.zeros((length, vocab_size))
            ans[np.arange(length), array] = 1
            return ans

        response = [encode(i)[0] for i in character]
        while len(response) < max_tokens + 1:

            if len(response) < self.length:
                padding = self.length - len(response)
                current_sequence = [encode(".")[0] for i in range(padding)]
                for c in response:
                    current_sequence.append(c)
            else:
                current_sequence = response[-self.length]

            current_context = one_hot(np.array(current_sequence), self.vocab_size)
            current_context = np.expand_dims(current_context, axis=(0))

            logits, probs = self(current_context)
            probs = probs[0, -1]
            next_token = np.random.choice(
                [i for i in range(self.vocab_size)], 1, [i for i in probs]
            )[0]
            response.append(next_token)

        return response
