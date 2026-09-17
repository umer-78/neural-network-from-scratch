"""Layers. Each one implements forward(x) and backward(grad).

Shapes: a batch is (batch_size, features). Every layer stores what it needs from
the forward pass to compute gradients, and nothing else.
"""

from __future__ import annotations

import numpy as np


class Layer:
    trainable = False

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params_and_grads(self):
        return []


class Dense(Layer):
    """Fully connected layer: y = xW + b."""

    trainable = True

    def __init__(self, n_in: int, n_out: int, weight_init: str = "he", rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        if weight_init == "he":        # good with ReLU
            scale = np.sqrt(2.0 / n_in)
        elif weight_init == "xavier":  # good with tanh/sigmoid
            scale = np.sqrt(1.0 / n_in)
        else:
            raise ValueError(f"unknown weight_init {weight_init!r}")
        self.W = rng.normal(0, scale, (n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._x is not None, "backward() called before forward()"
        # average over the batch so the learning rate does not depend on batch size
        n = self._x.shape[0]
        self.dW = self._x.T @ grad / n
        self.db = grad.mean(axis=0)
        return grad @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU(Layer):
    def forward(self, x, training=True):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask


class Sigmoid(Layer):
    def forward(self, x, training=True):
        self._out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self._out

    def backward(self, grad):
        return grad * self._out * (1 - self._out)


class Tanh(Layer):
    def forward(self, x, training=True):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad):
        return grad * (1 - self._out ** 2)


class Softmax(Layer):
    """Numerically stable softmax. Pair it with CrossEntropy, which folds the
    derivative into a single (p - y) term, so backward() here is the identity."""

    def forward(self, x, training=True):
        shifted = x - x.max(axis=1, keepdims=True)
        e = np.exp(shifted)
        self._out = e / e.sum(axis=1, keepdims=True)
        return self._out

    def backward(self, grad):
        return grad


class Dropout(Layer):
    def __init__(self, p: float = 0.2, rng: np.random.Generator | None = None):
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.rng = rng or np.random.default_rng(0)

    def forward(self, x, training=True):
        if not training or self.p == 0:
            self._mask = None
            return x
        # inverted dropout: scale at training time so inference needs no change
        self._mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * self._mask

    def backward(self, grad):
        return grad if self._mask is None else grad * self._mask
