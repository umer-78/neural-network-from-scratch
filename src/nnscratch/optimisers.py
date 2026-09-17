"""Optimisers. `step` updates parameters in place from the gradients."""

from __future__ import annotations

import numpy as np


class Optimiser:
    def step(self, params_and_grads) -> None:
        raise NotImplementedError


class SGD(Optimiser):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(self, lr: float = 0.1, momentum: float = 0.0, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: dict[int, np.ndarray] = {}

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            g = grad + self.weight_decay * param if self.weight_decay else grad
            if self.momentum:
                v = self._velocity.get(id(param))
                v = self.momentum * v - self.lr * g if v is not None else -self.lr * g
                self._velocity[id(param)] = v
                param += v
            else:
                param -= self.lr * g


class Adam(Optimiser):
    """Adam with bias correction (Kingma & Ba, 2014)."""

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.weight_decay = weight_decay
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}
        self._t = 0

    def step(self, params_and_grads):
        self._t += 1
        for param, grad in params_and_grads:
            g = grad + self.weight_decay * param if self.weight_decay else grad
            key = id(param)
            m = self._m.get(key, np.zeros_like(param))
            v = self._v.get(key, np.zeros_like(param))
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g ** 2
            self._m[key], self._v[key] = m, v
            m_hat = m / (1 - self.beta1 ** self._t)
            v_hat = v / (1 - self.beta2 ** self._t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
