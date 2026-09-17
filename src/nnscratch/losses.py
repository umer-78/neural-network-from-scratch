"""Loss functions. `backward` returns dL/d(prediction)."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


class Loss:
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanSquaredError(Loss):
    def forward(self, pred, target):
        return float(np.mean((pred - target) ** 2))

    def backward(self, pred, target):
        return 2 * (pred - target) / pred.shape[1]


class CrossEntropy(Loss):
    """Categorical cross-entropy for one-hot targets, applied after Softmax.

    The gradient is (p - y): the softmax Jacobian and the log derivative cancel,
    which is both faster and numerically stabler than doing it in two steps.
    """

    def forward(self, pred, target):
        return float(-np.sum(target * np.log(np.clip(pred, EPS, 1.0))) / pred.shape[0])

    def backward(self, pred, target):
        return pred - target
