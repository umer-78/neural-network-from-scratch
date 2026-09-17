"""Numerical gradient checking: the safety net for hand-written backprop.

Compares every analytic gradient against a central finite difference. If backprop
is wrong, the relative error jumps from ~1e-9 to something you cannot miss.
"""

from __future__ import annotations

import numpy as np

from .network import Network


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def check_gradients(net: Network, x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Returns the worst relative error over all parameters (want < 1e-5)."""
    pred = net.forward(x, training=False)
    net.backward(net.loss.backward(pred, y))
    worst = 0.0
    for param, grad in net.params_and_grads():
        numeric = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            original = param[idx]
            param[idx] = original + eps
            plus = net.loss.forward(net.forward(x, training=False), y)
            param[idx] = original - eps
            minus = net.loss.forward(net.forward(x, training=False), y)
            param[idx] = original
            numeric[idx] = (plus - minus) / (2 * eps)
            it.iternext()
        worst = max(worst, relative_error(grad, numeric))
    return worst
