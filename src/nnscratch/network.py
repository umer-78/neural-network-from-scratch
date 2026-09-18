"""The network: a list of layers, a loss and an optimiser."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .layers import Dense, Layer
from .losses import CrossEntropy, Loss
from .optimisers import SGD, Optimiser


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


class Network:
    def __init__(self, layers: list[Layer], loss: Loss | None = None, optimiser: Optimiser | None = None):
        self.layers = layers
        self.loss = loss or CrossEntropy()
        self.optimiser = optimiser or SGD()
        self.history = History()

    # ----------------------------------------------------------------- forward
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params_and_grads(self):
        for layer in self.layers:
            yield from layer.params_and_grads()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.asarray(x, dtype=float), training=False)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    # ----------------------------------------------------------------- training
    def train_batch(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.forward(x, training=True)
        loss = self.loss.forward(pred, y)
        self.backward(self.loss.backward(pred, y))
        self.optimiser.step(list(self.params_and_grads()))
        return loss

    def fit(self, x, y, *, epochs: int = 20, batch_size: int = 32, validation: tuple | None = None,
            shuffle: bool = True, seed: int = 0, verbose: bool = False, early_stopping: int = 0) -> History:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(seed)
        best, waited, best_state = np.inf, 0, None
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x)) if shuffle else np.arange(len(x))
            losses = []
            for start in range(0, len(x), batch_size):
                idx = order[start:start + batch_size]
                losses.append(self.train_batch(x[idx], y[idx]))
            self.history.loss.append(float(np.mean(losses)))
            self.history.accuracy.append(self.score(x, y))
            line = f"epoch {epoch:>3}  loss {self.history.loss[-1]:.4f}  acc {self.history.accuracy[-1]:.4f}"
            if validation is not None:
                vx, vy = np.asarray(validation[0], dtype=float), np.asarray(validation[1], dtype=float)
                vloss = self.loss.forward(self.forward(vx, training=False), vy)
                vacc = self.score(vx, vy)
                self.history.val_loss.append(vloss)
                self.history.val_accuracy.append(vacc)
                line += f"  val_loss {vloss:.4f}  val_acc {vacc:.4f}"
                if early_stopping:
                    if vloss < best - 1e-4:
                        best, waited = vloss, 0
                        best_state = self.get_weights()
                    else:
                        waited += 1
                        if waited >= early_stopping:
                            if verbose:
                                print(f"{line}\nstopping early: no improvement for {early_stopping} epochs")
                            if best_state is not None:
                                self.set_weights(best_state)
                            break
            if verbose:
                print(line)
        return self.history

    def score(self, x, y) -> float:
        pred = self.predict(x)
        truth = np.asarray(y).argmax(axis=1) if np.ndim(y) > 1 else np.asarray(y)
        return float((pred == truth).mean())

    # ----------------------------------------------------------------- weights
    def get_weights(self) -> list[np.ndarray]:
        return [p.copy() for layer in self.layers for p, _ in layer.params_and_grads()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        params = [p for layer in self.layers for p, _ in layer.params_and_grads()]
        if len(params) != len(weights):
            raise ValueError(f"expected {len(params)} arrays, got {len(weights)}")
        # strict: a saved file with a different number of arrays is a corrupted
        # or mismatched checkpoint, and silently loading half of it is worse
        # than refusing.
        for param, value in zip(params, weights, strict=True):
            param[...] = value

    def save(self, path: str | Path) -> None:
        arrays = {f"w{i}": w for i, w in enumerate(self.get_weights())}
        shape = [(type(layer).__name__, getattr(layer, "W", np.empty((0, 0))).shape) for layer in self.layers]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, meta=json.dumps([[n, list(s)] for n, s in shape]), **arrays)

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path) if str(path).endswith(".npz") else f"{path}.npz", allow_pickle=False)
        self.set_weights([data[f"w{i}"] for i in range(len(data.files) - 1)])


def mlp(sizes: list[int], *, activation=None, dropout: float = 0.0, seed: int = 0,
        optimiser: Optimiser | None = None) -> Network:
    """Build a classifier: Dense/ReLU stack ending in Dense + Softmax."""
    from .layers import Dropout, ReLU, Softmax

    rng = np.random.default_rng(seed)
    act = activation or ReLU
    layers: list[Layer] = []
    for i in range(len(sizes) - 1):
        layers.append(Dense(sizes[i], sizes[i + 1], rng=rng))
        if i < len(sizes) - 2:
            layers.append(act())
            if dropout:
                layers.append(Dropout(dropout, rng=rng))
    layers.append(Softmax())
    return Network(layers, CrossEntropy(), optimiser or SGD(0.1, momentum=0.9))
