"""Datasets used by the demos. Both ship with scikit-learn, so nothing downloads."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def one_hot(y: np.ndarray, classes: int | None = None) -> np.ndarray:
    classes = classes or int(y.max()) + 1
    return np.eye(classes)[np.asarray(y).astype(int)]


def digits(test_size: float = 0.2, seed: int = 42):
    """8x8 handwritten digits: 1,797 samples, 64 features, 10 classes.

    The fifth return value is the 8x8 image of every *test* sample, in test order,
    so a chart of the mistakes shows the digit the network actually got wrong.
    """
    data = load_digits()
    x = data.data / 16.0  # pixels are 0..16
    index = np.arange(len(x))
    x_train, x_test, y_train, y_test, _, test_idx = train_test_split(
        x, data.target, index, test_size=test_size, random_state=seed, stratify=data.target
    )
    return x_train, x_test, one_hot(y_train, 10), one_hot(y_test, 10), data.images[test_idx]


def moons(n: int = 1000, noise: float = 0.2, seed: int = 42):
    """Two interleaving half-moons: the classic 'needs a hidden layer' problem."""
    x, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    return x_train, x_test, one_hot(y_train, 2), one_hot(y_test, 2)
