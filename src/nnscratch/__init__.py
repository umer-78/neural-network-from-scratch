"""A small neural network library written with NumPy only."""

from .layers import Dense, Dropout, ReLU, Sigmoid, Softmax, Tanh
from .losses import CrossEntropy, MeanSquaredError
from .network import Network
from .optimisers import SGD, Adam

__all__ = ["Adam", "CrossEntropy", "Dense", "Dropout", "MeanSquaredError", "Network",
           "ReLU", "SGD", "Sigmoid", "Softmax", "Tanh"]
__version__ = "1.0.0"
