import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def softmax(x):
    return np.exp(np.array(x)) / np.sum(np.exp(np.array(x)))


def softmax_deriv(x):
    return softmax(x) * (1.0 - softmax(x))
