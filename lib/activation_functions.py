import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def linear(x):
    return x


def linear_derivative(x):
    return np.ones_like(x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(x):
    eps = 1e-9  # small epsilon value
    return softmax(x) * (1 - softmax(x) + eps)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)
