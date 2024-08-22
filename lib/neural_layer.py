import numpy as np

from lib.activation_functions import *


class NeuralLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

        # Weights are a matrix of shape (input_size x output_size)
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))

        # Bias is a matrix of shape (1 x output_size)
        self.bias = np.random.uniform(-1, 1, (1, output_size))

    def feed_forward(self, inputs):
        # Calculate the dot product of the inputs and weights
        z = np.dot(inputs, self.weights) + self.bias
        # Apply the activation function
        if self.activation_function == "sigmoid":
            return self.sigmoid(z)
        elif self.activation_function == "relu":
            return self.relu(z)
        elif self.activation_function == "softmax":
            return self.softmax(z)
        elif self.activation_function == "linear":
            return z
        return z

    def activation_derivative(self, z):
        # Calculate the derivative of the activation function
        if self.activation_function == "sigmoid":
            return self.sigmoid_derivative(z)
        elif self.activation_function == "relu":
            return self.relu_derivative(z)
        elif self.activation_function == "softmax":
            return self.softmax_derivative(z)
        elif self.activation_function == "linear":
            return self.linear_derivative(z)
        return z

    def update_weights_and_bias(self, delta_weights, delta_bias):
        # Update the weights and bias using the calculated deltas
        self.weights -= delta_weights
        self.bias -= delta_bias
