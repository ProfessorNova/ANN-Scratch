import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def backpropagation(self, inputs, targets):
        activations = [inputs]  # List to store the activations of each layer
        zs = []  # List to store the raw outputs of each layer (before activation)

        # Feed forward
        for layer in self.layers:
            z = np.dot(activations[-1], layer.weights) + layer.bias
            zs.append(z)
            activations.append(layer.feed_forward(activations[-1]))

        # Initial backpropagation
        # Calculate the error at the output layer (difference between the predicted and the true target values)
        # It is basically the derivative of the loss function with respect to the output of the last layer
        delta = activations[-1] - targets
        # Calculate the loss as mean squared error
        loss = np.sum(delta**2) / 2

        delta_weights = []  # stores the gradients of the weights
        delta_biases = []  # stores the gradients of the biases

        # iterate over the layers in reverse order
        for i in reversed(range(len(self.layers))):

            # The gradient is the dot product of the activations of layer i and the delta of layer i+1
            # Calculate the gradient of the loss with respect to the weights in layer i
            delta_weights.insert(0, np.dot(activations[i].T, delta))
            # Calculate the gradient of the loss with respect to the biases in layer i
            delta_biases.insert(0, np.sum(delta, axis=0, keepdims=True))

            # When not at the input layer, calculate the delta for the next (previous) layer
            if i > 0:
                # Backpropagate the error to the previous layer
                # by multiplying delta the transpose of the weights of the current layer the error is effectively distributed
                # to adjust to the activations of the previous layer it is then multiplied by the derivative of the activation function to get the z values of the previous layer
                delta = np.dot(delta, self.layers[i].weights.T) * self.layers[
                    i - 1
                ].activation_derivative(zs[i - 1])

        return loss, delta_weights, delta_biases

    # TODO: Implement SGD
