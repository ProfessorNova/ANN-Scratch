import numpy as np

from lib.neural_layer import NeuralLayer


class NeuralNetwork:
    def __init__(self):
        self.layers: list[NeuralLayer] = []

    def add_layer(self, layer: NeuralLayer):
        self.layers.append(layer)

    def __str__(self):
        layer_descriptions = []
        for i, layer in enumerate(self.layers):
            layer_description = (
                f"Layer {i + 1}: "
                f"{layer.input_size} -> {layer.output_size}, "
                f"Activation: {layer.activation_function}, "
                f"Weights Shape: {layer.weights.shape}, "
                f"Bias Shape: {layer.bias.shape}"
            )
            layer_descriptions.append(layer_description)
        return "\n".join(layer_descriptions)

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def backpropagation(self, inputs, targets):
        activations = [inputs]  # List to store the activations of each layer
        zs = []  # List to store the raw outputs of each layer (before activation)

        # Feed forward to fill the zs and activations lists
        for layer in self.layers:
            z = np.dot(activations[-1], layer.weights) + layer.bias
            zs.append(z)
            activations.append(layer.feed_forward(activations[-1]))

        # Initial backpropagation
        # Calculate the error at the output layer (difference between the predicted and the true target values)
        # It is basically the derivative of the loss function with respect to the output of the last layer
        delta = activations[-1] - targets
        # Calculate the loss as mean squared error
        loss = np.sum(delta ** 2) / 2

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
                # Backpropagate the error to the previous layer the error is effectively distributed to adjust to the
                # activations of the previous layer it is then multiplied by the derivative of the activation
                # function to get the z values of the previous layer
                delta = (np.dot(delta, self.layers[i].weights.T) *
                         self.layers[i - 1].activation_derivative(zs[i - 1]))

        return loss, delta_weights, delta_biases

    def update_batch(self, batch_x, batch_y, learning_rate):
        # Create lists to store the gradients of the weights and biases for each layer
        delta_weights_sum = [np.zeros_like(layer.weights) for layer in self.layers]
        delta_biases_sum = [np.zeros_like(layer.bias) for layer in self.layers]

        total_loss = 0

        # Iterate over the data in the batch
        for x, y in zip(batch_x, batch_y):
            # Perform backpropagation for each data point
            loss, delta_weights, delta_biases = self.backpropagation(x.reshape(1, -1), y.reshape(1, -1))
            total_loss += loss

            # Add the gradients to the total gradients
            for i in range(len(self.layers)):
                delta_weights_sum[i] += delta_weights[i]
                delta_biases_sum[i] += delta_biases[i]

        # Update the weights and biases using the average gradients multiplied by the learning rate
        for i in range(len(self.layers)):
            self.layers[i].update_weights_and_bias(
                learning_rate * delta_weights_sum[i] / len(batch_x),
                learning_rate * delta_biases_sum[i] / len(batch_x)
            )

    def calculate_loss(self, data):
        # Calculate the cross-entropy loss for the given data
        data_x, data_y = data
        total_loss = 0
        for x, y in zip(data_x, data_y):
            output_probs = self.feed_forward(x.reshape(1, -1))
            total_loss += -np.sum(y * np.log(output_probs + 1e-12))
        return total_loss / len(data_x)

    def evaluate(self, test_data):
        # Evaluate the model on the test data
        test_x, test_y = test_data
        correct = 0
        for x, y in zip(test_x, test_y):
            output_probs = self.feed_forward(x.reshape(1, -1))
            if np.argmax(output_probs) == np.argmax(y):
                correct += 1
        return correct / len(test_x)

    def stochastic_gradient_descent(self, train_data, epochs, batch_size, learning_rate, test_data=None, log=True):
        train_x, train_y = train_data
        indices = np.arange(len(train_x))

        for epoch in range(epochs):
            # Shuffle the indices
            np.random.shuffle(indices)

            # Iterate over the batches
            for start_idx in range(0, len(train_x), batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]

                # Update the weights and biases using the current batch
                self.update_batch(batch_x, batch_y, learning_rate)

            if log:
                # Calculate the loss and accuracy on the training data
                loss = self.calculate_loss(train_data)
                accuracy = self.evaluate(train_data)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

                # Calculate the loss and accuracy on the test data
                if test_data:
                    test_loss = self.calculate_loss(test_data)
                    test_accuracy = self.evaluate(test_data)
                    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    def save(self, path, log=True):
        if log:
            print(f"Saving model to {path}")
        # Save the model to a file
        with open(path, 'wb') as f:
            np.save(f, len(self.layers))
            for layer in self.layers:
                np.save(f, layer.input_size)
                np.save(f, layer.output_size)
                np.save(f, layer.activation_function)
                np.save(f, layer.weights)
                np.save(f, layer.bias)
        if log:
            print(f"Saved model:\n{self}")

    def load(self, path, log=True):
        if log:
            print(f"Loading model from {path}")
        # Load the model from a file
        with open(path, 'rb') as f:
            num_layers = np.load(f)
            self.layers = []
            for _ in range(num_layers):
                input_size = np.load(f)
                output_size = np.load(f)
                activation_function = np.load(f)
                layer = NeuralLayer(input_size, output_size, activation_function)
                layer.weights = np.load(f)
                layer.bias = np.load(f)
                self.layers.append(layer)
        if log:
            print(f"Loaded model:\n{self}")
