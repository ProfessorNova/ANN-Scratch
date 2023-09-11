import pandas as pd
import numpy as np
import random

# Test to see if data is loaded correctly

# from PIL import Image

# def load_data(path):
#     data = pd.read_csv(path)
#     # just keep first row
#     data = data.iloc[0, 1:] # should be 5
#     # reshape to 28x28
#     data = data.values.reshape(28, 28)
#     # convert to image
#     data = Image.fromarray(data.astype('uint8'))
#     # show image
#     data.show()
#     return data

# print(load_data('mnist_train.csv'))

# usefull functions----------------------------------------------

def load_data(path):
    data = pd.read_csv(path)
    # data_x -> features
    data_x = data.iloc[:, 1:]
    # normalize data
    data_x = data_x / 255.0
    # data_y -> labels
    data_y = data.iloc[:, 0]
    desired_output = []
    for i in range(len(data_y)):
        desired_output.append(np.zeros(10))
        desired_output[i][data_y.iloc[i]] = 1
    data_y = pd.DataFrame(desired_output)
    return data_x, data_y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def linear(x):
    return x

def linear_derivative(x):
    return 1

def softmax(outputs):
    probabilities = np.exp(outputs) / np.sum(np.exp(outputs))
    return probabilities

def softmax_derivative(outputs):
    probabilities = softmax(outputs)
    return probabilities * (1 - probabilities)

# def relu(x):
#     return np.maximum(0, x)

# Neural Network Classes-----------------------------------------

class Neuron:
    def __init__(self, num_inputs: int=None, activation_function: str='sigmoid'):
        """
        Neural Network Neuron
        """
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        # apply random weights to inputs between -1 and 1
        self.weights = []
        if self.num_inputs is not None:
            for i in range(self.num_inputs):
                self.weights.append(random.uniform(-1, 1))
        # apply random bias
        self.bias = np.random.rand(1)
    
    def update_neuron_structure(self, num_inputs: int, 
                                activation_function='sigmoid'):
        """
        This function will update the structure of the neuron
        will also reset weights and bias!!!
        """
        self.num_inputs = num_inputs
        self.activation_function = activation_function
        self.weights = []
        if self.num_inputs is not None:
            for i in range(self.num_inputs):
                self.weights.append(random.uniform(-1, 1))
        self.bias = np.random.rand(1)

    def update_weights_and_bias(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def get_z(self, inputs):
        # z is the dot product of inputs and weights without activation function
        return np.dot(inputs, self.weights) + self.bias

    def feed_forward(self, inputs):
        # dot product of inputs and weights
        if self.activation_function == 'sigmoid':
            self.output = sigmoid(np.dot(inputs, self.weights) + self.bias)
        else: 
            # for linear and softmax activation functions
            # softmax needs to be done at layer level
            self.output = linear(np.dot(inputs, self.weights) + self.bias)
        return self.output


class NeuralLayer:
    def __init__(self, num_neurons: int, neuron_prototype: Neuron=Neuron(), 
                 type: str='hidden'):
        self.type = type
        self.num_neurons = num_neurons
        self.neuron_prototype = neuron_prototype
        self.neurons = []
        for i in range(self.num_neurons):
            self.neurons.append(Neuron())

    def get_weights(self):
        weights = []
        for neuron in self.neurons:
            weights.append(np.array(neuron.get_weights()))
        weights = np.array(weights)
        return weights
    
    def get_biases(self):
        bias = []
        for neuron in self.neurons:
            bias.append(neuron.get_bias())
        bias = np.array(bias)
        bias = bias.reshape(len(bias),)
        return bias

    def get_z(self, inputs):
        # return Z for each neuron
        if self.type == 'input':
            return inputs
        z = []
        for neuron in self.neurons:
            z.append(neuron.get_z(inputs))
        z = np.array(z)
        z = z.reshape(len(z),)
        return z

    def update_weights_and_bias(self, weights, bias):
        for i in range(len(self.neurons)):
            self.neurons[i].update_weights_and_bias(weights[i], bias[i])

    def feed_forward(self, inputs):
        outputs = []

        # use linear activation function for input layer
        if self.type == 'input':
            # just feed forward with no changes
            for input in inputs:
                outputs.append(input)

        # use softmax activation function for output layer
        elif self.type == 'output':
            # feed forward and get outputs
            for neuron in self.neurons:
                outputs.append(neuron.feed_forward(inputs))
            # apply softmax activation function
            outputs = softmax(outputs)
        
        # use sigmoid activation function for hidden layers
        elif self.type == 'hidden':
            # feed forward and get outputs
            for neuron in self.neurons:
                outputs.append(neuron.feed_forward(inputs))
        # convert to numpy array
        outputs = np.array(outputs)
        # reshape to 1D array
        outputs = outputs.reshape(len(outputs),)
        return outputs
    

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.leanring_rate = 0.01

    def add_layer(self, layer: NeuralLayer, output_layer: bool=False):
        # in case of output layer
        if output_layer:
            layer.type = 'output'
            # update neuron structure for all neurons in layer
            for neuron in layer.neurons:
                neuron.update_neuron_structure(self.layers[-1].num_neurons, 
                                               activation_function='softmax')
        
        # first layer is input layer
        elif len(self.layers) == 0:
            # will just feed forward the inputs with no changes
            layer.type = 'input'
            # update neuron structure for all neurons in layer
            for neuron in layer.neurons:
                neuron.update_neuron_structure(None, activation_function='linear')

        # all other layers are hidden layers
        elif len(self.layers) > 0:
            layer.type = 'hidden'
            # update neuron structure for all neurons in layer
            for neuron in layer.neurons:
                # get number of inputs from previous layer (number of neurons)
                neuron.update_neuron_structure(self.layers[-1].num_neurons,
                                               activation_function='sigmoid')
        self.layers.append(layer)

    def feed_forward(self, inputs):
        outputs = []
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
            #print(f"{layer.type} layer with {layer.num_neurons} neurons -> outputs:")
            #print(inputs)
        outputs = inputs
        return outputs

    def stochastic_gradient_descent(self, train_data: tuple, epochs: int, batch_size: int, learning_rate: float, 
                                    test_data: tuple=None):
        """
        Train neural network using stochastic gradient descent
        """
        train_x, train_y = train_data
        n = len(train_x) # number of training examples

        # if test data is not provided, use train data for testing
        if test_data is None:
            test_data = train_data

        for i in range(epochs):
            print(f"Epoch {i+1}")
            # shuffle data
            print("Shuffling data...",  end='\r')
            train_data = self.shuffle_data(train_data)
            print("Shuffling data... Done")
            train_x, train_y = train_data

            # create batches
            batches = []
            for j in range(0, n, batch_size):
                batches.append((train_x.iloc[j:j+batch_size], train_y.iloc[j:j+batch_size]))
            progress = 0
            for batch in batches:
                progress += 1
                if progress < len(batches):
                    print(f"Progress: {progress}/{len(batches)}", end='\r')
                else:
                    print(f"Progress: Finished!")
                self.update_batch(batch, batch_size, learning_rate)
            # test neural network after each epoch
            accuracy = self.test(test_data)
            accuracy = round(accuracy*100, 2)
            print(f"Accuracy: {accuracy}%")

    def shuffle_data(self, data: tuple):
        """
        Shuffle data
        """
        data_x, data_y = data
        data = pd.concat([data_x, data_y], axis=1)
        data = data.sample(frac=1)
        data_x = data.iloc[:, :-10]
        data_y = data.iloc[:, -10:]
        return data_x, data_y

    def update_batch(self, batch, batch_size, learning_rate):
        delta_weights = []
        delta_biases = []

        for layer in self.layers:
            if layer.type == 'input':
                # input layer has no weights and biases
                continue
            # initialize delta weights and biases as zeros
            delta_weights.append(np.zeros(layer.get_weights().shape))
            delta_biases.append(np.zeros(layer.get_biases().shape))
            # shape with:
            # input layer: 784
            # hidden layer: 16
            # hidden layer: 16
            # output layer: 10
            # is [(16, 784), (16, 16), (10, 16)] for weights
            # and [(16,), (16,), (10,)] for biases

        for i in range(batch_size):
            gradients = self.back_propagation(batch[0].iloc[i].values, batch[1].iloc[i].values)
            for j in range(len(gradients)):
                # add gradients to delta weights and biases
                delta_weights[j] += gradients[j][0]
                delta_biases[j] += gradients[j][1]
                # average delta weights and biases
                delta_weights[j] /= batch_size
                delta_biases[j] /= batch_size

        # update weights and biases
        for i in range(len(self.layers)):
            if self.layers[i].type == 'input':
                # input layer has no weights and biases
                continue
            weights = self.layers[i].get_weights()
            biases = self.layers[i].get_biases()
            weights -= learning_rate * delta_weights[i-1] # i-1 because delta_weights has no input layer
            biases -= learning_rate * delta_biases[i-1] # i-1 because delta_biases has no input layer
            self.layers[i].update_weights_and_bias(weights, biases)

    def back_propagation(self, inputs, desired_output):
        """
        Determine how a single training example would change the weights and biases
        """
        gradients = []
        # shape with:
        # input layer: 784
        # hidden layer: 16
        # hidden layer: 16
        # output layer: 10
        # is [(16, 784), (16, 16), (10, 16)] for weights
        # and [(16,), (16,), (10,)] for biases

        activations = [] # outputs of each layer with activation function applied
        z = [] # outputs of each layer without activation function applied
        for layer in self.layers:
            activations.append(layer.feed_forward(inputs))
            z.append(layer.get_z(inputs))
            inputs = activations[-1] # outputs of previous layer are inputs of next layer

        # Calculate gradients for output layer
        loss = activations[-1] - desired_output
        delta_z = loss

        # reverse everything
        activations = activations[::-1]
        z = z[::-1]
        reversed_layers = self.layers[::-1]

        # Backpropagate gradients
        for i, layer in enumerate(reversed_layers):
            # no calculation for input layer
            if layer.type == 'input':
                continue
            delta_biases = delta_z #/ len(activations[i])
            delta_weights = np.outer(delta_z, activations[i+1]) #/ len(activations[i])
            if layer.type == 'output':
                delta_z = np.matmul(layer.get_weights().T, delta_z) * softmax_derivative(z[i+1])
            else:
                delta_z = np.matmul(layer.get_weights().T, delta_z) * sigmoid_derivative(z[i+1])
            gradients.append((delta_weights, delta_biases))
        # reverse gradients
        gradients = gradients[::-1]
        return gradients
    
    def test(self, test_data: tuple):
        """
        Test neural network
        """
        data_x, data_y = test_data
        correct = 0
        for i in range(len(data_x)):
            # feed forward
            outputs = self.feed_forward(data_x.iloc[i].values)
            # get prediction
            prediction = np.argmax(outputs)
            # get correct output
            correct_output = np.argmax(data_y.iloc[i].values)
            # check if prediction is correct
            if prediction == correct_output:
                correct += 1
        # calculate accuracy
        accuracy = correct / len(data_x)
        return accuracy
    
# Test Neural Network--------------------------------------------

def main():
    # load data
    train_x, train_y = load_data('mnist_train.csv')
    test_x, test_y = load_data('mnist_test.csv')
    # create neural network
    neural_network = NeuralNetwork()
    # add layers
    neural_network.add_layer(NeuralLayer(784)) # first layer = input layer
    neural_network.add_layer(NeuralLayer(128)) # hidden layer
    neural_network.add_layer(NeuralLayer(10), output_layer=True) # output layer

    # train neural network
    neural_network.stochastic_gradient_descent(train_data=(train_x, train_y), epochs=100, 
                                               batch_size=12, learning_rate=0.01,
                                               test_data=(test_x, test_y))

if __name__ == '__main__':
    main()