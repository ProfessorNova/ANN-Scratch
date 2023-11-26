import numpy as np
import pandas as pd
import logging
from fastprogress import master_bar, progress_bar
from types import SimpleNamespace
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

config = SimpleNamespace(
    run_name="more_complex_run",
    epochs=50,  # Keeping a higher number of epochs
    batch_size=64,  # Same batch size as before
    learning_rate=0.05,  # Starting with a slightly lower learning rate
    lr_decay_rate=0.95,  # Less aggressive decay rate
    lr_decay_frequency=5,  # Decay learning rate every 5 epochs
    early_stop_patience=10,  # More patience before early stopping
    train_data_path="mnist_train.csv",
    test_data_path="mnist_test.csv",
)

# USEFULL FUNCTIONS

def load_data(path):
    """
    Load data from csv file.
    (data loader for MNIST dataset)
    """
    logging.info(f"Loading {path}")
    data = pd.read_csv(path)
    # data_x -> input data
    # data_y -> output data
    data_x = data.iloc[:, 1:]
    data_y = data.iloc[:, 0]
    # normalize data
    data_x = data_x / 255.0
    # convert labels to one hot vectors
    desired_output = []
    for i in range(len(data_y)):
        desired_output.append(np.zeros(10))
        desired_output[i][data_y.iloc[i]] = 1.0
    data_y = pd.DataFrame(desired_output)
    logging.info(f"Loaded {path}")
    return data_x, data_y

def shuffle_data(data: tuple):
    """
    Shuffle data.
    The data is a tuple of (data_x, data_y).
    (data shuffler for MNIST dataset)
    """
    data_x, data_y = data
    data = pd.concat([data_x, data_y], axis=1)
    data = data.sample(frac=1)
    data_x = data.iloc[:, :-10]
    data_y = data.iloc[:, -10:]
    return data_x, data_y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def linear(z):
    return z

def linear_derivative(z):
    return 1

def softmax(vector):
    # test if a value is too large or too small
    if np.any(vector > 709):
        vector = np.where(vector > 709, 709, vector)
    if np.any(vector < -709):
        vector = np.where(vector < -709, -709, vector)
    return np.exp(vector) / np.sum(np.exp(vector))

def softmax_derivative(vector):
    return softmax(vector) * (1 - softmax(vector))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# NN CLASSES

class Neuron:
    """
    Neuron for the Neural Network
    """
    def __init__(self, num_inputs: int, activation_function: str=None):
        """
        Supports sigmoid, linear and softmax activation functions.
        Weights and biases are initialized randomly between -1 and 1
        """
        self.activation_function = activation_function
        self.num_inputs = num_inputs
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.bias = np.random.uniform(-1, 1)

    def update_structure(self, num_inputs: int,
                            activation_function: str=None):
            """
            Update structure of neuron
            (this will reset the weights and biases)
            """
            self.activation_function = activation_function
            self.num_inputs = num_inputs
            self.weights = np.random.uniform(-1, 1, num_inputs)
            self.bias = np.random.uniform(-1, 1)

    def get_z(self, inputs):
        # output of neuron before activation function is applied
        return np.dot(inputs, self.weights) + self.bias
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    def update_weights_and_bias(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed_forward(self, inputs):
        """
        Returns the output of the neuron with activation function applied
        """
        z = self.get_z(inputs)
        if self.activation_function == "sigmoid":
            return sigmoid(z)
        elif self.activation_function == "relu":
            return relu(z)
        elif self.activation_function == "linear":
            return linear(z)
        elif self.activation_function == "softmax":
            # softmax has to be applied at layer level
            return z
        elif self.activation_function is None:
            raise Exception("Activation function not defined")
        else:
            raise Exception("Activation function not supported")
        
class NeuralLayer:
    """
    Layer for the Neural Network
    """
    def __init__(self, num_neurons: int, type: str, num_inputs: int=None,
                    activation_function: str=None, dropout_rate: float=0.0):
        """
        Type can be input, hidden or output
        """
        self.num_neurons = num_neurons
        self.type = type
        self.dropout_rate = dropout_rate
        self.neurons = []
        # test if activation function is supported
        if type == "input":
            if activation_function is not None:
                raise Exception("Input layer cannot have activation function")
        elif type == "hidden":
            if activation_function not in ["sigmoid", "relu", "linear", "softmax", None]:
                raise Exception("Activation function not supported")
        elif type == "output":
            if activation_function not in ["sigmoid", "relu", "linear", "softmax", None]:
                raise Exception("Activation function not supported")
            
        self.activation_function = activation_function
        for i in range(self.num_neurons):
            self.neurons.append(Neuron(num_inputs, activation_function))

    def update_structure(self, num_neurons: int, type: str, num_inputs: int,
                            activation_function: str):
        """
        Update structure of layer
        (this will reset the weights and biases)
        """
        self.num_neurons = num_neurons
        self.type = type
        self.neurons = []
        self.activation_function = activation_function
        for i in range(self.num_neurons):
            self.neurons.append(Neuron(num_inputs, activation_function))

    def get_num_neurons(self):
        return self.num_neurons
    
    def get_type(self):
        return self.type

    def get_activation_function(self):
        return self.activation_function

    def get_z(self, inputs):
        # return input if layer is input layer
        if self.type == "input":
            return inputs
        z = []
        for neuron in self.neurons:
            z.append(neuron.get_z(inputs))
        # convert to numpy array
        z = np.array(z)
        z = z.reshape(len(z),)
        return z
    
    def get_weights(self):
        weights = []
        for neuron in self.neurons:
            weights.append(neuron.get_weights())
        # convert to numpy array
        weights = np.array(weights)
        return weights
    
    def get_bias(self):
        bias = []
        for neuron in self.neurons:
            bias.append(neuron.get_bias())
        # convert to numpy array
        bias = np.array(bias)
        bias = bias.reshape(len(bias),)
        return bias
    
    def update_weights_and_bias(self, weights, bias):
        for i in range(len(self.neurons)):
            self.neurons[i].update_weights_and_bias(weights[i], bias[i])

    def feed_forward(self, inputs, train_mode: bool=False):
        """
        Returns the output of the layer with activation function applied
        """
        # return input if layer is input layer
        if self.type == "input":
            return inputs
        z = self.get_z(inputs)
        # apply dropout if layer is not input layer and train_mode is True
        if self.type != "input" and train_mode and self.dropout_rate > 0:
            # Apply dropout during training
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=z.shape)
            z *= dropout_mask
        # apply activation function to whole layer for performance
        if self.activation_function == "sigmoid":
            return sigmoid(z)
        elif self.activation_function == "relu":
            return relu(z)
        elif self.activation_function == "linear":
            return linear(z)
        elif self.activation_function == "softmax":
            return softmax(z)
        elif self.activation_function is None:
            raise Exception("Activation function not defined")
        
class NeuralNetwork:
    """
    Neural Network
    """
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: NeuralLayer):
        """
        The first layer added will be set as the input layer.
        The last layer added will be the output layer
        and has to be specified as such.
        The number of neurons has to be specified for all layers.
        The number of inputs will be overwritten.
        If no activation function is specified, it will be set appropriately.
        """
        # handle input layer
        if len(self.layers) == 0:
            # first layer added will be the input layer
            layer.update_structure(layer.get_num_neurons(), "input", None, None)
        # handle output layer
        elif layer.get_type() == "output":
            # last layer added will be the output layer
            set_activation_function = layer.get_activation_function()
            if set_activation_function is None:
                # if no activation function for output layer, set softmax
                set_activation_function = "softmax"
            layer.update_structure(layer.get_num_neurons(), "output",
                                    self.layers[-1].get_num_neurons(),
                                    set_activation_function)
        # handle hidden layers
        else:
            set_activation_function = layer.get_activation_function()
            if set_activation_function is None:
                # if no activation function for hidden layer, set sigmoid
                set_activation_function = "sigmoid"
            layer.update_structure(layer.get_num_neurons(), "hidden",
                                    self.layers[-1].get_num_neurons(),
                                    set_activation_function)
        # add layer to network
        self.layers.append(layer)

    def feed_forward(self, inputs, train_mode: bool=False):
        """
        Returns the output of the network with activation function applied
        """
        for layer in self.layers:
            inputs = layer.feed_forward(inputs, train_mode=train_mode)
        return inputs

    def calculate_loss(self, data):
        """
        Calculate the average cross-entropy loss on a given dataset.
        
        Args:
        - data: A tuple containing the features and labels in the form (data_x, data_y)
        
        Returns:
        - The average loss as a float.
        """
        data_x, data_y = data
        total_loss = 0
        n = len(data_x)

        # Iterate over all examples
        for i in range(n):
            # Feed forward to get the output probabilities
            output_probs = self.feed_forward(data_x.iloc[i].values)
            
            # Calculate cross-entropy loss for the current example
            # Note: Adding a small value epsilon to avoid log(0)
            epsilon = 1e-12
            output_probs = np.clip(output_probs, epsilon, 1. - epsilon)
            loss = -np.sum(data_y.iloc[i].values * np.log(output_probs))
            
            total_loss += loss

        # Return the average loss
        return total_loss / n


    def stochastic_gradient_descent(self, train_data: tuple,
                                    config: SimpleNamespace,
                                    test_data: tuple=None):
        """
        Train the network using stochastic gradient descent
        train_data is a tuple of (train_x, train_y)
        (optional) test_data is a tuple of (test_x, test_y)
        """
        logging.info("Starting training")
        train_x, train_y = train_data
        n = len(train_x) # number of training examples

        # if test data is not given, use train data as test data
        if test_data is None:
            test_data = train_data

        # start training
        mb = master_bar(range(config.epochs))
        mb.names = ['Training loss', 'Validation loss']

        # Initial loss values before training starts
        training_losses, validation_losses, accuracies = [], [], []
        best_validation_loss = float('inf')
        epochs_no_improve = 0
        learning_rate = config.learning_rate

        for epoch in mb:
            mb.main_bar.comment = 'total progress'


            # shuffle training data
            train_x, train_y = shuffle_data((train_x, train_y))

            # Initialize loss accumulation
            epoch_training_loss = 0

            # split training data into batches
            batches = []
            for i in range(0, n, config.batch_size):
                batches.append((train_x.iloc[i:i+config.batch_size],
                                train_y.iloc[i:i+config.batch_size]))

            # process each batch
            for batch in progress_bar(batches, parent=mb):
                mb.child.comment = 'epoch progress'
                # update weights and biases batch
                batch_loss = self.update_batch(batch, learning_rate, train_mode=True)
                epoch_training_loss += batch_loss * len(batch[0])
            
            # Calculate average training loss for the epoch
            epoch_training_loss /= n
            training_losses.append(epoch_training_loss)

            # Calculate validation loss
            epoch_validation_loss = self.calculate_loss(test_data)
            validation_losses.append(epoch_validation_loss)

            # Update graph with the new losses
            mb.update_graph([[range(1, epoch + 2),  training_losses],
                            [range(1, epoch + 2), validation_losses]], 
                            [1, config.epochs], [0, 10])

            # Update wandb
            wandb.log({"loss" : wandb.plot.line_series(
                xs=range(1, epoch + 2),
                ys=[training_losses, validation_losses],
                keys=["training", "validation"],
                title="Loss",
                xname="Epoch",
            )})

            # Evaluate accuracy if desired
            accuracy = self.evaluate(test_data)
            accuracies.append(accuracy)
            mb.write(f'Epoch {epoch + 1} - Training Loss: {epoch_training_loss:.4f}, Validation Loss: {epoch_validation_loss:.4f}, ' +
                     f'Accuracy: {accuracy:.2%}')
            
            # Update wandb
            wandb.log({"accuracy" : wandb.plot.line_series(
                xs=range(1, epoch + 2),
                ys=[accuracies],
                keys=["accuracy"],
                title="Accuracy",
                xname="Epoch",
            )})

            # Update learning rate with schedule
            if epoch % config.lr_decay_frequency == 0:
                learning_rate *= config.lr_decay_rate
                wandb.log({"epoch": epoch, "learning_rate": learning_rate})

            # Early stopping check
            if epoch_validation_loss < best_validation_loss:
                best_validation_loss = epoch_validation_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == config.early_stop_patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break  # Stop training if validation loss has not improved in 'early_stop_patience' epochs.

            # Log training and validation loss and accuracy
            wandb.log({
                "epoch": epoch,
                "training_loss": epoch_training_loss,
                "validation_loss": epoch_validation_loss,
                "accuracy": accuracy
            })

    def update_batch(self, batch: tuple, learning_rate: float, train_mode: bool=True):
        """
        Update weights and biases for a batch
        batch is a tuple of (batch_x, batch_y)
        """
        # keep in mind that the input layer has no weights and biases
        # thus the index of the layer is one less than the index of the batch
        delta_weights = []
        delta_bias = []

        # initialize delta_weights and delta_bias
        for layer in self.layers:
            if layer.get_type() == "input":
                continue
            delta_weights.append(np.zeros(layer.get_weights().shape))
            delta_bias.append(np.zeros(layer.get_bias().shape))

        # get the gradients for each training example in the batch
        batch_loss = 0
        for i in range(len(batch[0])):
            gradients, loss = self.backpropagation(batch[0].iloc[i].values,
                                                   batch[1].iloc[i].values,
                                                   train_mode=train_mode)
            batch_loss += loss

            # add gradients to delta_weights and delta_bias
            for j in range(len(gradients)):
                delta_weights[j] += gradients[j][0]
                delta_bias[j] += gradients[j][1]
                # average gradients
                delta_weights[j] /= len(batch[0])
                delta_bias[j] /= len(batch[0])

        # update weights and biases
        for i in range(len(self.layers)-1):
            # input layer is skipped again
            self.layers[i+1].update_weights_and_bias(
                self.layers[i+1].get_weights() - learning_rate * delta_weights[i],
                self.layers[i+1].get_bias() - learning_rate * delta_bias[i]
            )

        batch_loss /= len(batch[0])
        return batch_loss

    def backpropagation(self, inputs, desired_output, train_mode):
        """
        Determine how a single training example 
        would change the weights and biases and calculate the loss.
        
        Returns:
        - The gradients for each layer (except input layer).
        - The cross-entropy loss for the given input and desired output.
        """
        # gradients for each layer (except input layer)
        gradients = [None]*(len(self.layers)-1)
        activations = [None]*len(self.layers)
        z_values = [None]*(len(self.layers))

        # feed forward
        activations[0] = inputs
        z_values[0] = inputs
        for i in range(1, len(self.layers)):
            z_values[i] = self.layers[i].get_z(activations[i-1])
            activations[i] = self.layers[i].feed_forward(activations[i-1], train_mode)

        # calculate gradients for output layer
        delta_z = activations[-1] - desired_output
        loss = -np.sum(desired_output * np.log(activations[-1] + 1e-12))  # cross-entropy loss

        # calculate gradients for hidden layers
        for i in range(len(self.layers)-1, 0, -1):
            # calculate gradients for weights and biases
            delta_bias = delta_z
            delta_weights = np.outer(delta_z, activations[i-1])

            # calculate new delta_z
            if self.layers[i].get_activation_function() == "sigmoid":
                delta_z = np.matmul(self.layers[i].get_weights().T, delta_z) * \
                            sigmoid_derivative(z_values[i-1])
            elif self.layers[i].get_activation_function() == "relu":
                delta_z = np.matmul(self.layers[i].get_weights().T, delta_z) * \
                            relu_derivative(z_values[i-1])
            elif self.layers[i].get_activation_function() == "linear":
                delta_z = np.matmul(self.layers[i].get_weights().T, delta_z) * \
                            linear_derivative(z_values[i-1])
            elif self.layers[i].get_activation_function() == "softmax":
                delta_z = np.matmul(self.layers[i].get_weights().T, delta_z) * \
                            softmax_derivative(z_values[i-1])
            # save gradients
            gradients[i-1] = (delta_weights, delta_bias)
        # reverse gradients
        return gradients, loss
    
    def evaluate(self, test_data: tuple):
        """
        Returns the accuracy of the network
        test_data is a tuple of (test_x, test_y)
        """
        test_x, test_y = test_data
        correct = 0
        for i in range(len(test_x)):
            # feed forward with train_mode=False to disable dropout
            output = self.feed_forward(test_x.iloc[i].values, train_mode=False)
            # check if prediction is correct
            if np.argmax(output) == np.argmax(test_y.iloc[i].values):
                correct += 1
        return correct / len(test_x)
    
    def save(self, path):
        """
        save network to file
        """
        # save layers
        layers = []
        for layer in self.layers:
            layers.append({
                "num_neurons": layer.get_num_neurons(),
                "type": layer.get_type(),
                "activation_function": layer.get_activation_function(),
                "weights": layer.get_weights(),
                "bias": layer.get_bias()
            })
        # save network
        network = {
            "layers": layers
        }
        np.save(path, network)
        logging.info(f"Saved network to {path}")
        at = wandb.Artifact("network", type="network", metadata={"layers": layers})
        at.add_file(path)
        wandb.log_artifact(at)

    def load(self, path):
        """
        load network from file
        """
        # load network
        network = np.load(path, allow_pickle=True).item()
        # load layers
        self.layers = []
        for layer in network["layers"]:
            self.add_layer(NeuralLayer(layer["num_neurons"], layer["type"],
                                        None, layer["activation_function"]))
            self.layers[-1].update_weights_and_bias(layer["weights"], layer["bias"])
        logging.info(f"Loaded network from {path}")