from lib.data_loader import load_data
from lib.neural_layer import NeuralLayer
from lib.neural_network import NeuralNetwork

if __name__ == "__main__":
    # Load the data
    train_data = load_data("data/mnist_train.csv")
    test_data = load_data("data/mnist_test.csv")

    # Create a neural network
    nn = NeuralNetwork()
    nn.add_layer(NeuralLayer(784, 128, "relu"))
    nn.add_layer(NeuralLayer(128, 64, "relu"))
    nn.add_layer(NeuralLayer(64, 10, "softmax"))
    print(nn)

    # Train the neural network
    nn.stochastic_gradient_descent(train_data,
                                   epochs=50,
                                   batch_size=32,
                                   learning_rate=0.01,
                                   test_data=test_data)

    # Save the model
    nn.save("model.npy")
