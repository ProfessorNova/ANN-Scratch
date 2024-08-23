# ANN-Scratch

This repository implements a simple Artificial Neural Network (ANN) from scratch
using only numpy. It was tested on the MNIST dataset and achieved an accuracy of
around 95% after 50 epochs (The hyperparameters were not tuned so there is room
for improvement).

---

## Getting Started

### Installation

You basically just have to have numpy installed (as well as matplotlib if you want
to plot the data). You can install them using pip:

```bash
pip install numpy matplotlib
```

Then clone the repository:

```bash
git clone https://github.com/ProfessorNova/ANN-Scratch.git
cd ANN-Scratch
```

The code was tested on Python 3.10.11.

### Usage

The functionality is shown with visualisation in ![train.ipynb](https://github.com/ProfessorNova/ANN-Scratch/blob/main/train.ipynb). You can also run the code in ![train.py](https://github.com/ProfessorNova/ANN-Scratch/blob/main/train.py)
with the following command (there you will only see the loss and accuracy printed in the console):

```bash
python train.py
```

---

## Components

The repository is divided into the following components:

- ![lib/activations_functions.py](https://github.com/ProfessorNova/ANN-Scratch/blob/main/lib/activation_functions.py): Contains the activation functions and their derivatives. The following activation
  functions are implemented:
    - Sigmoid
    - Linear
    - Softmax
    - ReLU

- ![lib/neural_layer.py](https://github.com/ProfessorNova/ANN-Scratch/blob/main/lib/neural_layer.py): Contains the NeuralLayer class which represents a layer in the neural network. It contains
  the forward and backward methods as well as a method to update the weights and biases.

- ![lib/neural_network.py](https://github.com/ProfessorNova/ANN-Scratch/blob/main/lib/neural_network.py): Contains the NeuralNetwork class which represents the neural network. It implements the
  backpropagation algorithm and stochastic gradient descent. It also has methods to save and load the model.

- ![lib/data_loader.py](https://github.com/ProfessorNova/ANN-Scratch/blob/main/lib/data_loader.py): Contains a function to load the given `mnist_test.csv` and `mnist_train.csv` files.
  Furthermore, it automatically preprocesses the data by normalizing it and converting the labels to one-hot encoding.
