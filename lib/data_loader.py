import numpy as np


def load_data(path, log=True):
    if log:
        print(f"Loading data from {path}")

    # Load the data from a CSV file
    data = np.genfromtxt(path, delimiter=',', skip_header=1)

    # Split the data into inputs and labels
    data_x = data[:, 1:]
    data_y = data[:, 0].astype(int)

    # Normalize the inputs
    data_x = data_x / 255.0

    # Convert the labels to one-hot encoding
    data_y_one_hot = np.zeros((len(data_y), 10))
    data_y_one_hot[np.arange(len(data_y)), data_y] = 1

    if log:
        print(f"Loaded {len(data_x)} samples")
        
    return data_x, data_y_one_hot
