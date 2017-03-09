"""
Train a neural network on breast cancer data
"""

import numpy as np
from numpy import genfromtxt

DATASET_LOCATION = 'res/breast-cancer-wisconsin.data'

DATASET = genfromtxt(DATASET_LOCATION, delimiter=',')
INPUT_DATA = np.delete(DATASET, [0, 10], axis=1)
OUTPUT_DATA = DATASET[:, -1]

# Global network properties
LAYERS = [1, 9, 1]
ALL_WEIGHTS = np.array([
    np.identity(9),
    np.random.rand(9, 9),
    np.random.rand(9, 9)
])


def sigmoid_activation(val):
    """
    Sigmoid activation function. This function is used
    to squash values between 0 and 1.
    :param val: the variable used in the sigmoid
    :return: a value between 0 and 1
    """
    return 1 / (1 + np.exp(-val))


def forwardpass():
    """
    Forward pass
    """

    # Setup
    current_layer = 0
    current_input = INPUT_DATA[0]

    # Forward pass: loop through network
    for layer_size in LAYERS:
        layer_weights = ALL_WEIGHTS[current_layer]

        # multiply weights by input
        layer_weights = np.tile(current_input, [layer_size, 1]) * layer_weights

        new_input = []

        for activation_input in [np.sum(neuron_input) for neuron_input in layer_weights]:
            new_input.append(sigmoid_activation(activation_input))

        current_input = new_input
        current_layer += 1

    print(current_input)


forwardpass()
