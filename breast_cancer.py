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
LAYERS = [9, 9, 1]
BIAS = 1
# ALL_WEIGHTS = np.array([
#     np.random.rand(10, 9) / 10,
#     np.random.rand(10, 9) / 10
# ])
ALL_WEIGHTS = np.array(
    [np.random.rand(weights_count, LAYERS[index] + BIAS) for index, weights_count in enumerate(LAYERS[1:])])
ALL_WEIGHTS[-1] = ALL_WEIGHTS[-1][:, :-BIAS]


def sigmoid_activation(val):
    """
    Sigmoid activation function. This function is used
    to squash values between 0 and 1.
    :param val: the variable used in the sigmoid
    :return: a value between 0 and 1
    """
    return 1 / (1 + np.exp(-val))


def forwardpass(input_data: list):
    """
    The forward pass calculates the output of the neural network based on the given input values
    :type input_data: list
    """

    # Setup
    current_layer = 0
    current_input = input_data

    # Start with hidden layers
    hidden_layers = LAYERS[1:-1]
    for layer_size in hidden_layers:
        layer_weights = ALL_WEIGHTS[current_layer]

        # Add a bias node if necessary
        if BIAS > 0:
            current_input = np.concatenate((current_input, np.ones((BIAS,))), axis=0)

        # multiply weights by input
        layer_weights = np.tile(current_input, [layer_size, 1]) * layer_weights

        new_input = []

        for activation_input in [np.sum(neuron_input) for neuron_input in layer_weights]:
            new_input.append(sigmoid_activation(activation_input))

        current_input = new_input
        current_layer += 1

    # End with output layer
    neuron_count = LAYERS[-1]
    layer_weights = np.tile(current_input, [neuron_count, 1]) * ALL_WEIGHTS[current_layer]
    output = []

    for activation_input in [np.sum(neuron_input) for neuron_input in layer_weights]:
        output.append(sigmoid_activation(activation_input))

    return output


print(forwardpass(INPUT_DATA[0]))
