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
    [np.random.rand(weights_count, LAYERS[index] + BIAS)
     for index, weights_count in enumerate(LAYERS[1:])])
ALL_WEIGHTS[-1] = ALL_WEIGHTS[-1][:, :-BIAS]


def sigmoid_activation(val: float):
    """
    Sigmoid activation function. This function is used
    to squash values between 0 and 1.
    :param val: the variable used in the sigmoid
    :return: a value between 0 and 1
    """
    return 1 / (1 + np.exp(-val))


def sigmoid_derivative(val: float):
    """
    Return the slope of the sigmoid function at position val
    :param val: position
    :return: slope
    """
    sigmoid = sigmoid_activation(val)
    return sigmoid * (1 - sigmoid)


def cost_function(val: float, target: float) -> float:
    """
    Calculate the cost using the "sum of squares" function
    :param val: calculated output
    :param target: expected output
    :return: cost
    """
    return (1 / 2) * (abs(target - val)) ** 2


def forward_pass(input_data: list):
    """
    The forward pass calculates the output of the neural network based on the given input values
    :type input_data: list
    """

    # Setup
    global NEURON_VALUES
    NEURON_VALUES = []

    current_layer = 0
    current_input = input_data

    # Start with the input layer
    NEURON_VALUES.append(np.array(input_data).tolist())

    # Continue with hidden layers
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
        NEURON_VALUES.append(new_input)
        current_layer += 1

    # End with output layer
    neuron_count = LAYERS[-1]
    layer_weights = np.tile(current_input, [neuron_count, 1]) * ALL_WEIGHTS[current_layer]
    output = []

    for activation_input in [np.sum(neuron_input) for neuron_input in layer_weights]:
        output.append(sigmoid_activation(activation_input))

    NEURON_VALUES.append(output)

    return output


def backward_pass():
    input_data = INPUT_DATA[0]
    expected_output = OUTPUT_DATA[0]
    actual_output = forward_pass(input_data)[0]

    error_signal = cost_function(actual_output, expected_output)
    if not isinstance(error_signal, list):
        error_signal = [error_signal]

    all_weights_index = ALL_WEIGHTS.size
    gradients = [sigmoid_derivative(error) for error in error_signal]
    for neuron_layer_output in NEURON_VALUES[:-1:-1]:
        all_weights_index -= 1

        gradients = np.multiply(np.array([sigmoid_derivative(output) for output in neuron_layer_output]),
                                (ALL_WEIGHTS[all_weights_index].transpose().dot(gradients)))


print(forward_pass(INPUT_DATA[0]))
backward_pass()
