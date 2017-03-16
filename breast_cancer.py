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

LEARNING_RATE = 0.5
NEURON_INPUT = []
NEURON_OUTPUT = []
# ALL_WEIGHTS = np.array([
#     np.random.rand(10, 9) / 10,
#     np.random.rand(10, 9) / 10
# ])
NEURON_WEIGHTS = np.array(
    [np.random.rand(weights_count, LAYERS[index]) - 0.5
     for index, weights_count in enumerate(LAYERS[1:])])
BIAS_WEIGHTS = np.array(
    [np.random.rand(bias_count) - 0.5
     for bias_count in LAYERS[1:]]
)


# ALL_WEIGHTS[-1] = ALL_WEIGHTS[-1][:, :-BIAS]


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
    global NEURON_OUTPUT, NEURON_INPUT
    NEURON_INPUT = []
    NEURON_OUTPUT = []

    current_layer = 0
    current_input = input_data

    # Start with the input layer
    # Add a bias node if necessary
    # if BIAS > 0:
    #     current_input = np.concatenate((current_input, np.ones((BIAS,))), axis=0)
    # NEURON_OUTPUT.append(np.array(current_input).tolist())
    NEURON_INPUT.append(current_input)
    NEURON_OUTPUT.append(current_input)

    current_layer += 1

    # Continue with hidden layers
    hidden_layers = LAYERS[1:]
    for layer_size in hidden_layers:
        layer_weights = NEURON_WEIGHTS[current_layer - 1]

        # multiply weights by input
        layer_weights = np.multiply(
            np.tile(np.matrix(current_input), [layer_size, 1]),
            layer_weights)

        new_input = []
        neuron_inputs = [float(np.sum(weighted_input) + bias) for weighted_input, bias in
                         zip(layer_weights, BIAS_WEIGHTS[current_layer - 1])]

        for activation_input in neuron_inputs:
            new_input.append(sigmoid_activation(activation_input))

        NEURON_INPUT.append(neuron_inputs)

        current_input = new_input
        NEURON_OUTPUT.append(new_input)
        current_layer += 1

    # End with output layer
    # neuron_count = LAYERS[-1]
    # layer_weights = np.tile(current_input, [neuron_count, 1]) * ALL_WEIGHTS[current_layer]
    output = current_input

    # NEURON_OUTPUT.append(output)

    return output


def backward_pass(input_data, output_data):
    global NEURON_WEIGHTS
    input_data = input_data
    expected_output = output_data
    actual_output = forward_pass(input_data)[0]
    gradients = []

    # Start by calculating gradients using backpropagation
    error_signal = cost_function(actual_output, expected_output)
    if not isinstance(error_signal, list):
        error_signal = [error_signal]

    gradients.append(np.dot(error_signal, [sigmoid_derivative(NEURON_INPUT[-1][0])]))

    for index, weights in enumerate(NEURON_WEIGHTS[::-1]):
        a = np.dot(weights.T, gradients[-1])
        b = [sigmoid_derivative(neuron_input) for neuron_input in NEURON_INPUT[- index - 1]]
        c = np.multiply(a, b)

        gradients.append(c)

    gradients.reverse()  # backpropagation works from the end to the beginning of the network

    # Update weights according to gradients
    NEURON_WEIGHTS = np.subtract(NEURON_WEIGHTS, np.array(gradients)[:-1] * LEARNING_RATE)


def train(epochs):
    training_rows = 500
    training_input = INPUT_DATA[:training_rows, ]
    training_output = OUTPUT_DATA[:training_rows, ]

    for current_epoch in range(epochs):
        for input_data, output_data in zip(training_input, training_output):
            backward_pass(input_data, output_data)

        test_input = INPUT_DATA[:-199, ]
        test_output = OUTPUT_DATA[:-199, ]
        # new_error = [cost_function(forward_pass(x)[0], y)
        #               for x, y in zip(test_input, test_output)]
        new_errors = []
        for x, y in zip(test_input, test_output):
            calc_out = forward_pass(x)[0]
            new_errors.append(cost_function(calc_out, y))

        print("%d - New error: %d" % (current_epoch, sum(new_errors)))

train(200)
