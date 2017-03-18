"""
Train a neural network on breast cancer data
"""

import numpy as np
from numpy import genfromtxt
from decimal import Decimal

DATASET_LOCATION = 'res/breast-cancer-wisconsin.data'

DATASET = genfromtxt(DATASET_LOCATION, delimiter=',')

# Remove rows containing nan values
DATASET = DATASET[~np.isnan(DATASET).any(axis=1)]

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


def sigmoid_activation(val: Decimal):
    """
    Sigmoid activation function. This function is used
    to squash values between 0 and 1.
    :param val: the variable used in the sigmoid
    :return: a value between 0 and 1
    """
    result = 1 / (1 + np.exp(-val))
    return result


def sigmoid_derivative(val: Decimal):
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
    NEURON_INPUT.append(current_input)
    NEURON_OUTPUT.append(current_input)

    current_layer += 1

    # Continue with hidden layers
    hidden_layers = LAYERS[1:]
    for layer_size in hidden_layers:
        layer_weights = NEURON_WEIGHTS[current_layer - 1]

        # multiply weights by input
        # layer_weights = np.multiply(
        #     np.tile(np.matrix(current_input), [layer_size, 1]),
        #     layer_weights)
        layer_weights = layer_weights.dot(current_input)

        new_input = []
        neuron_inputs = [weighted_input + bias for weighted_input, bias in
                         zip(layer_weights, BIAS_WEIGHTS[current_layer - 1])]

        for activation_input in neuron_inputs:
            new_input.append(sigmoid_activation(activation_input))

        NEURON_INPUT.append(neuron_inputs)

        current_input = new_input
        NEURON_OUTPUT.append(new_input)
        current_layer += 1

    # End with output layer
    output = current_input

    return output


def backward_pass(input_data, output_data):
    global NEURON_WEIGHTS, NEURON_INPUT, NEURON_OUTPUT
    input_data = input_data
    expected_output = output_data
    actual_output = forward_pass(input_data)[0]
    errors = []
    gradients = []

    # Start by calculating errors using backpropagation
    error_signal = cost_function(actual_output, expected_output)
    if not isinstance(error_signal, list):
        error_signal = [error_signal]

    errors.append(np.multiply(error_signal, [sigmoid_derivative(NEURON_INPUT[-1][0])]))

    for index, weights in enumerate(NEURON_WEIGHTS[::-1]):
        # TODO merge lines
        w = weights.T
        b = np.dot(w, errors[-1])
        a = [sigmoid_derivative(neuron_input) for neuron_input in NEURON_INPUT[- index - 1]]

        b = np.asarray(b)
        a = np.array(a).reshape(-1, 1)

        c = a * b

        errors.append(c)

    errors.reverse()  # backpropagation works from the end to the beginning of the network δ

    for index, (layer_error, neuron_output) in enumerate(zip(errors[:-1], np.array(NEURON_OUTPUT)[1:])):
        neuron_output = np.array(neuron_output).reshape(-1, 1)
        gradients.append(
            # np.split(
                np.array(np.outer(layer_error, neuron_output))
            #     NEURON_WEIGHTS[index].shape[0]
            # )
        )

    # Update weights according to gradients
    newgradients = tuple(gradients)
    d = np.array(newgradients)
    e = d * LEARNING_RATE

    f = NEURON_WEIGHTS - e

    # NEURON_WEIGHTS = np.subtract(NEURON_WEIGHTS, np.asarray(gradients)[:-1] * LEARNING_RATE)
    NEURON_WEIGHTS = f

    # Update biases according to the gradients
    np.subtract(BIAS_WEIGHTS, np.array(errors)[1:] * LEARNING_RATE)


def train(epochs):
    data_size = INPUT_DATA.shape[0]
    training_rows = int(data_size * .8)
    test_rows = data_size - training_rows
    training_input = INPUT_DATA[:training_rows, ]
    training_output = OUTPUT_DATA[:training_rows, ]

    for current_epoch in range(epochs):
        a = np.column_stack((training_input, training_output))
        np.random.shuffle(a)

        training_input = a[:, :-1]
        training_output = a[:, -1]

        for input_data, output_data in zip(training_input, training_output):
            backward_pass(input_data, output_data)

        test_input = INPUT_DATA[:-test_rows, ]
        test_output = OUTPUT_DATA[:-test_rows, ]
        # new_error = [cost_function(forward_pass(x)[0], y)
        #               for x, y in zip(test_input, test_output)]
        new_errors = []
        for x, y in zip(test_input, test_output):
            calc_out = forward_pass(x)[0]
            print(calc_out)
            new_errors.append(cost_function(calc_out, y))

            # print("%d - New error: %d" % (current_epoch, sum(new_errors)))


if __name__ == '__main__':
    train(2)
