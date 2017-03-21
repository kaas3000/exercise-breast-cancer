"""
Train a neural network on breast cancer data
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from decimal import Decimal

DATASET_LOCATION = 'res/breast-cancer-wisconsin.data'

DATASET = genfromtxt(DATASET_LOCATION, delimiter=',')

# Remove rows containing nan values
DATASET = DATASET[~np.isnan(DATASET).any(axis=1)]

# Update output values to 0 and 1 (because the sigmoid function outputs between 0 and 1)
DATASET[:, -1] = (DATASET[:, -1] / 2) - 1

INPUT_DATA = np.delete(DATASET, [0, 10], axis=1)
OUTPUT_DATA = [[output] for output in DATASET[:, -1]]
OUTPUT_DATA = np.asarray(OUTPUT_DATA)

# The error values
ERRORS = []

# Global network properties
LAYERS = [9, 9, 1]

LEARNING_RATE = 0.005
NEURON_INPUT = []
NEURON_OUTPUT = []
NEURON_WEIGHTS = np.array(
    [np.random.rand(weights_count, LAYERS[index]) - 0.5
     for index, weights_count in enumerate(LAYERS[1:])])
BIAS_WEIGHTS = np.array(
    [np.random.rand(bias_count) - 0.5
     for bias_count in LAYERS[1:]]
)


def sigmoid_activation(val: Decimal) -> Decimal:
    """
    Sigmoid activation function. This function is used
    to squash values between 0 and 1.
    :param val: the variable used in the sigmoid
    :return: a value between 0 and 1
    """
    result = 1 / (1 + np.exp(-val))
    return result


def sigmoid_derivative(val: Decimal) -> Decimal:
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


def forward_pass(input_data: list) -> np.ndarray:
    """
    The forward pass calculates the output of the neural network based on the given input values
    :type input_data: list
    :return: Output of the neural network
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
    for _ in hidden_layers:
        layer_weights = NEURON_WEIGHTS[current_layer - 1]

        # multiply weights by input
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


def backward_pass(input_data: np.ndarray, output_data: np.ndarray) -> None:
    """
    Do the backward pass of the backpropagation algorithm.
    Calculate the total error and calculate the error gradients (= current derivative) for every output.
    After that, propagate the error gradient back over the entire network. Use the error gradients to update
    the weights by the learning rate.
    :param input_data: training input data for the forward pass
    :param output_data: expected output for the provided input data
    :return: None
    """
    global NEURON_WEIGHTS, NEURON_INPUT, NEURON_OUTPUT, ERRORS
    input_data = input_data
    expected_output = np.asarray(output_data)
    actual_output = np.asarray(forward_pass(input_data))
    errors = []
    gradients = []

    # Start by calculating errors using backpropagation
    error_signal = actual_output - expected_output
    output_layer_input_derivative = np.asarray([sigmoid_derivative(neuron_input) for neuron_input in NEURON_INPUT[-1]])

    errors.append(np.multiply(error_signal, output_layer_input_derivative))

    for index, weights in enumerate(NEURON_WEIGHTS[::-1]):
        # TODO merge lines
        w = weights.T
        b = np.dot(w, errors[-1])
        a = [sigmoid_derivative(neuron_input) for neuron_input in NEURON_INPUT[- index - 2]]

        b = np.array(b).reshape(-1, 1)  # convert to column vector (2d matrix)
        a = np.array(a).reshape(-1, 1)  # convert to column vector (2d matrix)

        c = a * b

        errors.append(c)

    errors.reverse()  # backpropagation works from the end to the beginning of the network δ

    for index, (layer_error, neuron_output) in enumerate(zip(errors[1:], np.array(NEURON_OUTPUT))):
        neuron_output = np.array(neuron_output).reshape(-1, 1)
        gradients.append(np.array(np.outer(layer_error, neuron_output)))

    # Update weights according to gradients
    d = np.array(gradients)
    e = d * LEARNING_RATE

    f = NEURON_WEIGHTS - e

    # NEURON_WEIGHTS = np.subtract(NEURON_WEIGHTS, np.asarray(gradients)[:-1] * LEARNING_RATE)
    NEURON_WEIGHTS = f

    # Update biases according to the gradients
    np.subtract(BIAS_WEIGHTS, np.array(errors)[1:] * LEARNING_RATE)


def train(epochs: int, plot: bool = False) -> None:
    """
    Train the neural network
    :param epochs: The number of times to loop through the training dataset
    :param plot: Show a graph of the error progression during training after training is done
    :return: None
    """
    data_size = INPUT_DATA.shape[0]
    training_rows = int(data_size * .8)
    test_rows = data_size - training_rows
    training_input = INPUT_DATA[:training_rows, ]
    training_output = OUTPUT_DATA[:training_rows, ]

    # One epoch goes once through the entire training set
    for current_epoch in range(epochs):
        combined_training_data = np.column_stack((training_input, training_output))
        np.random.shuffle(combined_training_data)

        training_columns_count = training_output.shape[1]

        current_training_input = combined_training_data[:, :-training_columns_count]
        current_training_output = combined_training_data[:, -training_columns_count]

        for input_data, output_data in zip(current_training_input, current_training_output):
            backward_pass(input_data, output_data)

        test_input = INPUT_DATA[-test_rows:, ]
        test_output = OUTPUT_DATA[-test_rows:, ]

        new_errors = []
        for x, y in zip(test_input, test_output):
            calc_out = forward_pass(x)

            new_errors.append(sum(cost_function(calc_out, y)))

        ERRORS.append(sum(new_errors) / float(len(new_errors)))
        print("Epoch {:d}/{:d} - error: {:f}".format(current_epoch + 1, epochs, ERRORS[-1]))

    if plot:
        plt.plot(ERRORS)
        plt.show()


if __name__ == '__main__':
    train(100, True)
