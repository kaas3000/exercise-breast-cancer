"""
Train a neural network on breast cancer data
"""
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from decimal import Decimal


def scale(x):
    """
    Scale the input values between 0 and 1.
    From: http://stackoverflow.com/a/1735122
    """
    x /= np.max(np.abs(x), axis=0)
    return x


DATASET_LOCATION = 'res/motorcycle-bpm.csv'

parse_date = lambda x: datetime.strptime(x.decode("utf-8"), '%d/%m/%Y').timestamp()
# col_headers = ["Kenteken", "Merk", "Datum eerste toelating", "Datum tenaamstelling", "Catalogusprijs",
#                "Massa ledig voertuig", "Wielbasis",
#                "Aantal cilinders", "Cilinderinhoud", "WAM verzekerd", "Bruto BPM"]
DATASET = genfromtxt(DATASET_LOCATION, delimiter=',', skip_header=1, usecols=(2, 3, 4, 5, 6, 7, 8, 10), converters={
    2: parse_date,
    3: parse_date
})
# DATASET = np.delete(DATASET, [0, 1], axis=1)

# Remove rows containing nan values
DATASET = DATASET[~np.isnan(DATASET).any(axis=1)]

# Update output values to 0 and 1 (because the sigmoid function outputs between 0 and 1)
# Note the max value of bruto BPMN in the training set is 6200
DATASET[:, -1] = (DATASET[:, -1] / 10000)

# Scale input values between 0 and 1
DATASET[:, :7] = scale(DATASET[:, :7])
test = DATASET[:, :7]

INPUT_DATA = DATASET[:, :7]
OUTPUT_DATA = [[output] for output in DATASET[:, -1]]
OUTPUT_DATA = np.asarray(OUTPUT_DATA)

# The error values
ERRORS = []

# Global network properties
LAYERS = [7, 1]
print(LAYERS)

LEARNING_RATE = 0.2
NEURON_INPUT = []
NEURON_OUTPUT = []
NEURON_WEIGHTS = np.array(
    [np.random.rand(weights_count, LAYERS[index]) - 0.5
     for index, weights_count in enumerate(LAYERS[1:])])
BIAS_WEIGHTS = np.array(
    [np.random.rand(bias_count) - 0.5
     for bias_count in LAYERS[1:]]
)


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


def backward_pass(input_data, output_data):
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


def train(epochs, plot=False):
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

        test_input = INPUT_DATA[:-test_rows, ]
        test_output = OUTPUT_DATA[:-test_rows, ]

        new_errors = []
        for x, y in zip(test_input, test_output):
            calc_out = forward_pass(x)

            new_errors.append(np.sqrt(sum(cost_function(calc_out, y))))

        ERRORS.append(sum(new_errors) / float(len(new_errors)))
        print("Epoch {:d}/{:d} - error: {:f}".format(current_epoch + 1, epochs, ERRORS[-1]))

    if plot:
        plt.plot(ERRORS)
        plt.show()


if __name__ == '__main__':
    train(100, True)

    test_data_location = 'res/test.csv'

    parse_date = lambda x: datetime.strptime(x.decode("utf-8"), '%d/%m/%Y').timestamp()
    col_headers = ["Kenteken", "Merk", "Datum eerste toelating", "Datum tenaamstelling", "Catalogusprijs",
                   "Massa ledig voertuig", "Wielbasis",
                   "Aantal cilinders", "Cilinderinhoud", "WAM verzekerd", "Bruto BPM"]
    test_dataset = genfromtxt(test_data_location, delimiter=',', skip_header=1, usecols=(2, 3, 4, 5, 6, 7, 8),
                              converters={
                                  2: lambda x: datetime.strptime(x.decode("utf-8"), '%d/%m/%Y').timestamp(),
                                  3: lambda x: datetime.strptime(x.decode("utf-8"), '%d/%m/%Y').timestamp()
                              })

    autogegevens = genfromtxt(test_data_location, delimiter=',', skip_header=1, usecols=(0, 1), dtype=object)

    # Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = np.nanmean(test_dataset, axis=0)

    # Find indicies that you need to replace
    inds = np.where(np.isnan(test_dataset))

    # Place column means in the indices. Align the arrays using take
    test_dataset[inds] = np.take(col_mean, inds[1])

    print(test_dataset.shape)

    should_keep_row = ~np.isnan(test_dataset).any(axis=1)
    test_dataset = test_dataset[should_keep_row]
    autogegevens = autogegevens[should_keep_row]

    print(test_dataset.shape)

    # Update output values to 0 and 1 (because the sigmoid function outputs between 0 and 1)
    # Note the max value of bruto BPMN in the training set is 6200
    # test_dataset[:, -1] = (test_dataset[:, -1] / 10000)

    # Scale input values between 0 and 1
    test_dataset = scale(test_dataset)

    input_data = test_dataset[:, :7]

    with open("results.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Kenteken', 'Prediction'])

        for index, row in enumerate(input_data):
            network_result = forward_pass(row)
            prediction = (network_result[0] * 10000)
            prediction = str(prediction)
            kenteken = autogegevens[index][0].decode("utf-8")
            writer.writerow([
                kenteken,
                prediction
            ])
            print(kenteken, prediction)

