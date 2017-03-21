"""
Train a neural network on breast cancer data
"""

from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt


class NeuralNetwork:
    """
    Neural network implementation using Numpy
    """

    def __init__(self):
        self.dataset = genfromtxt('res/breast-cancer-wisconsin.data', delimiter=',')

        # Remove rows containing nan values
        self.dataset = self.dataset[~np.isnan(self.dataset).any(axis=1)]

        # Update output values to 0 and 1 (because the sigmoid function outputs between 0 and 1)
        self.dataset[:, -1] = (self.dataset[:, -1] / 2) - 1

        self.input_data = np.delete(self.dataset, [0, 10], axis=1)
        self.output_data = np.asarray([[output] for output in self.dataset[:, -1]])

        # The error values
        self.errors = []

        # Global network properties
        self.layers = [9, 9, 1]

        self.learning_rate = 0.005

        self.neuron_input = []
        self.neuron_output = []

        self.neuron_weights = np.array(
            [np.random.rand(weights_count, self.layers[index]) - 0.5
             for index, weights_count in enumerate(self.layers[1:])])
        self.bias_weights = np.array(
            [np.random.rand(bias_count) - 0.5
             for bias_count in self.layers[1:]]
        )

    @staticmethod
    def sigmoid_activation(val: Decimal) -> float:
        """
        Sigmoid activation function. This function is used
        to squash values between 0 and 1.
        :param val: the variable used in the sigmoid
        :return: a value between 0 and 1
        """
        result = 1 / (1 + np.exp(-val))
        return result

    def sigmoid_derivative(self, val: Decimal) -> float:
        """
        Return the slope of the sigmoid function at position val
        :param val: position
        :return: slope
        """
        sigmoid = self.sigmoid_activation(val)
        return sigmoid * (1 - sigmoid)

    @staticmethod
    def cost_function(val: float, target: float) -> float:
        """
        Calculate the cost using the "sum of squares" function
        :param val: calculated output
        :param target: expected output
        :return: cost
        """
        return (1 / 2) * (abs(target - val)) ** 2

    def forward_pass(self, input_data: list) -> list:
        """
        The forward pass calculates the output of the neural network based on the given input values
        :type input_data: list
        :return: Output of the neural network
        """

        # Setup
        self.neuron_input = []
        self.neuron_output = []

        current_layer = 0
        current_input = input_data

        # Start with the input layer
        self.neuron_input.append(current_input)
        self.neuron_output.append(current_input)

        current_layer += 1

        # Continue with hidden layers
        hidden_layers = self.layers[1:]
        for _ in hidden_layers:
            layer_weights = self.neuron_weights[current_layer - 1]

            # multiply weights by input
            layer_weights = layer_weights.dot(current_input)

            new_input = []
            neuron_inputs = [weighted_input + bias for weighted_input, bias in
                             zip(layer_weights, self.bias_weights[current_layer - 1])]

            for activation_input in neuron_inputs:
                new_input.append(self.sigmoid_activation(activation_input))

            self.neuron_input.append(neuron_inputs)

            current_input = new_input
            self.neuron_output.append(new_input)
            current_layer += 1

        # End with output layer
        output = current_input

        return output

    def backward_pass(self, input_data: list, output_data: list) -> None:
        """
        Do the backward pass of the backpropagation algorithm.
        Calculate the total error and calculate the error gradients (= current derivative)
        for every output. After that, propagate the error gradient back over the entire network.
        Use the error gradients to update the weights by the learning rate.
        :param input_data: training input data for the forward pass
        :param output_data: expected output for the provided input data
        :return: None
        """
        input_data = input_data
        expected_output = np.asarray(output_data)
        actual_output = np.asarray(self.forward_pass(input_data))
        errors = []
        gradients = []

        # Start by calculating errors using backpropagation
        error_signal = actual_output - expected_output
        output_layer_input_derivative = np.asarray(
            [self.sigmoid_derivative(neuron_input) for neuron_input in self.neuron_input[-1]])

        errors.append(np.multiply(error_signal, output_layer_input_derivative))

        for index, weights in enumerate(self.neuron_weights[::-1]):
            # Calculate the hadamar product of the derivative of neuron inputs (layer k) and
            # the weights matrix (layer k) transposed multiplied by the error matrix (layer k + 1)
            errors.append(np.multiply(
                [self.sigmoid_derivative(neuron_input) for neuron_input in
                 self.neuron_input[- index - 2]],
                np.dot(weights.T, errors[-1])
            ))

        errors.reverse()  # backpropagation works from the end to the beginning of the network δ

        for index, (layer_error, neuron_output) in enumerate(
                zip(errors[1:], np.array(self.neuron_output))):
            neuron_output = np.array(neuron_output).reshape(-1, 1)
            gradients.append(np.array(np.outer(layer_error, neuron_output)))

        # Update weights according to gradients
        self.neuron_weights -= np.multiply(gradients, self.learning_rate)

        # Update biases according to the gradients
        self.bias_weights -= np.multiply(errors[1:], self.learning_rate)

    def train(self, epochs: int, plot: bool = False) -> None:
        """
        Train the neural network
        :param epochs: The number of times to loop through the training dataset
        :param plot: Show a graph of the error progression during training after training is done
        :return: None
        """
        data_size = self.input_data.shape[0]
        training_rows = int(data_size * .8)
        test_rows = data_size - training_rows

        # One epoch goes once through the entire training set
        for current_epoch in range(epochs):
            # Shuffle the training data for better training accuracy
            combined_training_data = np.column_stack((
                # Training input data
                self.input_data[:training_rows, ],
                # Training output data
                self.output_data[:training_rows, ]))
            np.random.shuffle(combined_training_data)

            output_columns_count = self.output_data.shape[1]
            for input_data, output_data in zip(combined_training_data[:, :-output_columns_count],
                                               combined_training_data[:, -output_columns_count]):
                self.backward_pass(input_data, output_data)

            epoch_errors = []
            # Run the training dataset through the updated network
            for test_input, expected_output in zip(self.input_data[-test_rows:, ],
                                                   self.output_data[-test_rows:, ]):
                calculated_output = self.forward_pass(test_input)

                epoch_errors.append(sum(self.cost_function(calculated_output, expected_output)))

            # Add the average error to self.errors
            self.errors.append(sum(epoch_errors) / float(len(epoch_errors)))

            # Lekker data's naar de console sturen voor de gebuiker
            print(
                "Epoch {:d}/{:d} - error: {:f}".format(current_epoch + 1, epochs, self.errors[-1]))

        if plot:
            plt.plot(self.errors)
            plt.show()


if __name__ == '__main__':
    NeuralNetwork().train(100, True)
