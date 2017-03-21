# Basic implementation of a neural network in python

## Installation
When using virtualenv, create a new environment and activate it with
the following commands:
```shell
virtualenv .
. bin/activate
```

To install the dependencies using pip run:
```shell
pip install -r requirements.txt
```

## Usage
The usage is currently quite limited. The provided dataset
`res/breast-cancer-wisconsin.data` is hardcoded. A different dataset
can't be used without modifying the code.
The application can be started using the command:
```shell
python3 breast_cancer.py
```


## About the application
The application will initialize a multilayered perceptron network.
This network contains an input layer of 9 neurons, followed by a
hidden layer of another 9 neurons and an output layer of 1 neuron.
The network will be trained using the backpropagation algorithm (SGD)
wich is implemented completely using only raw python and numpy arrays
and numpy array operations.


## Thanks
This script is the result of an attempt to learn how neural networks work
and can be trained.
The sources used to obtain this information are:
* The CS224d course, specifically lecture 6.
([lecture](https://youtu.be/MeIrQCZvlkE?list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG) ,
[notes](http://cs224d.stanford.edu/lecture_notes/notes3.pdf))
* A tutorial by Matt Mazur which explains the required calculations in
a different way ([link](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/))
* Another Python/Numpy implementation by [hdmetor](https://github.com/hdmetor/NeuralNetwork).