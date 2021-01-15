
'''
This file contains the class for the neural network, the main part of the framework. The main functions of the neural network are:
 * Feedforward (done)
 * Backpropagation
'''

# libraries
import numpy as np


def sigmoid(z):
    '''return sigmoid z'''
    return 1.0/(1.0+np.exp(-z))


class Network():


    def __init__(self, sizes=[2, 1], eta=0.0001):
        '''
        initialise a neural network with the given sizes, and learning rate.
        sizes: an array containing the number of neurons in each layer. the default network is a perceptron (2,1).
        eta: the learning rate. factor to make the step size smaller.
        '''
        self.eta = eta
        self.number_of_layers = len(sizes)
        # np.zeros([rows, columns])
        self.biases = [np.zeros([rows, 1]) for rows in sizes[1:]] # don't include the 0th size, as its the input nodes
        # each weight matrix has:
        # rows = nodes in next layer
        # columns = nodes in previous layer
        self.weights = [np.zeros([rows, columns]) for columns, rows in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, x):
        '''make a guess, where x and y are input and output matrices of dimensions (3, 1) and (2, 1)
        respectively.'''
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.matmul(w, x) + b)
        return x


if __name__ == "__main__":
    # if this file is run, then do some console output
    print(Network().feedforward([[1], [1]]))
