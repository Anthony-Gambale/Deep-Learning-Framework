
'''
This file contains the class for the neural network, the main part of the framework. The main functions of the neural network are:
 * Feedforward (done)
 * Backpropagation
'''

# libraries
import numpy as np
from random import choice


def sigmoid(z):
    '''return sigmoid z'''
    return 1.0/(1.0+np.exp(-z))


class Network():


    def __init__(self, sizes=[2, 1], eta=0.0001):
        ''' initialise a neural network.
        sizes: an array containing the number of neurons in each layer. the default network is a perceptron [2,1].
        eta: the learning rate. factor to make the step size smaller. '''
        self.eta = eta
        self.number_of_layers = len(sizes)
        # np.zeros([rows, columns])
        self.biases = [np.zeros([rows, 1]) for rows in sizes[1:]] # don't include the 0th size, as its the input nodes
        # rows is no. of nodes in layer after the weights, columns is no. of nodes layer before the weights
        self.weights = [np.zeros([rows, columns]) for rows, columns in zip(sizes[1:], sizes[:-1])]


    # def feedforward(self, x):
    #     '''make a guess based on current parameters, and given input
    #     x: input matrix'''
    #     activation = x
    #     for b, w in zip(self.biases, self.weights):
    #         activation = sigmoid(np.matmul(w, x) + b)
    #     return activation

    def feedforward(self, x):
        '''make a guess based on current parameters, and given input
        output an array of activations and z values calculated'''
        


    def gradients(self, x, y):
        '''based on the given input and output, calculate the error and output the gradients of all the parameters.
        x: input matrix
        y: output matrix'''
        # init matrices for the gradients
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        


    def update_mini_batch(self, mini_batch, eta):
        '''update parameters (weights and biases)
        mini_batch: a list of [x,y] points to train on.
        eta: the learning rate, or multiplier, for the descent step.'''
        # placeholder matrices for gradients
        total_nabla_b = [np.zeros(b.shape) for b in self.biases]
        total_nabla_w = [np.zeros(w.shape) for w in self.weights]

        # calculate gradient
        for x, y in mini_batch:

            nabla_w, nabla_b = self.backprop_step(x, y)
            total_nabla_w = [nw + dnw for nw, dnw in zip(total_nabla_w, nabla_w)]
            total_nabla_b = [nb + dnb for nb, dnb in zip(total_nabla_b, nabla_b)]

        # now that the gradients have been calculated, convert them into steps and apply
        self.weights = [w - dw * eta for w, dw in zip(self.weights, total_nabla_w)]
        self.biases = [b - db * eta for b, db in zip(self.biases, total_nabla_b)]


    def stochastic_gradient_descent(self, data, iterations, sgd_percentage, eta):
        ''' run stochastic gradient descent on self.
        data: the training data that will be used. of form: [[x,y], [x,y], [x,y], ... ] where x and y are matrices.
        iterations: the number of times to iterate over the training data
        sgd_percentage: the percentage of datapoints from data to use in one iteration
        eta: the learning rate, or multiplier, for the descent step. '''
        # length of data etc.
        n = len(data)
        sgd_n = int(n * sgd_percentage)

        # iterate
        for i in range(iterations):
            
            # generate mini batch, and do gradient descent
            mini_batch = [choice(data) for i in range(sgd_n)]
            self.update_mini_batch(mini_batch, eta)





if __name__ == "__main__":
    # if this file is run, then do some console output
    print(Network().feedforward([[1], [1]]))
