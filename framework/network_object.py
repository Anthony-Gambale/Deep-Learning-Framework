

# imports
import numpy as np
from matrices_tools import *


# sigmoid
def sigmoid(z):
    '''return sigmoid z'''
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    '''return dsigmoid/dz'''
    return sigmoid(z)*(1-sigmoid(z))


class Network():


    def __init__(self, layer_node_numbers=(2,1), eta=0.00001):
        '''
        eta: learning rate. multiply by negative gradient to get small step size.
        sizes: array containing number of neurons in each layer.
        '''
        self.eta = eta

        self.L = len(layer_node_numbers)

        # rows: number of output neurons
        # cols: number of input neurons
        # cols are 'size' before it, rows are 'size' after it, hense the [1:] and [:-1]
        # make both weights and biases have no 0th index. this is to make sure that the indices of weights and biases line up with L from the math.
        # nothing should have a 0th index, except for a.

        self.weights = [None] + [np.zeros((rows, columns))+1 for rows, columns in zip(layer_node_numbers[1:], layer_node_numbers[:-1])]

        self.biases = [None] + [np.zeros((rows, 1))+1 for rows in layer_node_numbers[1:]]


    def feedforward(self, x):
        '''
        x: a single input matrix
        '''
        z = [None] # store all z values. make z[0] = none, because nothing has a 0th index except for a
        a = [x] # store all a values. make a[0] = x as it should be

        for L in range(1, self.L):

            z.append( np.matmul(self.weights[L], a[L-1]) + self.biases[L] )
            a.append( sigmoid(z[L]) )

        return z, a


    def feedforward_multi_example(self, x_list):
        '''
        x_list: an array of x input matrices. feedforward every single input with only one matrix operation.
        Note that all letters used are uppercase, to denote the fact that they contain many copies of lowercase versions of themselves within them, as they were used
        the regular feedforward algorithm above.
        '''
        X = combine(x_list)

        Zs = [None] # no 0th element for Zs
        As = [X] # the 0th element of As is X

        for L in range(1, self.L):

            Zs.append( np.matmul(self.weights[L], As[L-1]) + self.biases[L] )
            As.append( sigmoid(Zs[L]) )
        
        # remember, Zs and As are not Z and A matrices themselves. they are a list containing all Z and A matrices for their respective layers, which themselves
        # contain all z and a column vectors for their respective training examples.
        return Zs, As


def main():
    '''run some code to test that it all works.'''
    perceptron = Network((2, 3, 1))

    x1 = [[1], [1]]
    x2 = [[4], [4]]

    examples = [x1, x2]

    _, As = perceptron.feedforward_multi_example(examples)


    print("First input:")
    [print(inv_combine(A)[0], "\n") for A in As]
    
    print("Second input:")
    [print(inv_combine(A)[1], "\n") for A in As]


if __name__ == "__main__":
    main()
