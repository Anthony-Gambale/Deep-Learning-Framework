

# imports
import numpy as np
import matrices_tools as mt


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
        zs = [None] # store all z values. make z[0] = none, because nothing has a 0th index except for a
        activations = [x] # store all a values. make a[0] = x as it should be

        for L in range(1, self.L):

            zs.append( np.matmul(self.weights[L], activations[L-1]) + self.biases[L] )
            activations.append( sigmoid(zs[L]) )

        return zs, activations


    def feedforward_multi_example(self, x_list):
        '''
        x_list: an array of x input matrices. feedforward every single input with only one matrix operation.
        Note that all letters used are uppercase, to denote the fact that they contain many copies of lowercase versions of themselves within them, as they were used
        the regular feedforward algorithm above.
        '''
        X = mt.combine(x_list)

        Zs = [None] # no 0th element for Zs
        As = [X] # the 0th element of As is X

        for L in range(1, self.L):

            Zs.append( np.matmul(self.weights[L], As[L-1]) + self.biases[L] )
            As.append( sigmoid(Zs[L]) )
        
        # remember, Zs and As are not Z and A matrices themselves. they are a list containing all Z and A matrices for their respective layers, which themselves
        # contain all z and a column vectors for their respective training examples.
        return Zs, As


    # def dC_dz_final_layer(self, x, y):
    #     '''
    #     find the derivative of the Cost with respect to the z values of the output layer, **with respect to a single input/output pair** (x,y).
        
    #     Cx = 0.5 * (a - y) ^ 2
    #     a = s(z)
    #     dCx/da = 0.5 * 2 * (a - y) = (a - y)
    #     da/dz = s'(z)
    #     dCx/dz = dCx/da * da/dz = (a - y) * s'(z)

    #     the output is a column vector, with the error for each node in the output layer stacked in a column.
    #     '''
    #     zs, activations = self.feedforward(x)

    #     return (activations[-1] - y) * sigmoid_prime(zs[-1]) # hadamard product
    

    # def dC_dz_final_layer_multi_example(self, x_list, y_list):
    #     '''
    #     do the same thing as the previous function, but with a big X matrix, where each column is an x vector, and a big Y matrix, where each column is a y vector.
    #     also, A matrices are a big matrix where each column is an a vector, etc. the formula holds for this case.

    #     the output is a matrix, where each column is a column vector of the error for each node in the output layer for a certain x/y pair. each column is a different x/y pair, and each row is a different
    #     node in the output layer.
    #     '''
    #     Y = mt.combine(y_list)
    #     Zs, As = self.feedforward_multi_example(x_list)

    #     return (As[-1] - Y) * sigmoid_prime(Zs[-1])


    def dC_dz_all_layers(self, x, y):
        '''
        find the dC/dz functions for all layers, with respect to the given input and output, x and y.

        BP2:
        dC/dz(L)  =  (w(L+1)(T) . dC/dz(L+1)) . da/dz(L)
        dC/dz(L)  =  (w(L+1)(T) . dC/dz(L+1)) . sigmoid_prime(z(L))

        Note that the weight matrix needs to be transposed, because the errors in the layer on the right are effectively being feedforwarded, but backwards. the number of rows and columns needs
        to be swapped so that the input size is what would normally be the output in ff, and vice versa; the output size is what would normally be the input in ff.
        '''
        deltas = [None for i in range(self.L)] # the delta character is used for the error matrices, dC/dz.

        zs, activations = self.feedforward(x)

        deltas[-1] = (activations[-1] - y) * sigmoid_prime(zs[-1])

        for L in range(-2, -self.L, -1):

            deltas[L] = np.matmul( np.transpose(self.weights[L+1]), deltas[L+1] ) * sigmoid_prime(zs[L])
        
        return deltas


def main():
    '''run some code to test that it all works.'''
    #  -- init --
    perceptron = Network((2, 3, 1))

    x1 = [[1], [1]]
    x2 = [[4], [4]]

    y1 = [[1], [1]]
    y2 = [[1], [1]]

    x_list = [x1, x2]
    y_list = [y1, y2]

    # -- error --
    error1 = perceptron.dC_dz_final_layer(x1, y1)
    error2 = perceptron.dC_dz_final_layer(x2, y2)
    error_matrix = perceptron.dC_dz_final_layer_multi_example(x_list, y_list)

    print(error1, 2*"\n", error2, 2*"\n", error_matrix)

    # -- feedforward --
    # _, As = perceptron.feedforward_multi_example(examples)

    # print("First input:")
    # [print(mt.inv_combine(A)[0], "\n") for A in As]

    # print("Second input:")
    # [print(mt.inv_combine(A)[1], "\n") for A in As]


if __name__ == "__main__":
    main()
