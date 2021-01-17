

# imports
import numpy as np


def combine(matrices):
    '''
    examples: a list of x input matrices, or y output matrices. combine them into one matrix, where each column is one of the given matrices.
    X = combine([x1, x2, x3, x4 ... xm])
    Y = combine([y1, y2, y3, y4 ... ym])
    '''
    return np.concatenate(matrices, axis=1)


def inv_combine(matrix):
    '''
    take a matrix and split it into column vectors. essentially, do the opposite of combine(), defined above.
    '''
    columns_number = matrix.shape[1]
    matrices = np.split(matrix, columns_number, axis=1)
    return matrices
