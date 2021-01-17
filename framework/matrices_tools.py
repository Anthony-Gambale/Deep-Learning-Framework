

# imports
import numpy as np


def combine(examples):
    '''
    examples: a list of x input matrices, or y output matrices. combine them into one matrix, where each column is one of the given matrices.
    X = combine([x1, x2, x3, x4 ... xm])
    Y = combine([y1, y2, y3, y4 ... ym])
    '''
    return np.concatenate(examples, axis=1)
