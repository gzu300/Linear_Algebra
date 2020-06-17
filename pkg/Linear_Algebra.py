import numpy as np
from numpy.linalg import inv
from functools import reduce
import argparse

'''
This is just for practice purpose. Scipy and numpy
have already provided enough tools.
'''
def x_hat(A, b):
    '''
    the x to project b onto A when b is not in col space of A
    '''
    aTa = A.T.dot(A)
    aTa_inv = inv(aTa)
    return aTa_inv.dot(A.T).dot(b)

def make_perm_mx(size, original_row, to_swap):
    P = np.identity(size)
    P[original_row, :] = P[to_swap, :] = 0
    P[original_row, to_swap] = P[to_swap, original_row] = 1
    return P

def make_subtract_mx(size, current_row, current_col, factor):
    I = np.identity(size)
    I[current_row, current_col] = factor
    return I

def LU(A):
    '''
    LU factorization by elimination.
    U = e1*p1*e2*...*A = E*A
    L = inv(E)
    A = L*U
    E as elimination matrix. p and e are permutation and subtraction mx at each step.
    Pivots located in U
    Note: this workflow is not optimized for performance. But efficiency is not considered here.
    '''
    nrows, ncols = A.shape
    U = A.copy()
    E = np.identity(nrows)
    npivots = nrows if nrows <= ncols else ncols # mx might not be square. number of pivots take the smaller value
    for col in range(npivots): # loop through columns
        row_w_max_val = A[:, col].argmax() # find the largest value as the pivot
        p = make_perm_mx(npivots, col, row_w_max_val)
        U = p.dot(U)
        E = p.dot(E)
        for row in range(col,npivots):
            if row != col: # omit the diagnal
                factor = U[row, col]/U[col, col]*-1
                e = make_subtract_mx(npivots, row, col, factor) # make subtraction mx for current pivot col
                E = e.dot(E)
                U = e.dot(U)
    L = inv(E)
    return L, U

def find_pivot(A):
    _, U = LU(A)
    return np.diagonal(U)

def find_det(A):
    pivots = find_pivot(A)
    det = reduce(lambda x, y: x*y, pivots)
    return det

def factorize(A=None):
    L, U = LU(A)
    pivots = find_det(A)
    return {'L': L, 'U': U, 'pivots': find_pivot(A), 'determines': find_det(A)}

def project(A=None, b=None):
    '''
    * is dot product
    p = P*b = a*inv(aT*a)*aT*b
    '''
    aTa = A.T.dot(A)
    aTa_inv = inv(aTa)
    x_bar = aTa_inv.dot(A.T)
    p = A.dot(x_bar).dot(b)
    return p, x_bar



### command parsing###
def input_to_mx(i):
    return np.array(eval(i))

def run_cmd(parsed_arg, cmd_name):
    '''
    gather arguments for each sub-commands
    and parse arguments into correct data type
    '''
    cmd = eval(getattr(parsed_arg, cmd_name))
    _, *keys= parsed_arg.__dict__.keys() # exclude the first 'command' argument
    _, *values = parsed_arg.__dict__.values()
    transformed_values = [input_to_mx(each) for each in values if each.strip().startswith('[')]
    args = dict(zip(keys, transformed_values))
    return cmd(**args)

def show(result):
    print(result)

def main():
    '''
    reflection is used to call function.
    So the names of subcommands need to be same as the names of the function.
    '''

    arg = argparse.ArgumentParser(description='Calculations in Linear Algebra')
    sub_parsers = arg.add_subparsers(
        title='calculation',
        description='enter the calculation to be done',
        help='calculations',
        dest='command',
        required=True
    )

    ###projection###
    projection_args = sub_parsers.add_parser('project', help='project help')
    projection_args.add_argument('-A', '--A', help='matrix to be projected on. example: [[1,2,3], [1,2,3]] is a 2x3 matrix.')
    projection_args.add_argument('-b', '--b', help='matrix to poject')

    factorization_args = sub_parsers.add_parser('factorize', help='LU factorization. return L, U, pivots and determines')
    factorization_args.add_argument('-A', '--A', help='matrix to be projected on. example: [[2,1,0],[1,2,1],[0,1,2]] is a 3x3 matrix.')

    args = arg.parse_args()
    
    result = run_cmd(args, 'command')
    show(result)
    



if __name__ == '__main__':
    # A = np.array([1,0,1,1,1,2]).reshape(3,2)
    # b = np.array([6,0,0])
    # print(x_hat(A, b))
    main()