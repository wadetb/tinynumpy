# Copyright (c) 2014 Eric Allen Youngson

"""This module is used to implement native Python functions to replace those
    called from numpy, when not available"""
# Written by Eric Youngson eric@scneco.com / eayoungs@gmail.com
# Succession Ecological Services: Portland, Oregon

class LinAlgError(Exception):
    pass

def det(A):
    if len(A) == 3 and [len(vec)==3 for vec in A]:
        try:
            # http://mathworld.wolfram.com/Determinant.html
            det_A = (A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] *
                     A[2][0] + A[0][2] * A[1][0] * A[2][1] - (A[0][2] *
                     A[1][1] * A[2][0] + A[0][1] * A[1][0] * A[2][2] +
                     A[0][0] * A[1][2] * A[2][1]))
        except LinAlgError as e:
            det_A = e
    else:
        raise IndexError('Vector has invalid dimensions')
    return det_A
