# -*- coding: utf-8 -*-
# Copyright (c) 2014, Almar Klein and Wade Brainerd
# tinynumpy is distributed under the terms of the MIT License.

""" Test suite for tinynumpy
"""

import os
import sys
import ctypes

import pytest
from _pytest import runner
from pytest import raises, skip

import tinynumpy as tnp

# Numpy is optional. If not available, will compare against ourselves.
try:
    import numpy as np
except ImportError:
    np = tnp


def test_TESTING_WITH_NUMPY():
    # So we can see in the result whether numpy was used
    if np is None or np is tnp:
        skip('Numpy is not available')


def test_shapes_and_strides():
    
    for shape in [(9, ), (109, ), 
                  (9, 4), (109, 104), 
                  (9, 4, 5), (109, 104, 105),
                  (9, 4, 5, 6),  # not (109, 104, 105, 106) -> too big
                  ]:
        
        # Test shape and strides
        a = np.empty(shape)
        b = tnp.empty(shape)
        assert a.ndim == len(shape)
        assert a.ndim == b.ndim
        assert a.shape == b.shape
        assert a.strides == b.strides
        assert a.size == b.size
        
        # Also test repr length
        if b.size > 100:
            assert len(repr(b)) < 80
        else:
            assert len(repr(b)) > (b.size * 3)  # "x.0" for each element


def test_repr():
    
    for dtype in ['float32', 'int32']:
        for data in [[1, 2, 3, 4, 5, 6, 7, 8],
                    [[1, 2], [3, 4], [5, 6], [7, 8]],
                    [[[1, 2], [3, 4]],[[5, 6], [7, 8]]],
                    ]:
            a = np.array(data, dtype)
            b = tnp.array(data, dtype)
            # Compare line by line (forget leading whitespace)
            charscompared = 0
            for l1, l2 in zip(repr(a).splitlines(), repr(b).splitlines()):
                l1, l2 = l1.rstrip(), l2.rstrip()
                l1, l2 = l1.split('dtype=')[0], l2.split('dtype=')[0]
                assert l1 == l2
                charscompared += len(l1)
            assert charscompared > (3 * b.size)


def test_dtype():
    
    for shape in [(9, ), (9, 4), (9, 4, 5)]:
        for dtype in ['bool', 'int8', 'uint8', 'int16', 'uint16',
                      'int32', 'uint32', 'float32', 'float64']:
            a = np.empty(shape, dtype=dtype)
            b = tnp.empty(shape, dtype=dtype)
            assert a.shape == b.shape
            assert a.dtype == b.dtype
            assert a.itemsize == b.itemsize
    
    raises(TypeError, tnp.zeros, (9, ), 'blaa')
    
    assert tnp.array([1.0, 2.0]).dtype == 'float64'
    assert tnp.array([1, 2]).dtype == 'int64'


def test_reshape():
    
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    b = tnp.array([1, 2, 3, 4, 5, 6, 7, 8])
    
    for shape in [(2, 4), (4, 2), (2, 2, 2), (8,)]:
        a.shape = shape
        b.shape = shape
        assert a.shape == b.shape
        assert a.strides == b.strides
    
    a.shape = 2, 4
    b.shape = 2, 4
    
    # Test transpose
    assert b.T.shape == (4, 2)
    assert (a.T == b.T).all()
    assert (b.T.T == b).all()
    
    # Make non-contiguous versions
    a2 = a[:, 2:]
    b2 = b[:, 2:]
    
    # Test contiguous flag
    assert a.flags['C_CONTIGUOUS']
    assert not a2.flags['C_CONTIGUOUS']
    
    # Test base
    assert a2.base is a
    assert b2.base is b
    assert a2[:].base is a
    assert b2[:].base is b
    
    # Fail
    with raises(ValueError):  # Invalid shape
        a.shape = (3, 3)
    with raises(ValueError):
        b.shape = (3, 3)
    with raises(AttributeError):  # Cannot reshape non-contiguous arrays
        a2.shape = 4,
    with raises(AttributeError):
        b2.shape = 4,


def test_from_and_to_numpy():
    # This also tests __array_interface__
    
    for dtype in ['float32', 'float64', 'int32', 'uint32', 'uint8', 'int8']:
        for data in [[1, 2, 3, 4, 5, 6, 7, 8],
                    [[1, 2], [3, 4], [5, 6], [7, 8]],
                    [[[1, 2], [3, 4]],[[5, 6], [7, 8]]],
                    ]:
                        
            # Convert from numpy, from tinynumpy, to numpy
            a1 = np.array(data, dtype)
            b1 = tnp.array(a1)
            b2 = tnp.array(b1)
            a2 = np.array(b2)
            
            # Check if its the same
            for c in [b1, b2, a2]:
                assert a1.shape == c.shape
                assert a1.dtype == c.dtype
                assert a1.strides == c.strides
                assert (a1 == c).all()
    
    # Also test using a numpy array as a buffer
    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], 'float32')
    b = tnp.ndarray(a.shape, a.dtype, strides=a.strides, buffer=a.ravel())
    assert (a==b).all()
    
    # Test that is indeed same data
    a[0, 0] = 99
    assert (a==b).all()


def test_from_ctypes():
    
    for type, dtype in [(ctypes.c_int16, 'int16'), 
                        (ctypes.c_uint8, 'uint8'), 
                        (ctypes.c_float, 'float32'), 
                        (ctypes.c_double, 'float64')]:
        # Create ctypes array, possibly something that we get from a c lib
        buffer = (type*100)()
        
        # Create array!
        b = tnp.ndarray((4, 25), dtype, buffer=buffer)
        
        # Check that we can turn it into a numpy array
        a = np.array(b, copy=False)
        assert (a == b).all()
        assert a.dtype == dtype
        
        # Verify that both point to the same data
        assert a.__array_interface__['data'][0] == ctypes.addressof(buffer)
        assert b.__array_interface__['data'][0] == ctypes.addressof(buffer)
        
        # also verify offset in __array_interface__ here
        for a0, b0 in zip([a[2:], a[:, 10::2], a[1::2, 10:20:2]],
                          [b[2:], b[:, 10::2], b[1::2, 10:20:2]]):
            pa = a0.__array_interface__['data'][0]
            pb = b0.__array_interface__['data'][0]
            assert pa > ctypes.addressof(buffer)
            assert pa == pb


def test_from_bytes():
    skip('Need ndarray.frombytes or something')
    # Create bytes
    buffer = b'x' * 100
    
    # Create array!
    b = tnp.ndarray((4, 25), 'uint8', buffer=buffer)
    ptr = ctypes.cast(buffer, ctypes.c_void_p).value
    
    # Check that we can turn it into a numpy array
    a = np.array(b, copy=False)
    assert (a == b).all()
    
    # Verify readonly
    with raises(Exception):
        a[0, 0] = 1  
    with raises(Exception):
        b[0, 0] = 1  
    
    # Verify that both point to the same data
    assert a.__array_interface__['data'][0] == ptr
    assert b.__array_interface__['data'][0] == ptr
    
    # also verify offset in __array_interface__ here
    for a0, b0 in zip([a[2:], a[:, 10::2], a[1::2, 10:20:2]],
                        [b[2:], b[:, 10::2], b[1::2, 10:20:2]]):
        pa = a0.__array_interface__['data'][0]
        pb = b0.__array_interface__['data'][0]
        assert pa > ptr
        assert pa == pb


def test_creating_functions():
    
    # Test array
    b1 = tnp.array([[1, 2, 3], [4, 5, 6]])
    assert b1.shape == (2, 3)
    

def test_getitem():
     
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = tnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])


if __name__ == '__main__':
    
    # Run tests with or without pytest. Running with pytest creates
    # coverage report, running without allows PM debugging to fix bugs.
    if False:
        del sys.modules['tinynumpy']  # or coverage wont count globals
        pytest.main('-v -x --color=yes --cov tinynumpy --cov-config .coveragerc '
                    '--cov-report html %s' % repr(__file__))
        # Run these lines to open coverage report
        #import webbrowser
        #webbrowser.open_new_tab(os.path.join('htmlcov', 'index.html'))
    
    else:
        # Collect function names
        test_functions = []
        for line in open(__file__, 'rt').readlines():
            if line.startswith('def'):
                name = line[3:].split('(')[0].strip()
                if name.startswith('test_'):
                    test_functions.append(name)
        # Report
        print('Collected %i test functions.' % len(test_functions))
        # Run
        print('\nRunning tests ...\n')
        for name in test_functions:
            print('Running %s ... ' % name, end='')
            func = globals()[name]
            try:
                func()
            except runner.Skipped as err:
                print('SKIP:', err)
            except Exception:
                print('FAIL')
                raise
            else:
                print('OK')
