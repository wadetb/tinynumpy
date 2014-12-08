# -*- coding: utf-8 -*-
# Copyright (c) 2014, Almar Klein and Wade Brainerd
# tinynumpy is distributed under the terms of the MIT License.

""" Benchmarks for tinynumpy

Findings:

* A list of floats costs about 33 bytes per float
* A list if ints costs anout 41 bytes per int
* A huge list of ints 0-255, costs about 1 byter per int
* Python list takes about 5-6 times as much memory than array for 64bit 
  data types. Up to 40 times as much for uint8, unless Python can reuse
  values.
* __slots__ help reduce the size of custom classes
* tinynumpy is about 100 times slower than numpy
* using _toflatlist instead of flat taks 50-60% of time (but more memory)

"""

from __future__ import division

import os
import sys
import time
import subprocess

import numpy as np
import tinynumpy as tnp


def _prettymem(n):
    if n > 2**20:
        return '%1.2f MiB' % (n / 2**20)
    elif n > 2**10:
        return '%1.2f KiB' % (n / 2**10)
    else:
        return '%1.0f B' % n

def _prettysec(n):
    if n < 0.0001:
        return '%1.2f us' % (n * 1000000)
    elif n < 0.1:
        return '%1.2f ms' % (n * 1000)
    else:
        return '%1.2f s' % n


code_template = """
import psutil
import os
import random
import numpy as np
import tinynumpy as tnp

N = 100 * 1000
M = 1000 * 1000

class A(object):
    def __init__(self):
        self.foo = 8
        self.bar = 3.3

class B(object):
    __slots__ = ['foo', 'bar']
    def __init__(self):
        self.foo = 8
        self.bar = 3.3

def getmem():
    process = psutil.Process(os.getpid())
    return process.get_memory_info()[0]

M0 = getmem()
%s
M1 = getmem()
print(M1-M0)
"""

def measure_mem(what, code, divide=1):
    
    cmd = [sys.executable, '-c',  code_template % code]
    res = subprocess.check_output(cmd, cwd=os.getcwd()).decode('utf-8')
    m = int(res) / divide
    
    print('Memory for %s:%s%s' % 
          (what, ' '*(22-len(what)), _prettymem(m)))


def measure_speed(what, func, *args, **kwargs):
    
    
    N = 1
    t0 = time.perf_counter()
    func(*args, **kwargs)
    t1 = time.perf_counter()
    while (t1 - t0) < 0.2:
        N *= 10
        t0 = time.perf_counter()
        for i in range(N):
            func(*args, **kwargs)
        t1 = time.perf_counter()
    
    te = t1 - t0
    print('Time for %s:%s%s  (%i iters)' % 
          (what, ' '*(22-len(what)), _prettysec(te/N), N))


if __name__ == '__main__':
    N = 100 * 1000
    M = 1000 * 1000
    
    print('=== MEMORY ====')
    measure_mem('floats 0-M', 'L = [i*1.0 for i in range(N)]', N)
    measure_mem('ints 0-M', 'L = [i for i in range(N)]', N)
    measure_mem('ints 0-255', 'L = [int(random.uniform(0, 255)) for i in range(N)]', N)
    
    measure_mem('regular object', 'L = [A() for i in range(N)]', N)
    measure_mem('object with slots', 'L = [B() for i in range(N)]', N)
    
    measure_mem('    Numpy arr size 1', 'L = [np.ones((1,)) for i in range(N)]', N)
    measure_mem('Tinynumpy arr size 1', 'L = [tnp.ones((1,)) for i in range(N)]', N)
    
    measure_mem('    Numpy arr size M', 'a = np.ones((M,))')
    measure_mem('Tinynumpy arr size M', 'a = tnp.ones((M,))')
    
    print('=== SPEED ====')
    
    a1 = np.ones((100, 100))
    a2 = tnp.ones((100, 100))
    measure_speed('    numpy sum 10k', a1.sum)
    measure_speed('tinynumpy sum 10k', a2.sum)
    measure_speed('    numpy max 10k', a1.max)
    measure_speed('tinynumpy max 10k', a2.max)
    
    a1 = np.ones((1000, 1000))
    a2 = tnp.ones((1000, 1000))
    measure_speed('    numpy sum 1M', a1.sum)
    measure_speed('tinynumpy sum 1M', a2.sum)
    measure_speed('    numpy max 1M', a1.max)
    measure_speed('tinynumpy max 1M', a2.max)
