# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 19:53:50 2021

@author: Shay Kreymer
"""

import numpy as np

def generate_micrograph_1d(x, gamma, L, N):
    y = np.zeros((N, ))
    
    number_of_signals = int(gamma * N / L)
    zero_num = N - number_of_signals * (2 * L - 1) #the number of free zeroes that can be used
    total_num_of_blocks = number_of_signals + zero_num # the total number of space we can use while building the array
    x_modified = np.zeros((L + L - 1))
    x_modified[ :L] = x
    
    a = np.zeros((total_num_of_blocks, ))
    a[ :number_of_signals] = 1
    np.random.shuffle(a)
    signal_indices = (a == 1)
    
    ii = 0
    i = 0
    while i < N:
        if signal_indices[ii]:
            y[i: i + 2 * L - 1] = x_modified
            i += 2 * L - 1
        else:
            i += 1
        ii += 1

    return y
