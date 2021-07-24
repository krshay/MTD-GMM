# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 19:53:50 2021

@author: Shay Kreymer
"""

import numpy as np

import itertools

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

def ac1(z):
    return np.mean(z)

def ac2(z, shift):
    l = len(z)
    z_shifted = np.zeros((l + shift, ))
    z_shifted[shift: shift + l] = z
    z_shifted = z_shifted[ :l]
    return np.sum(z * z_shifted) / l

def ac3(z, shift1, shift2):
    l = len(z)
    z_shifted1 = np.zeros((l + shift1, ))
    z_shifted1[shift1: shift1 + l] = z
    z_shifted1 = z_shifted1[ :l]
    z_shifted2 = np.zeros((l + shift2, ))
    z_shifted2[shift2: shift2 + l] = z
    z_shifted2 = z_shifted2[ :l]
    return np.sum(z * z_shifted1 * z_shifted2) / l

def shifts_2nd(L):
    return list(np.arange(L))

def shifts_3rd(L):
    return list(itertools.product(np.arange(L), np.arange(L)))

def calc_g_dg(ac1_y, ac2_y, ac3_y, gamma, x, shifts_2nd, shifts_3rd):
    g = np.zeros((1 + len(shifts_2nd) + len(shifts_3rd), ))
    g[0] = ac1_y - gamma * ac1(x)
    for (i, shift) in enumerate(shifts_2nd):
        g[1 + i] = ac2_y[i] - gamma * ac2(x, shift)
    for (i, shifts) in enumerate(shifts_3rd):
        g[1 + len(shifts_2nd) + i] = ac3_y[i] - gamma * ac3(x, shifts[0], shifts[1])
    return g
    