# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 19:53:50 2021

@author: Shay Kreymer
"""

import numpy as np

import itertools

from scipy.optimize import minimize


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

def dac2(z, shift):
    l = len(z)
    z_shifted = np.zeros((l + shift, ))
    z_shifted[shift: shift + l] = z
    z_tmp = np.zeros((l + shift, ))
    z_tmp[ :l] = z
    z = z_tmp
    dac2 = np.zeros((l, ))
    for i in range(l):
        dac2[i] = (z_shifted[i] + z[i + shift]) / l
    return dac2

def ac3(z, shift1, shift2):
    l = len(z)
    z_shifted1 = np.zeros((l + shift1, ))
    z_shifted1[shift1: shift1 + l] = z
    z_shifted1 = z_shifted1[ :l]
    z_shifted2 = np.zeros((l + shift2, ))
    z_shifted2[shift2: shift2 + l] = z
    z_shifted2 = z_shifted2[ :l]
    return np.sum(z * z_shifted1 * z_shifted2) / l

def dac3(z, shift1, shift2):
    l = len(z)
    z_shifted1 = np.zeros((l + np.maximum(shift1, shift2), ))
    z_shifted1[shift1: shift1 + l] = z
    z_shifted2 = np.zeros((l + np.maximum(shift1, shift2), ))
    z_shifted2[shift2: shift2 + l] = z
    z_tmp = np.zeros((l + np.maximum(shift1, shift2), ))
    z_tmp[ :l] = z
    z = z_tmp
    dac3 = np.zeros((l, ))
    for i in range(l):
        dac3[i] = (z_shifted1[i] * z_shifted2[i] + z[i + shift1] * z_shifted2[i + shift1] + z[i + shift2] * z_shifted1[i + shift2]) / l
    return dac3

def shifts_2nd(L):
    return list(np.arange(L))

def shifts_3rd(L):
    return list(itertools.product(np.arange(L), np.arange(L)))

def shifts_3rd_reduced(L):
    return list(itertools.combinations(np.arange(L), r=2)) + [(np.arange(L)[i], np.arange(L)[i]) for i in range(L)]

def sample(y, L):
    samples = np.zeros((L, len(y) - L + 1))
    for i in range(len(y) - L + 1):
        samples[ :, i] = y[i: i + L]
    return samples

def calcM1(samples):
    return samples[0, :]

def calcM2(samples, shift):
    return samples[0, :] * samples[shift, :]

def calcM3(samples, shift1, shift2):
    return samples[0, :] * samples[shift1, :] * samples[shift2, :]

def calc_function_gmm(samples, gamma, x, shifts_2nd, shifts_3rd, sigma2):
    Nsamples = np.shape(samples)[1]
    L2 = len(shifts_2nd)
    L3 = len(shifts_3rd)
    f_gmm = np.zeros((1 + L2 + L3, Nsamples))
    f_gmm[0, :] = calcM1(samples) - gamma * ac1(x)
    for (i, shift) in enumerate(shifts_2nd):
        f_gmm[1 + i, :] = calcM2(samples, shift) - gamma * ac2(x, shift)
        if shift == 0:
            f_gmm[1 + i, :] -= sigma2
    for (i, shifts) in enumerate(shifts_3rd):
        f_gmm[1 + L2 + i, :] = calcM3(samples, shifts[0], shifts[1]) - gamma * ac3(x, shifts[0], shifts[1])
        if shifts[0] == 0:
            f_gmm[1 + L2 + i, :] -= sigma2 * gamma * ac1(x)
        if shifts[1] == 0:
            f_gmm[1 + L2 + i, :] -= sigma2 * gamma * ac1(x)
        if shifts[0] == shifts[1]:
            f_gmm[1 + L2 + i, :] -= sigma2 * gamma * ac1(x)
    return f_gmm
    
def calc_g_dg(ac1_y, ac2_y, ac3_y, gamma, x, shifts_2nd, shifts_3rd, sigma2):
    L = len(x)
    L2 = len(shifts_2nd)
    L3 = len(shifts_3rd)
    g = np.zeros((1 + L2 + L3, ))
    dg = np.zeros((L + 1, 1 + L2 + L3))
    g[0] = ac1_y - gamma * ac1(x)
    dg[:L, 0] = - gamma / L
    dg[-1, 0] = - ac1(x)
    for (i, shift) in enumerate(shifts_2nd):
        g[1 + i] = ac2_y[i] - gamma * ac2(x, shift)
        dg[ :L, 1 + i] = - gamma * dac2(x, shift)
        dg[-1, 1 + i] = - ac2(x, shift)
        if shift == 0:
            g[1 + i] -= sigma2
    for (i, shifts) in enumerate(shifts_3rd):
        g[1 + L2 + i] = ac3_y[i] - gamma * ac3(x, shifts[0], shifts[1])
        dg[ :L, 1 + L2 + i] = - gamma * dac3(x, shifts[0], shifts[1])
        dg[-1, 1 + L2 + i] = - ac3(x, shifts[0], shifts[1])
        if shifts[0] == 0:
            g[1 + L2 + i] -= sigma2 * gamma * ac1(x)
            dg[ :L, 1 + L2 + i] -= sigma2 * gamma / L
            dg[-1, 1 + L2 + i] -= sigma2 * ac1(x)
        if shifts[1] == 0:
            g[1 + L2 + i] -= sigma2 * gamma * ac1(x)
            dg[ :L, 1 + L2 + i] -= sigma2 * gamma / L
            dg[-1, 1 + L2 + i] -= sigma2 * ac1(x)
        if shifts[0] == shifts[1]:
            g[1 + L2 + i] -= sigma2 * gamma * ac1(x)
            dg[ :L, 1 + L2 + i] -= sigma2 * gamma / L
            dg[-1, 1 + L2 + i] -= sigma2 * ac1(x)
    return g, dg

def calc_W_heuristic(shifts_2nd, shifts_3rd):
    L2 = len(shifts_2nd)
    L3 = len(shifts_3rd)
    W = np.ones((1 + L2 + L3, ))
    W[1: 1 + L2] = 1 / L2
    W[1 + L2:] = 1/ L3
    return np.diag(W)

def calc_f_df(x_gamma, ac1_y, ac2_y, ac3_y, shifts_2nd, shifts_3rd, sigma2, W):
    gamma = x_gamma[-1]
    x = x_gamma[ :-1]
    g, dg = calc_g_dg(ac1_y, ac2_y, ac3_y, gamma, x, shifts_2nd, shifts_3rd, sigma2)
    f = g @ W @ g
    df = 2 * dg @ W @ g
    return f, df

def opt(x_gamma0, ac1_y, ac2_y, ac3_y, shifts_2nd, shifts_3rd, sigma2, W, gtol=1e-12):
    return minimize(fun=calc_f_df, x0=x_gamma0, method='BFGS', jac=True, options={'disp':True, 'gtol': gtol, 'maxiter':500}, args = (ac1_y, ac2_y, ac3_y, shifts_2nd, shifts_3rd, sigma2, W))

def calc_err(x, x_est):
    return np.linalg.norm(x - x_est) / np.linalg.norm(x)
    