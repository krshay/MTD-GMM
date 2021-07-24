# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:03:45 2021

@author: Shay Kreymer
"""

import numpy as np

import matplotlib.pyplot as plt

import utils

np.random.seed(1)

L = 15
N = 100000
gamma = 0.2

x = np.random.rand(L)
x = 10 * x / np.linalg.norm(x)
y_clean = utils.generate_micrograph_1d(x, gamma, L, N)

SNR = 500
sigma2 = (np.linalg.norm(x) ** 2) / (L * SNR)
y = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

shifts_2nd = utils.shifts_2nd(L)

shifts_3rd = utils.shifts_3rd_reduced(L)

ac1_y = utils.ac1(y)

ac2_y = np.zeros((len(shifts_2nd), ))
for (i, shift) in enumerate(shifts_2nd):
    ac2_y[i] = utils.ac2(y, shift)
    
ac3_y = np.zeros((len(shifts_3rd), ))
for (i, shifts) in enumerate(shifts_3rd):
    ac3_y[i] = utils.ac3(y, shifts[0], shifts[1])

L2 = len(shifts_2nd)
L3 = len(shifts_3rd)
W = np.eye(1 + L2 + L3)

gamma0 = 0.195
x0 = np.random.rand(L)
x0 = (np.linalg.norm(x0) ** 2) * x0 / np.linalg.norm(x0)

x_gamma0 = np.concatenate((x0, np.array([gamma0])))

estimation = utils.opt(x_gamma0, ac1_y, ac2_y, ac3_y, shifts_2nd, shifts_3rd, sigma2, W)
x_est = estimation.x[ :L]
gamma_est = estimation.x[-1]

err_est = utils.calc_err(x, x_est)

print(f'The error for the MoM is {err_est}.')

samples = utils.sample(y, L)

f_gmm = utils.calc_function_gmm(samples, gamma0, x0, shifts_2nd, shifts_3rd, sigma2)
cov_f = np.cov(f_gmm)
W_gmm = np.linalg.inv(cov_f)
W_gmm = W_gmm / np.sum(W_gmm)

estimation_gmm = utils.opt(x_gamma0, ac1_y, ac2_y, ac3_y, shifts_2nd, shifts_3rd, sigma2, W_gmm)
x_est_gmm = estimation_gmm.x[ :L]
gamma_est_gmm = estimation_gmm.x[-1]

err_est_gmm = utils.calc_err(x, x_est_gmm)

print(f'The error for the GMM is {err_est_gmm}.')
