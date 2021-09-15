# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:03:45 2021

@author: Shay Kreymer
"""

import numpy as np

import matplotlib.pyplot as plt

import utils

np.random.seed(100)

L = 21
N = 10**9
gamma = 0.2

x = 0.25 * np.ones((L ,))
x[2] = 0.5
x[14] = 0.97
x[15] = 0.85
x[16] = 0.55
x[17] = 0.40
x[2:14] = 1
x = x / np.linalg.norm(x)
y_clean = utils.generate_micrograph_1d(x, gamma, L, N)

SNR = 0.1
sigma2 = (np.linalg.norm(x) ** 2) / (L * SNR)
y = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

shifts_2nd = utils.shifts_2nd(L)

shifts_3rd = utils.shifts_3rd_reduced(L)


L2 = len(shifts_2nd)
L3 = len(shifts_3rd)
W = utils.calc_W_heuristic(shifts_2nd, shifts_3rd)


gamma0 = 0.18

estimations_mom = []
estimations_gmm = []

x0 = np.random.rand(L)
x0 = (np.linalg.norm(x0) ** 2) * x0 / np.linalg.norm(x0)

x_gamma0 = np.concatenate((x0, np.array([gamma0])))
    
for sz in np.array([10**4, 10**6, 10**9]):
    print(sz)
    yi = y[:sz]
    if sz==10**9:
        del y
    ac1_yi = utils.ac1(yi)
    
    ac2_yi = np.zeros((len(shifts_2nd), ))
    for (i, shift) in enumerate(shifts_2nd):
        ac2_yi[i] = utils.ac2(yi, shift)
        
    ac3_yi = np.zeros((len(shifts_3rd), ))
    for (i, shifts) in enumerate(shifts_3rd):
        ac3_yi[i] = utils.ac3(yi, shifts[0], shifts[1])
    
    samplesi = utils.sample(yi, L)
    del yi
    
    estimation_mom = utils.opt(x_gamma0, ac1_yi, ac2_yi, ac3_yi, shifts_2nd, shifts_3rd, sigma2, W)
    estimations_mom.append(estimation_mom)
    
    f_gmm = utils.calc_function_gmm(samplesi, gamma0, x0, shifts_2nd, shifts_3rd, sigma2)
    cov_f = np.cov(f_gmm)
    W_gmm = np.linalg.inv(np.sqrt(N) * cov_f)
    W_gmm = W_gmm
    
    estimation_gmm = utils.opt(x_gamma0, ac1_yi, ac2_yi, ac3_yi, shifts_2nd, shifts_3rd, sigma2, W_gmm)
    estimations_gmm.append(estimation_gmm)
    
plt.figure()
plt.plot(x); plt.plot(estimations_mom[0].x[:-1]); plt.plot(estimations_gmm[0].x[:-1])
    
plt.figure()
plt.plot(x); plt.plot(estimations_mom[1].x[:-1]); plt.plot(estimations_gmm[1].x[:-1])

 
plt.figure()
plt.plot(x); plt.plot(estimations_mom[2].x[:-1]); plt.plot(estimations_gmm[2].x[:-1])
