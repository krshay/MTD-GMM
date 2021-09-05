# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 18:08:24 2021

@author: Shay Kreymer
"""


import numpy as np

import time

import matplotlib.pyplot as plt

import utils

import scipy.optimize as optimize

import shelve

np.random.seed(1)

L = 21
gamma = 0.2
SNR = 50

shifts_2nd = utils.shifts_2nd(L)
shifts_3rd = utils.shifts_3rd_reduced(L)

gamma0 = 0.18
x_gamma0s = []
Nguesses = 5
for i in range(Nguesses):
    x0 = np.random.rand(L)
    x0 = x0 / np.linalg.norm(x0)
    
    x_gamma0 = np.concatenate((x0, np.array([gamma0])))
    x_gamma0s.append(x_gamma0)

Nsizes = 10
sizes = np.logspace(4, 7.5, num=Nsizes)

Niters = 20
xs = []
for it in range(Niters):
    x = np.random.rand(L)
    x = x / np.linalg.norm(x)
    xs.append(x)
sigma2 = (np.linalg.norm(x) ** 2) / (L * SNR)

results_mom = np.zeros((Niters, Nsizes, Nguesses), dtype=optimize.optimize.OptimizeResult)
results_gmm = np.zeros((Niters, Nsizes, Nguesses), dtype=optimize.optimize.OptimizeResult)
times_mom = np.zeros((Niters, Nsizes, Nguesses))
times_gmm = np.zeros((Niters, Nsizes, Nguesses))

results_final_mom = np.zeros((Niters, Nsizes), dtype=optimize.optimize.OptimizeResult)
results_final_gmm = np.zeros((Niters, Nsizes), dtype=optimize.optimize.OptimizeResult)
times_final_mom = np.zeros((Niters, Nsizes, Nguesses))
times_final_gmm = np.zeros((Niters, Nsizes, Nguesses))
for it in range(Niters):
    print(it)
    x = xs[it]
    for (idx, N) in enumerate(sizes):
        N = int(N)
        print(N)

        y_clean = utils.generate_micrograph_1d(x, gamma, L, N)
        y = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))
        del y_clean
        ac1_y = utils.ac1(y)
        ac2_y = np.zeros((len(shifts_2nd), ))
        for (i, shift) in enumerate(shifts_2nd):
            ac2_y[i] = utils.ac2(y, shift)
        ac3_y = np.zeros((len(shifts_3rd), ))
        for (i, shifts) in enumerate(shifts_3rd):
            ac3_y[i] = utils.ac3(y, shifts[0], shifts[1])
        

        W = utils.calc_W_heuristic(shifts_2nd, shifts_3rd)
        samples = utils.sample(y, L)
        del y
        estimations_mom = []
        estimations_gmm = []
        for i in range(Nguesses):
            start = time.time()
            estimation_mom = utils.opt(x_gamma0s[i], ac1_y, ac2_y, ac3_y, shifts_2nd, shifts_3rd, sigma2, W)
            times_mom[it, idx, i] = time.time() - start
            estimations_mom.append(estimation_mom)
            
            start = time.time()
            f_gmm = utils.calc_function_gmm(samples, x_gamma0s[i][-1], x_gamma0s[i][:-1], shifts_2nd, shifts_3rd, sigma2)
            cov_f = np.cov(np.sqrt(N) * f_gmm)
            W_gmm = np.linalg.inv(cov_f)
            W_gmm = W_gmm
            estimation_gmm = utils.opt(x_gamma0s[i], ac1_y, ac2_y, ac3_y, shifts_2nd, shifts_3rd, sigma2, W_gmm)
            times_gmm[it, idx, i] = time.time() - start
            estimations_gmm.append(estimation_gmm)
        del samples
        del f_gmm
        results_mom[it, idx, :] = estimations_mom
        results_gmm[it, idx, :] = estimations_gmm

for (idx, _) in enumerate(sizes):
    for it in range(Niters):
        estimations_mom = results_mom[it, idx, :]
        estimations_gmm = results_gmm[it, idx, :]
        estimation_mom = estimations_mom[0]
        time_mom = times_mom[it, idx, 0]
        estimation_gmm = estimations_gmm[0]
        time_gmm = times_gmm[it, idx, 0]
        for i in range(Nguesses - 1):
            if estimations_mom[i + 1].fun < estimation_mom.fun:
                estimation_mom = estimations_mom[i + 1]
                time_mom = times_mom[it, idx, i + 1]
            if estimations_gmm[i + 1].fun < estimation_gmm.fun:
                estimation_gmm = estimations_gmm[i + 1]
                time_gmm = times_gmm[it, idx, i + 1]
        results_final_mom[it, idx] = estimation_mom
        times_final_mom[it, idx] = time_mom
        results_final_gmm[it, idx] = estimation_gmm
        times_final_gmm[it, idx] = time_gmm
        
errs_mom = np.zeros((Niters, Nsizes))
Number_Iterations_mom = np.zeros((Niters, Nsizes))
errs_gmm = np.zeros((Niters, Nsizes))
Number_Iterations_gmm = np.zeros((Niters, Nsizes))
for (idx, _) in enumerate(sizes):
    for it in range(Niters):
        errs_mom[it, idx] = utils.calc_err(xs[it], results_final_mom[it, idx].x[:-1])
        Number_Iterations_mom[it, idx] = results_final_mom[it, idx].nit
        errs_gmm[it, idx] = utils.calc_err(xs[it], results_final_gmm[it, idx].x[:-1])
        Number_Iterations_gmm[it, idx] = results_final_gmm[it, idx].nit

plt.close("all")
with plt.style.context('ieee'):
    fig = plt.figure()
    plt.loglog(sizes, np.median(errs_mom, axis=0), 'b', label=r'Method of Moments', lw=2)
    plt.loglog(sizes, np.median(errs_gmm, axis=0), 'r', label=r'Generalized Method of Moments', lw=2)
    plt.loglog(sizes, np.median(errs_mom, axis=0)[0]*(sizes/sizes[0])**(-1/2), 'k--', lw=1)
    plt.loglog(sizes, np.median(errs_gmm, axis=0)[0]*(sizes/sizes[0])**(-1/2), 'k--', lw=1)
    plt.legend(loc=1, fontsize=6)
    plt.xlabel('Measurement length')
    plt.ylabel('Median estimation error')
    fig.tight_layout()
    plt.show()
    plt.savefig(r'C:/Users/kreym/Documents/GitHub/MTD-GMM/paper/figures/experiment_size_err.pdf')

    # fig = plt.figure()
    # plt.semilogx(sizes, np.median(Number_Iterations_mom, axis=0), label=r'Method of Moments', lw=2)
    # plt.semilogx(sizes, np.median(Number_Iterations_gmm, axis=0), label=r'Generalized Method of Moments', lw=2)
    # plt.xlabel('Measurement length [pixels]')
    # plt.ylabel('Median number of iterations')
    # plt.yticks(list(range(0,650,100)))
    # fig.tight_layout()
    # plt.show()
    # plt.savefig(r'C:/Users/kreym/Documents/GitHub/MTD-GMM/paper/figures/experiment_size_iters.pdf')

filename=r'shelve_size.out'
# my_shelf = shelve.open(filename,'n') # 'n' for new

# for key in dir():
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         #
#         # __builtins__, my_shelf, and imported modules can not be shelved.
#         #
#         print('ERROR shelving: {0}'.format(key))
#     except AttributeError:
#         print('ERROR shelving: {0}'.format(key))
# my_shelf.close()

# %% load
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()