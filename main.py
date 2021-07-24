# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:03:45 2021

@author: Shay Kreymer
"""

import numpy as np

# import matplotlib.pyplot as plt

import utils

np.random.seed(10)

L = 10
N = 1000
gamma = 0.4

x = np.random.rand(L)
x = x / np.linalg.norm(x)
y_clean = utils.generate_micrograph_1d(x, gamma, L, N)

SNR = 50
sigma2 = 1 / (L * SNR)
y = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

shifts_2nd = utils.shifts_2nd(L)

shifts_3rd = utils.shifts_3rd(L)

print(utils.ac3(y_clean, shifts_3rd[2][0], shifts_3rd[2][1]) / utils.ac3(x, shifts_3rd[2][0], shifts_3rd[2][1]))

