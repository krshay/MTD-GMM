# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:03:45 2021

@author: Shay Kreymer
"""

import numpy as np

import matplotlib.pyplot as plt
import utils

np.random.seed(10)

L = 10
N = 120
gamma = 0.4

x = np.random.rand(L)
x = x / np.linalg.norm(x)
y_clean = utils.generate_micrograph_1d(x, gamma, L, N)

SNR = 50
sigma2 = 1 / (L * SNR)
y1 = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

SNR = 0.1
sigma2 = 1 / (L * SNR)
y2 = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

width = 3.487
height = width * 1.618
plt.close("all")
fig = plt.figure()
plt.plot(y_clean, 'k', linewidth=10)
fig.set_size_inches(4*width, 4*height)
plt.savefig(r'paper/figures\y_clean.pdf')
fig = plt.figure()
plt.plot(y1, 'k', linewidth=10)
fig.set_size_inches(4*width, 4*height)
plt.savefig(r'paper/figures\y_SNR50.pdf')
fig = plt.figure()
plt.plot(y2, 'k', linewidth=10)
fig.set_size_inches(4*width, 4*height)
plt.savefig(r'paper/figures\y_SNR01.pdf')
