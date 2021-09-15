# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:03:45 2021

@author: Shay Kreymer
"""

import numpy as np

import matplotlib.pyplot as plt
import utils

np.random.seed(10)

L = 21
N = 180
gamma = 0.4

x = 0.25 * np.ones((L ,))
x[2] = 0.5
x[14] = 0.97
x[15] = 0.85
x[16] = 0.55
x[17] = 0.40
x[2:14] = 1
x = x / np.linalg.norm(x)
y_clean = utils.generate_micrograph_1d(x, gamma, L, N)

SNR = 50
sigma2 = np.linalg.norm(x)**2 / (L * SNR)
y1 = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

SNR = 0.1
sigma2 = np.linalg.norm(x)**2 / (L * SNR)
y2 = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

plt.close("all")
with plt.style.context('ieee'):
    plt.figure()
    plt.plot(y_clean, 'k')
    plt.savefig(r'paper/figures\y_clean.pdf')
    plt.figure()
    plt.plot(y1, 'k')
    plt.savefig(r'paper/figures\y_SNR50.pdf')
    plt.figure()
    plt.plot(y2, 'k')
    plt.savefig(r'paper/figures\y_SNR01.pdf')
