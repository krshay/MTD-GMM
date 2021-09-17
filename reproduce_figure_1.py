# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:03:45 2021

@author: Shay Kreymer
"""

import numpy as np

import matplotlib.pyplot as plt
import utils

np.random.seed(10)

L = 11
N = 120
gamma = 0.4

x = np.zeros((L ,))
x = np.zeros((L ,))
x[0] = 0.5; x[1] = 0.7; x[2] = 0.2; x[3] = 0.5; x[4] = 1; x[5] = 0.25; x[6] = 1; x[7] = 0.85; x[8] = 0.2; x[9] = 0.55; x[10] = 0.1

x = x / np.linalg.norm(x)
x = x / np.linalg.norm(x)
y_clean = utils.generate_micrograph_1d(x, gamma, L, N)

SNR = 50
sigma2 = np.linalg.norm(x)**2 / (L * SNR)
y1 = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

SNR = 0.5
sigma2 = np.linalg.norm(x)**2 / (L * SNR)
y2 = y_clean + np.random.normal(loc=0, scale=np.sqrt(sigma2), size=np.shape(y_clean))

plt.close("all")
with plt.style.context('ieee'):
    fig = plt.figure()
    plt.plot(y_clean, 'k')
    fig.tight_layout()
    plt.savefig(r'paper/figures\y_clean.pdf')
    fig = plt.figure()
    plt.plot(y1, 'k')
    fig.tight_layout()
    plt.savefig(r'paper/figures\y_SNR50.pdf')
    fig = plt.figure()
    plt.plot(y2, 'k')
    fig.tight_layout()
    plt.savefig(r'paper/figures\y_SNR01.pdf')
