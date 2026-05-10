# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:33:44 2022

@author: RZ
"""

import numpy as np 
import warnings
from numba import njit, int32

#suppress warnings
warnings.filterwarnings('ignore')

@njit((int32,int32),fastmath=True)
def randX(a, b): 
    P = np.random.rand(a, b) 
    return P 

@njit(fastmath=True)
def randn(a, b): 
    w = randX(a, b) * 2.0 - 1.0 
    return w 

@njit(fastmath=True)
def normRows(a): 
    P = a.copy() 
    rows, columns = a.shape 
    for x in range(0, rows): 
        sumSq = 0 
        for y in range(0, columns): 
            v = P[x, y] 
            sumSq += v ** 2.0 
        len = np.sqrt(sumSq) 
        for y in range(0, columns): 
            P[x, y] = P[x, y] / len 
    return P 

@njit(fastmath=True)
def sumsqr(a): 
    rows, columns = a.shape 
    sumSq = 0 
    for x in range(0, rows): 
        for y in range(0, columns): 
            v = a[x, y] 
            sumSq += v ** 2.0 
    return sumSq 

@njit((int32,int32),fastmath=True)
def nwtan(s, p): 
    magw = 0.7 * s ** (1.0 / p) 
    w = magw * normRows(randn(s,p)) 
    b = magw * randn(s,1) 
    rng = np.zeros((1, p)) 
    rng = rng + 2.0 
    mid = np.zeros((p, 1)) 
    w = 2.0 * w / np.dot(np.ones((s,1)), rng) 
    b = b - np.dot(w, mid) 
    return w, b 

@njit(fastmath=True)
def tansig(n, b): 
    n = n + b 
    a = 2.0 / (1.0 + np.exp(-2.0 * n)) - 1.0 
    rows, columns = a.shape 
    for x in range(0, rows): 
        for y in range(0, columns): 
            v = a[x, y] 
            if np.abs(v) == np.inf: 
                a[x, y] = np.sign(n[x, y]) 
    return a 

@njit(fastmath=True)
def deltatan(a, d, w=None):
    if w is None:
        return (1.0 - (a * a)) * d
    else:
        return (1.0 - (a * a)) * np.dot(w.T, d)

@njit(fastmath=True)
def learnbp(p, d, lr): 
    x = lr * d 
    dw = np.dot(x, np.transpose(p)) 
    Q = p.shape[1] 
    db = np.dot(x, np.ones((Q, 1))) 
    return dw, db
