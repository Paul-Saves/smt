#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:48:01 2021

@author: psaves
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from smt.surrogate_models import KRG, FLOAT, ORD, ENUM,    HOMO_HSPHERE_KERNEL,EXP_HOMO_HSPHERE_KERNEL,GOWER_KERNEL
from smt.applications.mixed_integer import (
    MixedIntegerSurrogateModel,
)


def f1(x1):
    return x1**2


def f2(x1, x2):
    return (x1**2) + 0.3 * x2



def f3(x1, x2,x3):
    return (x1**2) + 0.3 * x2- 0.1*x3**3



def f(X):
    y = []
    for x in X:
        if x[0] == 0:
            y.append(f1(x[1]))
        elif x[0] == 1:
            y.append(f2(x[1], x[2]))
        elif x[0] == 2:
            y.append(f3(x[1], x[2],x[3]))           
    return np.array(y)


print(f(np.atleast_2d([0, 2])))
print(f(np.atleast_2d([1, 2, 1])))
print(f(np.atleast_2d([2, 2, 1,-2])))

xdoe1 = np.zeros((6, 2))
xdoe1[:, 1] = np.linspace(0, 5, 6)
ydoe1 = f(xdoe1)

xdoe1 = np.zeros((6, 4))
xdoe1[:, 1] = np.linspace(0, 5, 6)


xdoe2 = np.zeros((36, 3))
x_cont = np.linspace(0, 5, 6)
u = []
v = []
for (xi, yi) in itertools.product(x_cont, x_cont):
    u.append(xi)
    v.append(yi)
x_cont = np.concatenate(
    (np.asarray(v).reshape(-1, 1), np.asarray(u).reshape(-1, 1)), axis=1
)
xdoe2[:, 0] = np.ones(36)
xdoe2[:, 1:3] = x_cont
ydoe2 = f(xdoe2)

xdoe2 = np.zeros((36, 4))
xdoe2[:, 0] = np.ones(36)
xdoe2[:, 1:3] = x_cont


xdoe3 = np.zeros((216, 4))

x_cont = np.linspace(0, 5, 6)
u = []
v = []
w=  []
for (xi, yi,zi) in itertools.product(x_cont, x_cont,x_cont):
    u.append(xi)
    v.append(yi)
    w.append(zi)
x_cont = np.concatenate(
    (np.asarray(w).reshape(-1,1), np.asarray(v).reshape(-1, 1), np.asarray(u).reshape(-1, 1)), axis=1
)
xdoe3[:, 0] = 2*np.ones(216)
xdoe3[:, 1:4] = x_cont
ydoe3 = f(xdoe3)

Xt = np.concatenate((xdoe1, xdoe2,xdoe3), axis=0)
Yt = np.concatenate((ydoe1, ydoe2,ydoe3), axis=0)
xlimits = [["Blue", "Red","Green"], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
xtypes = [(ENUM, 3), FLOAT, FLOAT,FLOAT]

xlimits = [[0,2], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
xtypes = [ORD, FLOAT, FLOAT,FLOAT]

xlimits = [[0,2], [0.0, 5.0], [0.0, 5.0], [0.0, 5.0]]
xtypes = [ORD, ORD, ORD,ORD]
    
# Surrogate
sm = MixedIntegerSurrogateModel(
    categorical_kernel=HOMO_HSPHERE_KERNEL,
    xtypes=xtypes,
    xlimits=xlimits,
    surrogate=KRG(theta0=[1e-2], n_start=30, corr="abs_exp"),
)
sm.set_training_values(Xt, Yt)
sm.train()
print(sm._surrogate.optimal_theta)
