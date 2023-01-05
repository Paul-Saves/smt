#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:48:01 2021

@author: psaves
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from smt.surrogate_models import (
    KRG,
    FLOAT,
    ORD,
    ENUM,
    NEUTRAL,
    META,
    DECREED,
    HOMO_HSPHERE_KERNEL,
    EXP_HOMO_HSPHERE_KERNEL,
    GOWER_KERNEL,
)
from smt.applications.mixed_integer import (
    MixedIntegerSurrogateModel,
)
import time


def f_neu(x1, x2, x3, x4):
    if x4 == 0:
        return 2 * x1 + x2 - 0.5 * x3
    if x4 == 1:
        return -x1 + 2 * x2 - 0.5 * x3
    if x4 == 2:
        return -x1 + x2 + 0.5 * x3


def f1(x1, x2, x3, x4, x5):
    return f_neu(x1, x2, x3, x4) + x5**2


def f2(x1, x2, x3, x4, x5, x6):
    return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6


def f3(x1, x2, x3, x4, x5, x6, x7):
    return f_neu(x1, x2, x3, x4) + (x5**2) + 0.3 * x6 - 0.1 * x7**3


def f(X):
    y = []
    for x in X:
        if x[0] == 1:
            y.append(f1(x[1], x[2], x[3], x[4], x[5]))
        elif x[0] == 2:
            y.append(f2(x[1], x[2], x[3], x[4], x[5], x[6]))
        elif x[0] == 3:
            y.append(f3(x[1], x[2], x[3], x[4], x[5], x[6], x[7]))
    return np.array(y)


print(f(np.atleast_2d([1, -1, -2, 8, 0, 2])))
print(f(np.atleast_2d([2, -1, -2, 16, 1, 2, 1])))
print(f(np.atleast_2d([3, -1, -2, 32, 2, 2, 1, -2])))

xdoe1 = np.zeros((48, 6))

u = []
v = []
w = []
x = []
y = []
for (xi, yi, zi, ai, bi) in itertools.product(
    np.linspace(0, 5, 2),
    np.linspace(-5, -2, 2),
    np.linspace(-5, -1, 2),
    np.array([32, 128]),
    np.array([0, 1, 2]),
):
    u.append(xi)

    v.append(bi)
    w.append(ai)
    x.append(zi)
    y.append(yi)
x_cont = np.concatenate(
    (
        np.asarray(y).reshape(-1, 1),
        np.asarray(x).reshape(-1, 1),
        np.asarray(w).reshape(-1, 1),
        np.asarray(v).reshape(-1, 1),
        np.asarray(u).reshape(-1, 1),
    ),
    axis=1,
)
xdoe1[:, 0] = np.ones(48)
xdoe1[:, 1:] = x_cont
ydoe1 = f(xdoe1)

xdoe1 = np.zeros((48, 8))
xdoe1[:, 0] = np.ones(48)
xdoe1[:, 1:6] = x_cont


xdoe2 = np.zeros((96, 7))
u = []
v = []
w = []
x = []
y = []
z = []
for (xi, yi, zi, ai, bi, ci) in itertools.product(
    np.linspace(0, 5, 2),
    np.linspace(0, 5, 2),
    np.linspace(-5, -2, 2),
    np.linspace(-5, -1, 2),
    np.array([32, 128]),
    np.array([0, 1, 2]),
):
    u.append(xi)
    v.append(yi)

    w.append(ci)
    x.append(bi)
    y.append(ai)
    z.append(zi)
x_cont = np.concatenate(
    (
        np.asarray(z).reshape(-1, 1),
        np.asarray(y).reshape(-1, 1),
        np.asarray(x).reshape(-1, 1),
        np.asarray(w).reshape(-1, 1),
        np.asarray(v).reshape(-1, 1),
        np.asarray(u).reshape(-1, 1),
    ),
    axis=1,
)

xdoe2[:, 0] = 2 * np.ones(96)
xdoe2[:, 1:7] = x_cont
ydoe2 = f(xdoe2)

xdoe2 = np.zeros((96, 8))
xdoe2[:, 0] = 2 * np.ones(96)
xdoe2[:, 1:7] = x_cont


xdoe3 = np.zeros((192, 8))

u = []
v = []
w = []
x = []
y = []
z = []
zz = []
for (xi, yi, zi, ai, bi, ci, di) in itertools.product(
    np.linspace(0, 5, 2),
    np.linspace(0, 5, 2),
    np.linspace(0, 5, 2),
    np.linspace(-5, -2, 2),
    np.linspace(-5, -1, 2),
    np.array([32, 128]),
    np.array([0, 1, 2]),
):
    u.append(xi)
    v.append(yi)
    w.append(zi)

    x.append(di)
    y.append(ci)
    z.append(bi)
    zz.append(ai)
x_cont = np.concatenate(
    (
        np.asarray(zz).reshape(-1, 1),
        np.asarray(z).reshape(-1, 1),
        np.asarray(y).reshape(-1, 1),
        np.asarray(x).reshape(-1, 1),
        np.asarray(w).reshape(-1, 1),
        np.asarray(v).reshape(-1, 1),
        np.asarray(u).reshape(-1, 1),
    ),
    axis=1,
)
xdoe3[:, 0] = 3 * np.ones(192)
xdoe3[:, 1:] = x_cont
ydoe3 = f(xdoe3)

Xt = np.concatenate((xdoe1, xdoe2, xdoe3), axis=0)
Yt = np.concatenate((ydoe1, ydoe2, ydoe3), axis=0)

xlimits = [
    [1, 3],  # meta ord
    [-5, -2],
    [-5, -1],
    ["8", "16", "32", "64", "128", "256"],
    ["ReLU", "SELU", "ISRLU"],
    [0.0, 5.0],  # decreed m=1
    [0.0, 5.0],  # decreed m=2
    [0.0, 5.0],  # decreed m=3
]
xtypes = [ORD, FLOAT, FLOAT, ORD, (ENUM, 3), ORD, ORD, ORD]
xroles = [META, NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, DECREED, DECREED, DECREED]

# Surrogate
sm = MixedIntegerSurrogateModel(
    categorical_kernel=HOMO_HSPHERE_KERNEL,
    xtypes=xtypes,
    xroles=xroles,
    xlimits=xlimits,
    surrogate=KRG(theta0=[1e-2], n_start=5, corr="abs_exp", nugget=1e-13),
)
sm.set_training_values(Xt, Yt)
print("training set")
a = time.time()
sm.train()
b = time.time()
t = b - a
print(t)

print(sm._surrogate.optimal_theta)
