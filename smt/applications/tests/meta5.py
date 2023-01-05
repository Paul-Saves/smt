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


def f_neu(x1):

    return 2 * x1


def f1(x1, x2):
    return f_neu(x1) + x2**2


def f2(x1, x2, x3):
    return f_neu(x1) + (x2**2) + 0.3 * x3


def f3(x1, x2, x3, x4):
    return f_neu(x1) + (x2**2) + 0.3 * x3 - 0.1 * x4**3


def f(X):
    y = []
    for x in X:
        if x[0] == 0:
            y.append(f1(x[1], x[2]))
        elif x[0] == 1:
            y.append(f2(x[1], x[2], x[3]))
        elif x[0] == 2:
            y.append(f3(x[1], x[2], x[3], x[4]))
    return np.array(y)


print(f(np.atleast_2d([0, -1, 2])))
print(f(np.atleast_2d([1, -1, 2, 1])))
print(f(np.atleast_2d([2, -1, 2, 1, -2])))

xdoe1 = np.zeros((18, 3))

u = []
v = []
for (xi, yi) in itertools.product(np.linspace(0, 5, 6), np.linspace(-5, -2, 3)):
    u.append(xi)
    v.append(yi)
x_cont = np.concatenate(
    (np.asarray(v).reshape(-1, 1), np.asarray(u).reshape(-1, 1)), axis=1
)

xdoe1[:, 1:] = x_cont
ydoe1 = f(xdoe1)

xdoe1 = np.zeros((18, 5))
xdoe1[:, 1:3] = x_cont


xdoe2 = np.zeros((108, 4))
u = []
v = []
w = []
for (xi, yi, zi) in itertools.product(
    np.linspace(0, 5, 6), np.linspace(0, 5, 6), np.linspace(-5, -2, 3)
):
    u.append(xi)
    v.append(yi)
    w.append(zi)
x_cont = np.concatenate(
    (
        np.asarray(w).reshape(-1, 1),
        np.asarray(v).reshape(-1, 1),
        np.asarray(u).reshape(-1, 1),
    ),
    axis=1,
)

xdoe2[:, 0] = np.ones(108)
xdoe2[:, 1:4] = x_cont
ydoe2 = f(xdoe2)

xdoe2 = np.zeros((108, 5))
xdoe2[:, 0] = np.ones(108)
xdoe2[:, 1:4] = x_cont


xdoe3 = np.zeros((81, 5))

x_cont = np.linspace(0, 5, 6)
u = []
v = []
w = []
x = []

for (xi, yi, zi, ai) in itertools.product(
    np.linspace(0, 5, 3),
    np.linspace(0, 5, 3),
    np.linspace(0, 5, 3),
    np.linspace(-5, -2, 3),
):
    u.append(xi)
    v.append(yi)
    w.append(zi)
    x.append(ai)
x_cont = np.concatenate(
    (
        np.asarray(x).reshape(-1, 1),
        np.asarray(w).reshape(-1, 1),
        np.asarray(v).reshape(-1, 1),
        np.asarray(u).reshape(-1, 1),
    ),
    axis=1,
)
xdoe3[:, 0] = 2 * np.ones(81)
xdoe3[:, 1:5] = x_cont
ydoe3 = f(xdoe3)

Xt = np.concatenate((xdoe1, xdoe2, xdoe3), axis=0)
Yt = np.concatenate((ydoe1, ydoe2, ydoe3), axis=0)

xlimits = [[0, 2], [-5, -2], [0.0, 5.0], [0.0, 5.0], [0.0, 5.0]]
xtypes = [ORD, FLOAT, ORD, ORD, ORD]
xroles = [META, NEUTRAL, DECREED, DECREED, DECREED]
# Surrogate
sm = MixedIntegerSurrogateModel(
    categorical_kernel=HOMO_HSPHERE_KERNEL,
    xtypes=xtypes,
    xlimits=xlimits,
    xroles=xroles,
    surrogate=KRG(theta0=[1e-2], n_start=5, corr="abs_exp"),
)
sm.set_training_values(Xt, Yt)
a = time.time()
sm.train()
b = time.time()
t = b - a
print(t)

print(sm._surrogate.optimal_theta)
