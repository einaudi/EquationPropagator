# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from EquationPropagator import EquationPropagator


ep = EquationPropagator(order=2)

gamma = 0.3
omega = 50
T = 20
dt = 1e-3

f1 = lambda x1, x2, t: -gamma*x1 - omega*x2 + 0.5*np.cos(np.sqrt(omega)*t)
f2 = lambda x1, x2, t: x1

q0 = [0, 1]
fs = [f1, f2]

ep.set_equation_functions(fs)
ep.set_initial_conditions(q0)
ts, xs = ep.propagate(dt, T)

plt.plot(ts, xs[1])
plt.show()