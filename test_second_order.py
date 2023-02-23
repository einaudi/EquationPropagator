# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from EquationPropagator import EquationPropagator


ep = EquationPropagator(order=2)

gamma = 0.3
omega = 50
T = 20
dt = 1e-3

f1 = lambda x1, x2, t: -gamma*x1 - omega*x2
f2 = lambda x1, x2, t: x1

q0 = [1, 0]
fs = [f1, f2]

ep.set_equation_functions(fs)
ep.set_initial_conditions(q0)
ts, xs = ep.propagate(dt, T)

plt.plot(ts, xs[1])
plt.show()