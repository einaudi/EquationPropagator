# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from EquationPropagator import EquationPropagator


ep = EquationPropagator()

f = lambda x, t: -x
q0 = 2
T = 5
dt = 1e-3

ep.set_equation_functions(f)
ep.set_initial_conditions(q0)
ts, qs = ep.propagate(dt, T)

plt.plot(ts, qs)
plt.show()