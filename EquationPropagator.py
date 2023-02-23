# -*- coding: utf-8 -*-

import numpy as np


class EquationPropagator():

    def __init__(self, order=1):

        if order == 1:
            self._propagator = _PropagatorFirstOrder()
        elif order == 2:
            self._propagator = _PropagatorSecondOrder()
        else:
            print('Unsupported equation order!')
            quit()

        self.order = 1
        self._eq_funcs = []
        self._q0 = []

    def set_equation_functions(self, funcs):

        self._eq_funcs = funcs

    def set_initial_conditions(self, q0):

        self._q0 = q0

    def propagate(self, dt, T):

        ts, qs = self._propagator.propagate(
            self._eq_funcs,
            self._q0,
            dt,
            T
        )

        return ts, qs


class _PropagatorFirstOrder():

    def __init__(self):

        pass

    def propagate(self, func, q0, dt, T):

        N = int(T/dt)

        ts = [0]
        qs = [q0]

        for _ in range(N):

            k1 = dt*func(qs[-1], ts[-1])
            k2 = dt*func(qs[-1]+0.5*k1, ts[-1]+0.5*dt)
            k3 = dt*func(qs[-1]+0.5*k2, ts[-1]+0.5*dt)
            k4 = dt*func(qs[-1]+k3, ts[-1]+dt)

            dq = (k1 + 2*k2 + 2*k3 + k4)/6

            qs.append(qs[-1] + dq)
            ts.append(ts[-1] + dt)

        return ts, qs

class _PropagatorSecondOrder():

    def __init__(self):

        pass
    
    def propagate(self, funcs, q0, dt, T):

        f1 = funcs[0]
        f2 = funcs[1]

        N = int(T/dt)

        ts = [0]
        q1s = [q0[0]]
        q2s = [q0[1]]

        for _ in range(N):

            k11 = dt*f1(q1s[-1], q2s[-1], ts[-1])
            k21 = dt*f2(q1s[-1], q2s[-1], ts[-1])

            k12 = dt*f1(q1s[-1]+0.5*k11, q2s[-1]+0.5*k21, ts[-1]+0.5*dt)
            k22 = dt*f2(q1s[-1]+0.5*k11, q2s[-1]+0.5*k21, ts[-1]+0.5*dt)
            
            k13 = dt*f1(q1s[-1]+0.5*k12, q2s[-1]+0.5*k22, ts[-1]+0.5*dt)
            k23 = dt*f2(q1s[-1]+0.5*k12, q2s[-1]+0.5*k22, ts[-1]+0.5*dt)
            
            k14 = dt*f1(q1s[-1]+k13, q2s[-1]+k23, ts[-1]+dt)
            k24 = dt*f2(q1s[-1]+k13, q2s[-1]+k23, ts[-1]+dt)

            dq1 = (k11 + 2*k12 + 2*k13 + k14)/6
            dq2 = (k21 + 2*k22 + 2*k23 + k24)/6

            q1s.append(q1s[-1] + dq1)
            q2s.append(q2s[-1] + dq2)
            ts.append(ts[-1] + dt)

        return ts, [q1s, q2s]


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # ep = EquationPropagator()

    # f = lambda x, t: -x
    # q0 = 2
    # T = 5
    # dt = 1e-3

    # ep.set_equation_functions(f)
    # ep.set_initial_conditions(q0)
    # ts, qs = ep.propagate(dt, T)

    # plt.plot(ts, qs)
    # plt.show()

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