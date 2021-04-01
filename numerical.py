import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

mu = 1.0


def solver(X, t, params):
    x = X[0]
    u = X[1]
    y = X[2]
    v = X[3]
    a, b = params

    dxdt = u
    # dudt = -a * u - b * y
    # dudt = -a * u - b * (y - 2 * x)
    dudt = -a * u - b * (3 * x**2 + 3 * y)
    dydt = v
    # dvdt = -a * v + b * x
    # dvdt = -a * v + b * (x + 2 * y)
    dvdt = -a * v + b * (-3 * y**2 + 3 * x)
    return [dxdt, dudt, dydt, dvdt]

X0 = [0.5, 0, 0.5, 0]
t = np.linspace(0, 40, 500)

min_x = 10000
min_y = 10000
for a in np.arange(-15, 15, 1):
    for b in np.arange(-15, 15, 1):
        params = [a, b]
        sol = odeint(solver, X0, t, args=(params,))

        x = sol[:, 0][-1]
        y = sol[:, 2][-1]

        if abs(x) < min_x and abs(y) < min_y:
            min_x = abs(x)
            min_y = abs(y)
            print "min x set to be: " + str(x) + " when a and b are: " + str(a) + ", " + str(b)
            print "min y set to be: " + str(y) + " when a and b are: " + str(a) + ", " + str(b)
