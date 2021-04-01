import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

mu = 1.0


def vanderpol(X, t, params):
    x = X[0]
    u = X[1]
    y = X[2]
    v = X[3]
    a, b = params

    dxdt = u
    # dudt = -a * u - b * y
    # a, b = 6, 1 / 8, -3
    dudt = -a * u - b * (y - 2 * x)
    # a, b = 9, -9/ 2, -3
    dydt = v
    # dvdt = -a * v + b * x
    dvdt = -a * v + b * (x + 2 * y)
    return [dxdt, dudt, dydt, dvdt]

X0 = [0.5, 0, 0.5, 0]
t = np.linspace(0, 80, 600)

a = 8
b = -3
params = [a, b]
sol = odeint(vanderpol, X0, t, args=(params,))

x = sol[:, 0]
u = sol[:, 1]
y = sol[:, 2]
v = sol[:, 3]

plt.plot(t, x, t, y)
plt.xlabel('t')
plt.legend(('x', 'y'))
# plt.savefig('images/vanderpol-1.png')

# phase portrait
plt.figure()
plt.plot(x, y)
plt.plot(x[0], y[0], 'ro')
plt.xlabel('x')
plt.ylabel('y')
# plt.savefig('images/vanderpol-2.png')

sign = (-y * u + x * v) / (abs(-y * u + x * v))
sign_x = (-y * u) / (abs(-y * u))
sign_y = (x * v) / (abs(x * v))
# sign_ = (-2 * x * u + 2 * y * v) / abs(-2 * x * u + 2 * y * v)
# sign__ = ((2 * x - y) * u + (x - 2 * y) * v) / abs((2 * x - y) * u + (x - 2 * y) * v)
# sign_x_v = (x * v) / abs(x * v)
plt.figure()
plt.scatter(t, sign, c="red")
# plt.scatter(t, sign_x, c="blue")
# plt.scatter(t, sign_y, c="green")
# plt.scatter(t, sign__, c="blue")
plt.xlabel('time')
plt.ylabel('sign')
plt.show()
