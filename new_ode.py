import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import sys


def ode(X, t, params):
    x = X[0]
    v = X[1]
    y = X[2]
    w = X[3]
    alpha_1, alpha_2, alpha_3 = params

    dxdt = alpha_1 * (v - x) - 2 * alpha_3 * (2 * x - y)
    dvdt = 1.41 * (x - v) - alpha_3 * (2 * x - y)
    dydt = alpha_1 * (w - y) + 2 * alpha_3 * (2 * y - x)
    dwdt = 1.41 * (y - w) + alpha_3 * (2 * y - x)
    return [dxdt, dvdt, dydt, dwdt]

X0 = [0.5, 0, 0.5, 0]
t = np.linspace(0, 80, 600)

# alpha_1 = 0
# mu = 2
# alpha_3 = 1 / 1.41
# alpha_2 = 1.9 * alpha_3
# if alpha_2 > (mu * alpha_3):
#     print "invalide alpha_2 and alpha_3"
#     sys.exit()

min_x = 10000
min_y = 10000
min_v = 10000
min_w = 10000

alpha_3 = 2
alpha_2 = 1.9 * alpha_3
alpha_1 = 0
params = [alpha_1, alpha_2, alpha_3]
sol = odeint(ode, X0, t, args=(params,))

# for alpha_1 in np.arange(-15, 15, 1):
#     for alpha_3 in np.arange(-15, 15, 1):
#         alpha_2 = 1.9 * alpha_3
#         params = [alpha_1, alpha_2, alpha_3]
#         sol = odeint(ode, X0, t, args=(params,))
#         x = sol[:, 0][-1]
#         v = sol[:, 1][-1]
#         y = sol[:, 2][-1]
#         w = sol[:, 3][-1]

#         if abs(x) < min_x and abs(y) < min_y:
#             min_x = abs(x)
#             min_y = abs(y)
#             print "min x set to be: " + str(x) + " when alpha_1 and alpha_3 are: " + str(alpha_1) + ", " + str(alpha_3)
#             print "min y set to be: " + str(y) + " when alpha_1 and alpha_3 are: " + str(alpha_1) + ", " + str(alpha_3)

#         if abs(v) < min_v and abs(w) < min_w:
#             min_v = abs(v)
#             min_w = abs(w)
#             print "min v set to be: " + str(v) + " when alpha_1 and alpha_3 are: " + str(alpha_1) + ", " + str(alpha_3)
#             print "min w set to be: " + str(w) + " when alpha_1 and alpha_3 are: " + str(alpha_1) + ", " + str(alpha_3)

x = sol[:, 0]
v = sol[:, 1]
y = sol[:, 2]
w = sol[:, 3]

plt.plot(t, x, t, y)
plt.xlabel('t')
plt.legend(('x', 'y'))

plt.figure()
plt.plot(x, y)
plt.plot(x[0], y[0], 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
