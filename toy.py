# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def RGB_to_hex(RGB):
    RGB = [int(x) for x in RGB]
    return "#" + "".join([
        "0{0:x}".format(v) if v < 16 else
        "{0:x}".format(v) for v in RGB])


def hex_to_RGB(hex):
    return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]


def linear_gradient(start_hex, finish_hex='#FFFFFF', n=10):
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    RGB_list = [RGB_to_hex(s)]
    for t in range(1, n):
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)]
        RGB_list.append(RGB_to_hex(curr_vector))

    return RGB_list


update_rule = 'new'
grid_length = 1
x_list = []
y_list = []
x = 0.5
y = 0.5

v_x = 0
v_y = 0

# # Negative Momentum Setting
# h = 5 * 1e-1  # Step Size
# mu = - 4.8  # Friction Coefficient
# Q = 0  # Charge

# Positive Momentum Setting
h = 1e-1  # Step Size
mu = 5  # Friction Coefficient
Q = (1 / mu) + mu  # Charge

beta = 1 / (1 + mu * h)  # Momentum Factor
eta = (h ** 2) * beta  # Learning Rate
alpha = Q * h / (1 + mu * h)

for t in range(11500):
    z = x * y
    # grad_x = y
    # grad_y = -x
    # if update_rule == 'sgd':
    #     x -= h * grad_x
    #     y -= h * grad_y
    if update_rule == 'new':
        # import ipdb; ipdb.set_trace()
        if t == 0 or t == 1:
            x_diff = 0
            y_diff = 0
        else:
            x_diff = x - x_list[-2]
            y_diff = y - y_list[-2]

        # Alternating Gradients
        # x = x + beta * x_diff - eta * y - alpha * y_diff
        # y = y + beta * y_diff + eta * x + alpha * x_diff

        # Simultaneous Gradients
        x_up = x + beta * x_diff - eta * y - alpha * y_diff
        y_up = y + beta * y_diff + eta * x + alpha * x_diff

        x = x_up
        y = y_up

    x_list.append(x)
    y_list.append(y)
print(x)
print(y)

fig1 = plt.scatter(
    x_list, y_list, color=linear_gradient('#445ACC', '#b30000', t + 1), s=1)
my_mesh = np.array(
    np.meshgrid(
        np.linspace(-grid_length, grid_length, 20),
        np.linspace(-grid_length, grid_length, 20))).reshape((2, -1)).T
for i in range(len(my_mesh)):
    if 'consensus' in update_rule:
        grad_x = my_mesh[i][1] + 0.5 * my_mesh[i][0]
        grad_y = my_mesh[i][0] - 0.5 * my_mesh[i][1]
    else:
        grad_x = my_mesh[i][1]
        grad_y = my_mesh[i][0]
    plt.arrow(my_mesh[i][0], my_mesh[i][1],
              -grad_x / 15.0, grad_y / 15.0,
              head_width=0.02, head_length=0.02,
              color='gray')

plt.xlim([-grid_length, grid_length])
plt.ylim([-grid_length, grid_length])
plt.show()
plt.draw()
