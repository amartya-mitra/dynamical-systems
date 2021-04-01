
import numpy as np
import matplotlib.pyplot as plt


def rgb_to_hex(rgb):
    rgb = [int(x) for x in rgb]
    return "#" + "".join([
        "0{0:x}".format(v) if v < 16 else
        "{0:x}".format(v) for v in rgb])


def hex_to_rgb(hex):
    return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]


def linear_gradient(start_hex, finish_hex='#FFFFFF', n=10):
    s = hex_to_rgb(start_hex)
    f = hex_to_rgb(finish_hex)
    rgb_list = [rgb_to_hex(s)]
    for t in range(1, n):
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)]
        rgb_list.append(rgb_to_hex(curr_vector))
    return rgb_list


update_rule = 'gd'
grid_length = 2
x_list = []
y_list = []
x = 0.8
y = 0.8
v_x = 0
v_y = 0
h = 1e-1
k = 0.25
rho = 2
q = -0.1

alpha = (1 - k * h ** 2 + rho * h) / (1 + rho * h)
beta = 1 / (1 + rho * h)
eta = (q * h) / (1 + rho * h)

nabla_x_y = 16

for t in range(50000):
    z = x * y

    if update_rule == 'gd':
        grad_x = y
        x = x - 0.0001 * grad_x
        grad_y = - x
        y = y - 0.0001 * grad_y

    if update_rule == 'new':
        if t == 0 or t == 1:
            x_diff = 0
            y_diff = 0
        else:
            x_diff = x - x_list[-2]
            y_diff = y - y_list[-2]
        x = x * alpha + beta * x_diff - nabla_x_y * x * y * y_diff * eta
        y = y * alpha + beta * y_diff + nabla_x_y * x * y * x_diff * eta
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
