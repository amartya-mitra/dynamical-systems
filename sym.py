from sympy import *

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')


def f(x, y, z):
    return x * y * z * x - y + x * (z - x * y)

u = f(x, y, z)
print('f(x,y,z) = ', u)
g = [u.diff(x), u.diff(y), u.diff(z)]
import ipdb; ipdb.set_trace()
print('grad f(x,y,z) =', g)