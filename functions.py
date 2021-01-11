#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
# from __future__ import division
# from __future__ import print_function
import math


# object for keeping instance data
# assuming this is constant
class INST:
    pass


inst = INST()
inst.t = 1  # time for probing
inst.s = 1  # traveling speed
inst.T = 100  # time limit for a route
inst.x0 = 0  # depot coordinates
inst.y0 = 0  # depot coordinates


# fake "real" data
class _data:
    def __init__(self):
        self.x, self.y, self.sigma, self.a = {}, {}, {}, {}
        self.x[1], self.y[1] = 0.35113, 0.070836
        self.a[1] = 100.
        self.sigma[1] = 0.1

        self.x[2], self.y[2] = 0.48802, 0.28421
        self.a[2] = 75.
        self.sigma[2] = 0.05

        self.x[3], self.y[3] = 0.032971, 0.20382
        self.a[3] = 50
        self.sigma[3] = 0.1

        self.x[4], self.y[4] = 0.52266, 0.19859
        self.a[4] = 75
        self.sigma[4] = 0.05

        self.x[5], self.y[5] = 0.24493, 0.7871
        self.a[5] = 25
        self.sigma[5] = 0.15

        self.x[6], self.y[6] = 0.19934, 0.50696
        self.a[6] = 5.3545
        self.sigma[6] = 0.49985

        self.x[7], self.y[7] = 0.63317, 0.34842
        self.a[7] = 80.985
        self.sigma[7] = 0.011509

        self.x[8], self.y[8] = 0.97123, 0.63791
        self.a[8] = 78.089
        self.sigma[8] = 0.12154

        self.x[9], self.y[9] = 0.9706, 0.27782
        self.a[9] = 67.355
        self.sigma[9] = 0.10809

        self.x[10], self.y[10] = 0.40523, 0.28157
        self.a[10] = 32.567
        self.sigma[10] = 0.029318


data = _data()


def _f(x, y, p):
    """parameters:
        - x: coordinate where to evaluate the function
        - y:
        - p: list of "centers" from 'data' to use
        """
    value = 0.
    for ii in p:
        value += data.a[ii] * math.exp(- ((x - data.x[ii]) / data.sigma[ii]) ** 2 / 2 - ((y - data.y[ii]) / data.sigma[ii]) ** 2 / 2)
    return value


# treino
def f0(x, y): return 0


def f1(x, y): return _f(x, y, [1])


def f2(x, y): return _f(x, y, [1, 2])


def f3(x, y): return _f(x, y, [1, 2, 3])


def f4(x, y): return _f(x, y, [1, 2, 3, 4])


def f5(x, y): return _f(x, y, [1, 2, 3, 4, 5])


# teste (??)
def f6(x, y): return _f(x, y, [6])


def f7(x, y): return _f(x, y, [6, 7])


def f8(x, y): return _f(x, y, [6, 7, 8])


def f9(x, y): return _f(x, y, [6, 7, 8, 9])


def f10(x, y): return _f(x, y, [6, 7, 8, 9, 10])


def plot(f, n, filename=None):
    import math
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import numpy as np

    # prepare data
    X = np.arange(0, 1.01, 1. / n)
    Y = np.arange(0, 1.01, 1. / n)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    Z = np.zeros((n + 1, n + 1))
    W = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            Z[i, j] = f(X[i, j], Y[i, j])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #    ax.set_zlabel('z(x,y)')
    ax.set_zlim(0, 100)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.05, antialiased=False)

    plt.show()


if __name__ == "__main__":
    # fs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    # fs = [f9, f10]
    fs = [f1, f5]
    for i in range(len(fs)):
        f = fs[i]
        plot(f, 100)
