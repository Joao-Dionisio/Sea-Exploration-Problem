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
from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, RBF, Matern, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Esse import evita "ValueError: Unknown projection '3d'"
from tsp import solve_tsp, sequence  # exact
from tsp import multistart_localsearch  # heuristic
import time
import itertools
from numpy.random import normal, uniform, randint, seed
from atsp import scf, sequence1

ngrid = 100


def kernel_selector(xys, zs):
    # define limits of the hyperparameter space
    # bounds = [(0.00001, 0.0001), (0.0001, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1), (1, 10), (10, 100), (100, 1000), (1000, 10000), (10000, 100000)]
    # nu_bounds = [0.1, 0.5, 1, 1.5, 2, 2.5, 5, 20]
    bounds = [(0.00001, 0.001), (0.001, 0.1), (0.1, 10), (10, 1000), (1000, 100000)]
    nu_bounds = [0.5, 1.5, 2.5]

    # generate list of kernels to assess
    quads = [RationalQuadratic(length_scale_bounds=(a, b), alpha_bounds=(c, d)) for a, b in bounds for c, d in bounds]
    rbfs = [RBF(length_scale_bounds=(a, b)) for a, b in bounds]
    mtns = [Matern(length_scale_bounds=(a, b), nu=c) for a, b in bounds for c in nu_bounds]
    # ExpSineSquared(),
    # kernels = [RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))]
    kernels = quads + rbfs + mtns

    # find best kernel on k-fold CV
    min_error = 99999
    for kernel0 in kernels:
        kernel = kernel0 + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, copy_X_train=True)
        # model = GaussianProcessRegressor(kernel=kernel + WhiteKernel(noise_level_bounds=(1e-5, 1e-2)), n_restarts_optimizer=10, copy_X_train=True)
        err = -1 * cross_val_score(gpr, xys, zs, scoring='neg_mean_absolute_error', cv=5).mean()
        # print((type(kernel).__name__[:12] + '\t:\t' + str(err)).expandtabs(13))
        if err < min_error:
            min_error = err
            min_error_kernel = kernel
    return min_error_kernel


def train_data(l, f, rnd):
    return data(f, l, rnd)


def test_data(f):
    mesh, zs = [], []
    for i in range(101):
        for j in range(101):
            x, y = i / 100., j / 100.
            mesh.append((x, y))
            zs.append(f(x, y))
    # z = estimator(X, z, mesh)
    # final = 0
    # for i in range(len(mesh)):
    #     (x, y) = mesh[i]
    #     final += abs(f(x, y) - float(z[i]))
    return mesh, zs


def data(f, n2, rnd):
    xys, zs = [], []
    for x in [i / n2 for i in range(0, n2)]:
        for y in [i / n2 for i in range(0, n2)]:
            if rnd:
                x = uniform()
                y = uniform()
            xys.append((x, y))
            zs.append(f(x, y))
            # print("{}\t&{}\t&{}\\\\".format(x, y, zs[-1]))
    return xys, zs


def gp(xys, zs):
    from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic
    from sklearn.gaussian_process import GaussianProcessRegressor
    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, copy_X_train=True)
    gpr.fit(xys, zs)
    return gpr


def max_var(g):
    # Instead of max var, try to 
    xs = np.arange(0.0, 1.01, 1. / ngrid)
    ys = np.arange(0.0, 1.01, 1. / ngrid)
    xys = [(xs[i], ys[j]) for i in range(ngrid + 1) for j in range(ngrid + 1)]
    zs, stds = g.predict(xys, return_std=True)
    stdmax, zmax, xymax = max(zip(stds, zs, xys))
    return xymax, zmax, stdmax


def rnd(g):
    xys = [(uniform(), uniform())]
    zs, stds = g.predict(xys, return_std=True)
    std, z, xy = stds[0], zs[0], xys[0]
    return xy, z, std


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def complete_cost(cost, n):
    # each probing has cost 1
    return cost + n


def tsp(n, c, cutoff):
    model = scf(n, c)
    model.Params.OutputFlag = 0  # silent mode
    # model.Params.Cutoff = cutoff # Prints AttributeError: b"Unable to retrieve attribute 'X'"
    model.Params.TimeLimit = 60
    model.optimize()
    cost = model.ObjVal
    x, f = model.__data
    arcs = [(i - 1, j - 1) for (i, j) in x if x[i, j].X > .5]
    return cost, arcs


def plan_tour(xys, budget, exact, fixed=[]):
    """
    Calculates a tour over all given points.
    :param xys: list of points to visit
    :param budget: limit of points to visit + length limit
    :param exact: whether or not to run Gurobi
    :param fixed: segments of the trip already travelled (using this option avoids heuristic mode [multistart_localsearch])
    :return: solution, feasibility, cost of the tour including probings, whether the tour is optimal
    """
    n = len(xys)
    pos = xys
    c = {}
    for i in range(n):
        for j in range(n):  # non optimized distance calculations!
            c[i, j] = dist(*pos[i], *pos[j])
            c[j, i] = c[i, j]
    cost_is_optimal_or_timeout = False

    if n > 2:
        # heuristic
        if len(fixed) == 0:
            sol_, cost = multistart_localsearch(100, n, c, cutoff=budget - n)  # inst.T - (N) * inst.t)
            idx = sol_.index(0)
            sol = sol_[idx:] + sol_[:idx]
            cost = complete_cost(cost, n)

        # exact (or time out)
        if len(fixed) > 0 or cost > budget:
            if exact:
                print("# trying exact solution")
                cost, edges = solve_tsp(range(n), c, False, fixed, 60000) # not using cutoff because it appears to be slow
                # cost, edges = tsp(n, c, cutoff=budget - n)
                cost = complete_cost(cost, n)
                cost_is_optimal_or_timeout = True
                try:
                    sol = sequence(range(n), edges)
                except ValueError:
                    print("# time out")
                    cost = 99999999
                    sol = []
                    # keeps cost_is_optimal_or_timeout=True to avoid adding more points in add_while_possible()
            else:
                pass
                print('# NOT trying exact solution')
    elif n == 1:
        cost = 0
        sol = [0]
        cost_is_optimal_or_timeout = True
    else:
        cost = dist(*pos[0], *pos[1])
        cost = complete_cost(cost, n)
        sol = [0, 1]
        cost_is_optimal_or_timeout = True

    return sol, cost <= budget, cost, cost_is_optimal_or_timeout


def evalu_max(g, xys, zs):
    predzs = g.predict(xys, return_std=False)
    return max([abs(a - b) for a, b in zip(zs, predzs)])


def evalu_sum(g, xys, zs):
    predzs = g.predict(xys, return_std=False)
    # print([abs(a - b) for a, b in zip(zs, predzs)])
    return sum([abs(a - b) for a, b in zip(zs, predzs)])


def evalu_var(trip, xys):
    stds = trip.predict_stds(xys)
    return sum(stds)


def probe(f, xys):
    return [f(x, y) for x, y in xys]


def fmt(txt):
    return "{:6.6f}".format(txt)


def show_path(path, label=''):
    for a, b in path:
        print(fmt(a), '\t', fmt(b), '\t\t=path\t', label)


def p(plt, fig, f, n, zmin, zmax, filename=None):
    import math
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

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #    ax.set_zlabel('z(x,y)')
    ax.set_zlim(zmin, zmax)
    pl = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.05, antialiased=False)
    # plt.show()
    fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)
    if filename is not None:
        plt.savefig(filename)
    # time.sleep(0.2)
    pl.remove()


def plot_path(plt, fig, points, tour):
    plotx, ploty = zip(*[points[idx] for idx in tour])
    pl, = plt.plot(plotx, ploty, 'xb-')
    fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)
    # time.sleep(0.5)
    pl.remove()


def flat(l):
    return list(itertools.chain(*l))


def tuplefy(x):
    it = iter(x)
    x2 = list(zip(it, it))
    return [tuple(l) for l in x2]


def log(str):
    print('# out:\t' + str)


def no_distortion(a, b, c, d, e, f):
    return c, d


def median_distortion(a, b, c, d, e, f):
    """Distortion towards median line = shortening the path."""
    m, n = (a + e) / 2, (b + f) / 2
    # offset = 0.1 * (dist(a, b, c, d) + dist(c, d, e, f) - dist(a, b, e, f))
    p = 0.1
    return c + p * (m - c), d + p * (n - d)


def random_distortion(a, b, c, d, e, f):
    s = 0.05 * (dist(a, b, c, d) + dist(c, d, e, f)) / 2
    x, y = c + normal(scale=s), d + normal(scale=s)
    if x > 1: x = 1
    if y > 1: y = 1
    if x < 0: x = 0
    if y < 0: y = 0
    return x, y


def current_milli_time():
    return int(round(time.time() * 1000))


def fo(n):
    return round(1000 * n) / 1000.0
