# #     This program is free software: you can redistribute it and/or modify
# #     it under the terms of the GNU Lesser General Public License as published by
# #     the Free Software Foundation, either version 3 of the License, or
# #     (at your option) any later version.
# #
# #     This program is distributed in the hope that it will be useful,
# #     but WITHOUT ANY WARRANTY; without even the implied warranty of
# #     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# #     GNU Lesser General Public License for more details.
# #
# #     You should have received a copy of the GNU Lesser General Public License
# #     along with this program.  If not, see <https://www.gnu.org/licenses/>.
# import random
# from mpl_toolkits.mplot3d import Axes3D  # Esse import evita "ValueError: Unknown projection '3d'"
# from functions import f5 as f
#
# random.seed(1)
#
# # choose test function and prepare preliminary data
# X, z = [], []
# N = 5
#
# # random points as initial set of known points
# for x in [i / N for i in range(1, N)]:
#     for y in [i / N for i in range(1, N)]:
#         x = random.random()
#         y = random.random()
#         X.append((x, y))
#         z.append(f(x, y))
#         print("{}\t&{}\t&{}\\\\".format(x, y, z[-1]))
#
# # test preliminary forecasting part
# from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic
# from sklearn.gaussian_process import GaussianProcessRegressor
#
# kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
# gpr.fit(X, z)
#
# # plot stddev landscape
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import numpy as np
#
# # grid
# n = 100
# X_ = np.arange(0, 1.01, 1. / n)
# Y_ = np.arange(0, 1.01, 1. / n)
# X_, Y_ = np.meshgrid(X_, Y_, indexing='ij')
# Z_ = np.zeros((n + 1, n + 1))
# W_ = np.zeros((n + 1, n + 1))
#
# # prediction
# for i in range(n + 1):
#     for j in range(n + 1):
#         val, val_std = gpr.predict([(X_[i, j], Y_[i, j])], return_std=True)
#         Z_[i, j] = val_std  # value to be plotted
#
# # plot
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax = fig.gca(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# #    ax.set_zlabel('z(x,y)')
# ax.set_zlim(0, 1.01)
# surf = ax.plot_surface(X_, Y_, Z_, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.05, antialiased=False)
#
# # plt.show()
# plt.savefig('output/stddev_00.pdf')
#
# # adding a new point in a precise location
# X.append((0.5, 0.1))
# z.append(f(0.5, 0.1))
# kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
# gpr.fit(X, z)
#
# # plot new stddev landscape
# for i in range(n + 1):
#     for j in range(n + 1):
#         val, val_std = gpr.predict([(X_[i, j], Y_[i, j])], return_std=True)
#         Z_[i, j] = val_std  # value to plotted
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax = fig.gca(projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlim(0, 1.01)
# surf = ax.plot_surface(X_, Y_, Z_, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.05, antialiased=False)
# plt.savefig('output/stddev_10.pdf')
