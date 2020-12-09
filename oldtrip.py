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
# from aux import *
# from numpy.random import normal
#
#
# class OldTrip:
#     """Handles model selection and tour
#
#     Keyword arguments:
#     depot -- tuple indicating starting point of the upcoming trip
#     first_xys -- list of points (tuples) already probed in previous trips
#     first_zs -- list of measurements in previous trips
#     """
#
#     def __init__(self, exact, depot, first_xys, first_zs, testsetxy, penal, debug=False):
#         self.exact = exact
#         self.should_penalize = penal
#         self.testsetxy = testsetxy
#         self.smallest_var = 9999999
#         self.debug = debug
#         self.log("init")
#         self.depot = depot
#         self.first_xys, self.first_zs = first_xys, first_zs
#         self.future_xys, self.future_zs = [], []
#         self.ismodel_cached = False
#         self.istour_cached = False
#         self.feasible = False
#         self.cost = 0
#         self.tour = []
#         self.previous_var = 3.1415123234
#         self.kernel = kernel_selector(self.first_xys, self.first_zs)
#         self.fit()
#         self.c = 0
#
#     def fit(self):
#         self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5, copy_X_train=True)
#         self.model.fit(self.first_xys + self.future_xys, self.first_zs + self.future_zs)
#
#     def log(self, str):
#         if self.debug: print('Trip: ', str, '.')
#
#     def refit(self, future_xys, future_zs):
#         """Update all points of the future trip, with provided probings."""
#         self.log('refit')
#         self.future_xys, self.future_zs = future_xys, future_zs
#         self.fit()
#         self.istour_cached = False
#         self.ismodel_cached = True
#
#     def refit2(self, future_xys):
#         """Update all points of the future trip, with simulated probings."""
#         self.log('refit2')
#         self.future_xys = future_xys
#         self.resimulate_probings()
#         self.refit(future_xys, self.future_zs)
#
#     def getmodel(self):
#         if not self.ismodel_cached: self.refit(self.future_xys, self.future_zs)
#         return self.model
#
#     def add(self, hxy, hz, hstd, txt):
#         self.log('add by ' + txt)
#         self.future_xys, self.future_zs = self.future_xys + [hxy], self.future_zs + [hz]
#         self.istour_cached = False
#         self.ismodel_cached = False
#
#     def add_maxvar_simulatedprobe(self):
#         self.add(*max_var(self.getmodel()), 'max var')
#
#     def add_rnd_simulatedprobe(self):
#         self.add(*rnd(self.getmodel()), 'rnd')
#
#     def calculate_tour(self, budget):
#         self.last_budget = budget
#         self.log('calc tour')
#         if not self.istour_cached:
#             self.previous_tour = self.tour
#             self.log('   tour not cached')
#             tour, self.feasible, cost = plan_tour([self.depot] + self.future_xys, budget, self.exact)
#             self.tour = tour if self.feasible else []
#             self.cost = cost  # if self.feasible else -1
#             self.istour_cached = True
#
#     def resimulate_probings(self):
#         self.log('resimulate')
#         self.future_zs = list(self.getmodel().predict(self.future_xys, return_std=False))
#
#     def isfeasible(self, budget):
#         self.last_budget = budget
#         self.calculate_tour(budget)
#         return self.feasible
#
#     def gettour(self, budget):
#         self.last_budget = budget
#         self.calculate_tour(budget)
#         return self.tour
#
#     def getcost(self, budget):
#         self.last_budget = budget
#         self.calculate_tour(budget)
#         return self.cost
#
#     def reset(self):
#         self.c = 0
#
#     def count(self):
#         tmp = self.c
#         time.sleep(random.uniform(0, 0.01))
#         self.c = tmp + 1
#
#     def print_count(self):
#         print(self.c)
#
#     def getvar(self):
#         return self.getvar_on(self.testsetxy)
#
#     def getvar_on(self, xys):
#         """Only this method (and indirectly issmallest_var()) update smallest_var."""
#         self.log('get tot var')
#         var = evalu_var(self.getmodel(), xys)
#         if var < self.smallest_var: self.smallest_var = var
#         return var
#
#     def geterr_on(self, xys, zs):
#         self.log('get err')
#         err = evalu_sum(self.getmodel(), xys, zs)
#         return err
#
#     def undo_last_simulatedprobing(self):
#         self.log('undo last probe')
#         self.future_xys.pop()
#         self.future_zs.pop()
#         self.tour = self.previous_tour
#         self.ismodel_cached = False
#         self.istour_cached = False
#
#     def store(self):
#         """Store current list of (future) points. The list can be restored later with restore()."""
#         self.stored_future_xys = self.future_xys.copy()
#
#     def store2(self):
#         """Store current list of (future) points. The list can be restored later with restore2()."""
#         self.stored_future_xys2 = self.future_xys.copy()
#
#     def restore(self):
#         self.future_xys = self.stored_future_xys.copy()
#         self.ismodel_cached = False
#         self.istour_cached = False
#         self.resimulate_probings()
#
#     def restore2(self):
#         self.future_xys = self.stored_future_xys2.copy()
#         self.ismodel_cached = False
#         self.istour_cached = False
#         self.resimulate_probings()
#
#     def distort(self, distortion_function):
#         """Apply a custom distortion function to all points, except depot and last. Call resimulate_probings() should be called after that."""
#         self.log('distort')
#         tour = self.tour
#         points = [self.depot] + self.future_xys
#         for ida, idb, idc in zip(tour, tour[1:], tour[2:]):
#             (a, b), (c, d), (e, f) = points[ida], points[idb], points[idc]
#             self.future_xys[idb - 1] = distortion_function(a, b, c, d, e, f)
#         self.ismodel_cached = False
#         self.istour_cached = False
#
#     def issmallest_var(self):
#         var = self.getvar()
#         res = var <= self.smallest_var and var != self.previous_var
#         self.previous_var = var
#         return res
#
#     def penalize(self):
#         return self.should_penalize and not self.isfeasible(self.last_budget)
#
#
# # eliminate a point at random to allow the insertion of a new one
# # idx = random.randrange(len(Nxy))
# # Nxy.pop(idx)
# # Nz.pop(idx)
# # tour.remove(idx)
# #
# # def fu(i):
# #     return i if i < idx else i - 1
# #
# #
# # tour = list(map(fu, tour))
#
#
# def ps_distortion(a, b, c, d, e, f):
#     pass
#
#
# def no_distortion(a, b, c, d, e, f):
#     return c, d
#
#
# def median_distortion(a, b, c, d, e, f):
#     """Distortion towards median line = shortening the path."""
#     m, n = (a + e) / 2, (b + f) / 2
#     # offset = 0.1 * (dist(a, b, c, d) + dist(c, d, e, f) - dist(a, b, e, f))
#     p = 0.1
#     return c + p * (m - c), d + p * (n - d)
#
#
# def random_distortion(a, b, c, d, e, f):
#     s = 0.01 * (dist(a, b, c, d) + dist(c, d, e, f)) / 2
#     return c + normal(scale=s), d + normal(scale=s)
