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

from pswarm_py import pswarm
from numpy import zeros, ones
from aux import tuplefy, probe, flat, current_milli_time
from trip import Trip


def swarm_distortion(trip, testset_xy, available_time, maxf=4000, maxit=4000, size=100):
    """Optimize points in trip via PSO, evaluating over a test set. The available time is almost never respected."""
    def py_outf(it, leader, fx, x):
        """Function called at every iteration. It is for logging purposes, but also to stop when a criterion is matched."""
        elapsed = current_milli_time() - start
        # print(available_time - elapsed )
        return available_time - elapsed  # negative number = stop

    def py_objf(xs):
        """Function called at every iteration. It is for swarm fitness evaluation."""
        def var(x):
            """Evaluates fitness of a given particle of the swarm."""
            xys = tuplefy(x)  # Ps.: According to my tests with oldtrip.count(), trip methods don't need to be thread-safe here.
            v, _ = trip.fitness(xys, testset_xy)
            return v

        return [var(x) for x in xs]

    # Initial PSO settings.
    start = current_milli_time()
    x0 = flat(trip.xys)
    variabs = len(x0)
    problem = {'Variables': variabs, 'objf': py_objf, 'lb': zeros(variabs), 'ub': ones(variabs), 'x0': x0}
    # , 'A': [[-1.0 / sqrt(3), 1], [-1.0, sqrt(3)], [1.0, sqrt(3)]], 'b': [0, 0, 6] # <- Defaults
    options = {'maxf': maxf, 'maxit': maxit, 'social': 0.5, 'cognitial': 0.5, 'fweight': 0.4
        , 'iweight': 0.9, 'size': size, 'iprint': 10, 'tol': 1E-5, 'ddelta': 0.5, 'idelta': 2.0
        , 'outputfcn': py_outf, 'vectorized': 1}

    # PSO running.
    result = pswarm(problem, options)

    # PSO results.
    if result['ret'] == 0:  # zero means successful
        new_xys = tuplefy(result['x'])
        trip.xys = new_xys.copy()
    return result['f']