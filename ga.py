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
from numpy import zeros
from numpy import ones
from aux import *
from oldtrip import *
from numpy.random import normal, uniform

popsize = 49
select, iters = round(math.sqrt(popsize)), round(5000 / popsize)


def select_fittest(distr):
    """Randomly select an element according to the given distribution."""
    i = 0
    v = uniform()
    while i < len(distr) and v > distr[i]:
        i += 1
    return i


def crossover(a, b):
    """Sibling is an average of parents."""
    c = []
    for i in range(0, len(a)):
        c.append(((a[i][0] + b[i][0]) / 2, (a[i][1] + b[i][1]) / 2))
    return c


def ga_distortion(trip, testset_xy):
    # Generate initial population.
    new_probings = []
    for i in range(0, popsize):
        new_probings.append(trip.xys.copy())

    for it in range(0, iters):
        probings = new_probings.copy()

        # Mutate population and calculate fitness information.
        fit = []
        for i in range(0, popsize):
            v, probings[i] = trip.fitness(probings[i], testset_xy, random_distortion)
            fit.append(v)

        # Define a probability distribution to select the fittest ones.
        minfit = min(fit)
        for i in range(0, popsize):
            fit[i] -= minfit
        sumfit = sum(fit)
        distr = []
        for i in range(0, popsize):
            fit[i] /= sumfit
        for i in range(1, popsize):
            distr.append(sum(fit[:i]))
        distr.append(1)

        # Select the fittest ones.
        selected = set()
        while len(selected) < select:
            selected.add(select_fittest(distr))

        # Crossover
        for i in selected:
            for j in selected:
                new_probings.append(crossover(probings[i], probings[j]))

    # Adopt best individual.
    vmin = 999999
    for i in range(0, popsize):
        v, _ = trip.fitness(probings[i], testset_xy)
        if v < vmin:
            best = i
            vmin = v
    trip.xys = probings[best].copy()
    return vmin