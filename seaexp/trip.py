import matplotlib.pyplot as plt
from dataclasses import dataclass, replace, field
from functools import lru_cache
from typing import Union

import numpy as np
from scipy.spatial import distance_matrix

from seaexp.tsp import multistart_localsearch


@dataclass
class Trip:
    """Sequence of n-D points visited or to visit within a budget

    Every trip start/end at the provided depot.

    Usage:
    >>> trip = Trip([(1, 1), (2, 1)])
    >>> trip.points
    array([[0, 0],
           [1, 1],
           [2, 1],
           [0, 0]])

    """
    offshore: Union[list, np.ndarray] = field(default_factory=list)
    depot: Union[tuple, np.ndarray] = field(default_factory=lambda: np.array([0, 0]))
    budget: float = 100
    cost_per_distance_unit: float = 1
    cost_per_probing: float = 1

    def __post_init__(self):
        self.d = len(self.depot)
        if len(self.offshore) > 0:
            if not isinstance(self.offshore[0], np.ndarray):
                self.offshore = np.array(self.offshore)
        else:
            self.offshore = np.empty((0, self.d))
        if not isinstance(self.depot, np.ndarray):
            self.depot = np.array(self.depot)
        self.points = np.vstack([self.depot, self.offshore, self.depot])
        self.n = len(self.points)
        if self.cost > self.budget:
            # TODO: See '<<' code regarding BudgetExhausted.
            raise BudgetExhausted(f"Budget exhausted: {round(self.cost, 6)}... > {self.budget}")

    @property
    @lru_cache
    def cost(self) -> float:
        """Total trip cost.

        Usage:
        >>> trip = Trip([(1, 1), (2, 1), (2, 0)], cost_per_probing=2)
        >>> round(trip.cost, 4) # 1.4142 + 1 + 1 + 2 + 2*3
        11.4142

        """
        prob = self.cost_per_probing * len(self.offshore)
        dist = self.cost_per_distance_unit * sum(
            [self.distances[i, i + 1] for i in range(self.n - 2)] + [self.distances[-1, 0]])
        return prob + dist

    def __lshift__(self, next_point: Union[tuple, list, np.ndarray]):
        """Add n_D points to the trip.

        Usage:
        >>> trip = Trip(offshore=[(0.20, 0.10), (0.21, 0.13)], budget=4)
        >>> trip.points
        array([[0.  , 0.  ],
               [0.2 , 0.1 ],
               [0.21, 0.13],
               [0.  , 0.  ]])
        >>> round(trip.cost, 6)
        2.502211

        >>> trip <<= (0.23, 0.2)
        >>> trip.points
        array([[0.  , 0.  ],
               [0.2 , 0.1 ],
               [0.21, 0.13],
               [0.23, 0.2 ],
               [0.  , 0.  ]])

        >>> round(trip.cost, 6)
        3.632826

        >>> try:
        ...     trip <<= [(0.4, 0.2), (0.38, 0.25)]
        ... except BudgetExhausted as e:
        ...     print(e)
        Budget exhausted: 4.945244... > 4

        Return an extended new trip including the new point, but still having the depot as ending point.
        Raise BudgetExhausted exception if beyond the budget.
        """
        # TODO: Regarding BudgetExhausted.
        #       Is it computationally worth to precalculate exhaustion here instead of checking it at the returned Trip?

        # Add a batch of points.
        if isinstance(next_point, list) or isinstance(next_point, np.ndarray) and len(next_point.shape) == 2:
            trip = self
            for point in next_point:
                trip <<= point
            return trip

        # Add a single point.
        # new_cost = self.cost  # Previous total.
        # new_cost -= self.distances[-1, 0]  # Discount distance from last point to depot.
        # new_cost += self.cost_per_probing + np.linalg.norm(self.offshore[-1] - next_point)  # Add new point costs.
        # new_cost += np.linalg.norm(next_point - self.depot)  # Add updated return to depot distance.
        return replace(self, offshore=np.vstack([self.offshore, next_point]))

    def __iter__(self):
        return iter(self.points)

    def __hash__(self):
        return id(self)

    @property
    @lru_cache
    def distances(self):
        """Graph of costs.

        Usage:
        >>> trip = Trip(offshore=[(0.20, 0.10), (0.21, 0.13)], budget=4)
        >>> trip.distances
        array([[0.        , 0.2236068 , 0.24698178],
               [0.2236068 , 0.        , 0.03162278],
               [0.24698178, 0.03162278, 0.        ]])

        """
        return distance_matrix(self.points[:-1], self.points[:-1])

    def shorter(self, rnd=None):
        """heuristic TSP

        Usage:
        >>> trip = Trip(offshore=[(0, 1), (1, 1), (2, 0), (1, 2), (1, 0)])
        >>> trip.show()
        (0 0) (0 1) (1 1) (2 0) (1 2) (1 0) (0 0)

        >>> round(trip.cost, 5)
        13.65028
        
        >>> trip = trip.shorter()
        >>> round(trip.cost, 5)
        11.82843
        
        >>> trip = trip.shorter()
        >>> round(trip.cost, 5)
        11.82843

        """
        if rnd is None:
            rnd = np.random.default_rng(0)
        # Up to 4, all solutions have the same cost: depot -> a -> b -> depot == depot -> b -> a -> depot.
        if self.n <= 4:
            return self
        available_budget = self.cost - (self.n - 2) * self.cost_per_probing
        sol_, _ = multistart_localsearch(100, self.n - 1, self.distances, cutoff=available_budget, rnd=rnd)

        # Consider depot as the start and the point before it as the end (may increase the cost obtained by solver):
        #   a -> b -> c -> depot -> d -> e      becomes     depot -> d -> e -> a -> b -> c.
        idx = sol_.index(0)
        sol = sol_[idx:] + sol_[:idx]
        # TODO: See '<<' code regarding BudgetExhausted.
        offshore_idxs = np.array(sol[1:]) - 1
        return replace(self, offshore=self.offshore[offshore_idxs])

    @lru_cache
    def __str__(self):
        """
        Usage:
        >>> trip = Trip(offshore=[(0, 1), (1, 1), (2, 0), (1, 2), (1, 0)])
        >>> trip.show()
        (0 0) (0 1) (1 1) (2 0) (1 2) (1 0) (0 0)

        Returns
        -------

        """
        return " ".join(["(" + " ".join([str(round(x, 5)) for x in row]) + ")" for row in self.points])

    def show(self):  # -pragma: no cover
        print(str(self))

    def plot(self, xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), name=None, block=True):
        """
        Plot trip as n-D connected points

        Usage:
        >>> trip = Trip.fromrandom(6)
        >>> trip.plot()  # _doctest: +SKIP
        >>> trip.shorter().plot()  # _doctest: +SKIP

        Returns
        -------

        """
        # TODO: move this to Plotter class<<
        if 1 <= self.d <= 2:
            # plt.ion()
            # fig = plt.figure()
            pl, = plt.plot(*zip(*self.points), 'xb-')
            # fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)
            # pl.remove()
            plt.show()
        elif self.d == 3:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
            z = np.linspace(-2, 2, 100)
            r = z ** 2 + 1
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            ax.plot(*zip(*self.points), label='parametric curve')
            ax.legend()
            plt.show()
        else:
            raise Exception(f"Cannot handle {self.d} dimensions.")

    @classmethod
    def fromrandom(cls, size, dims=2, rnd=0, name=None):
        """A new Trip object containing 'n'-D random points.

        2D usage:
        >>> trip = Trip.fromrandom(4)
        >>> print(trip)  # Random walk across points.
        (0.0 0.0) (0.63696 0.26979) (0.04097 0.01653) (0.81327 0.91276) (0.60664 0.7295) (0.0 0.0)

        Parameters
        ----------
        size
            Number of points.
        dims
            Dimension of the space
        rnd
            Seed.
        name
            Identifier, e.g., for plots.
        """
        if isinstance(rnd, int):
            rnd = np.random.default_rng(rnd)
        return Trip(rnd.random((size, dims)), depot=np.zeros(dims))


class BudgetExhausted(Exception):
    """Exception if beyond the budget."""
