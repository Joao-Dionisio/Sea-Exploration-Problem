from dataclasses import dataclass, replace, field
from functools import lru_cache
from typing import Union

import numpy as np


@dataclass
class Trip:
    """Sequence of n-D points visited or to visit within a budget

    Every trip start/end at the provided depot."""
    offshore: list = field(default_factory=list)
    depot: Union[tuple, np.ndarray] = 0, 0
    budget: float = 10
    distance_cost: float = 1  # Per unit.
    probing_cost: float = 0  # Per unit.
    _cost = None

    def __post_init__(self):
        self.points = [self.depot] + self.offshore + [self.depot]
        if isinstance(self.depot, np.ndarray):
            self.depot = tuple(self.depot)

    @property
    def cost(self) -> float:
        """Total trip cost."""
        if self._cost is None:
            cost = len(self.offshore) * self.probing_cost
            for a, b in zip(self.points, self.points[1:]):
                cost += self.leg_cost(a, b)
            self._cost = cost

        return self._cost

    @lru_cache
    def leg_cost(self, a: Union[tuple, np.ndarray], b: Union[tuple, np.ndarray]) -> float:
        """Travel cost between two points"""
        if isinstance(a, tuple):
            a = np.array(a)
        if isinstance(b, tuple):
            b = np.array(b)
        return np.linalg.norm(a - b) * self.distance_cost

    def __lshift__(self, next_point: Union[tuple, list, np.ndarray]):
        """Add n_D points to the trip.

        Usage:
        >>> trip = Trip(offshore=[(0.20, 0.10), (0.21, 0.13)], budget=1)
        >>> list(trip)
        [(0, 0), (0.2, 0.1), (0.21, 0.13), (0, 0)]

        >>> trip.cost
        0.5022113550562322

        >>> trip <<= (0.23, 0.2)
        >>> trip.cost
        0.6328256863270314

        >>> try:
        ...     trip <<= [(0.4, 0.2), (0.38, 0.25)]
        ... except BudgetExhausted as e:
        ...     print(e)
        Budget exhausted: 1.0067449379375795 > 1


        Return an extended new trip including the new point, but still having the depot as ending point.

        Raise BudgetExhausted exception if beyond the budget."""
        # Add a batch of points.
        if isinstance(next_point, (np.ndarray, list)):
            trip = self
            for point in next_point:
                trip <<= point
            return trip

        # Add a single point.
        last_point = self.points[-2]
        new_cost = (
                self.cost  # Previous total.
                - self.leg_cost(last_point, self.depot)  # Discount old leg to depot.
                + self.probing_cost + self.leg_cost(last_point, next_point)  # Add new point costs.
                + self.leg_cost(next_point, self.depot)  # Add updated leg to depot.
        )
        if new_cost > self.budget:
            raise BudgetExhausted(f"Budget exhausted: {new_cost} > {self.budget}")
        return replace(self, offshore=self.offshore + [next_point])

    def __iter__(self):
        return iter(self.points)

    def __hash__(self):
        return id(self)


# solve_tsp

class BudgetExhausted(Exception):
    """Exception if beyond the budget."""
