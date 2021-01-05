from functools import cached_property

import lange
import numpy as np
import random as rnd
from dataclasses import dataclass


@dataclass
class Probings:
    """Immutable set of 2D points where the "z" values are known or simulated

    Usage::
        >>> known_points = {
        ...     (0.0, 0.1): (0.12, True),
        ...     (0.2, 0.3): (0.39, True)
        ... }
        >>> probings = Probings(known_points)
        >>> probings <<= 4, 1, 0.34, False  # Add simulated point.
        >>> print(probings)
        Probings(points={(0.0, 0.1): (0.12, True), (0.2, 0.3): (0.39, True), (4, 1): (0.34, False)})
    """
    points: dict

    def __lshift__(self, tuple4):
        """Add a tuple (x, y, z, b) to the set, where 'b' is a boolean indicating if this is a true value.

        Return an extended set of points"""
        key, val = tuple4[:2], tuple4[2:]
        if key in self.points:
            print(f"W: overriding old {key} value {self.points[key]} with {val} in the new set of points")
        newpoints = self.points.copy()
        newpoints[key] = val
        return Probings(newpoints)

    @classmethod
    def fromgrid(cls, f, side=4, simulated=False):
        """A new Probings object containing 'side'x'side' 2D points with z values given by function 'f'.

        Leave a margin of '1 / (2 * side)' around extreme points.

        Usage::
            >>> probings = Probings.fromgrid(lambda x, y: x * y)
            >>> print(probings)
            [[0.015625 0.046875 0.078125 0.109375]
             [0.046875 0.140625 0.234375 0.328125]
             [0.078125 0.234375 0.390625 0.546875]
             [0.109375 0.328125 0.546875 0.765625]]
        """
        true_value = not simulated
        points = {}
        margin = 1 / (2 * side)
        for x in -[margin, 3 * margin, ..., 1]:
            for y in -[margin, 3 * margin, ..., 1]:
                points[x, y] = f(x, y), true_value
        return Probings(points)

    @classmethod
    def fromrandom(cls, f, size=16, seed=0, simulated=False):
        """A new Probings object containing 'size' 2D points with z values given by function 'f'."""
        true_value = not simulated
        rnd.seed(seed)
        points = {}
        for i in range(size):
            x = rnd.random()
            y = rnd.random()
            points[x, y] = f(x, y), true_value
        return Probings(points)

    @cached_property
    def np(self):
        """Convert to a 2D numpy ndarray."""
        xys = self.points.keys()
        xs, ys = zip(*xys)
        setx, sety = set(xs), set(ys)
        minx, miny = min(setx), min(sety)
        maxx, maxy = max(setx), max(sety)
        ordw, ordh = sorted(list(setx)), sorted(list(sety))
        difw = abs(np.array(ordw[1:] + ordw[0:1]) - np.array(ordw))
        difh = abs(np.array(ordh[1:] + ordh[0:1]) - np.array(ordh))
        minw, minh = min(difw), min(difh)
        w, h = int((maxx - minx) / minw) + 1, int((maxy - miny) / minh) + 1
        arr = np.zeros((w, h))
        for (x, y), (z, b) in self.points.items():
            arr[int((x - minx) / minw), int((y - miny) / minh)] = z
        return arr

    def items(self):
        return self.points.items()

    def __iter__(self):
        return iter(self.points)

    def __str__(self):
        return str(np.round(self.np * 1000) / 1000)

    @cached_property
    def xy_z(self):
        return [np.array(pair) for pair in self.points.keys()], [tup[1] for tup in self.points.values()]
