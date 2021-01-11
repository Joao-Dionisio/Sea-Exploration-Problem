from dataclasses import dataclass
from functools import cached_property

import numpy as np
from lange import ap
from scipy.sparse import csr_matrix


# from typing import TYPE_CHECKING


# if TYPE_CHECKING:


@dataclass
class Probings:
    """Immutable set of 2D points where the "z" values are known or simulated

    Usage::
        >>> known_points = {
        ...     (0.0, 0.1): 0.12,
        ...     (0.2, 0.3): 0.39
        ... }
        >>> probings = Probings(known_points)
        >>> probings <<= 4, 1, 0.34  # Add point.
        >>> print(probings)
        [[0.12 0.   0.   0.   0.  ]
         [0.   0.39 0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.  ]
         [0.   0.   0.   0.   0.34]]
    """
    points: dict
    _np = None

    def __lshift__(self, point):
        """Add a tuple (x, y, z) to the set.

        If z is a tuple (z', b), b is a boolean indicating whether z' is a true value.  (TODO)

        Return a Probings object which is an extended set of points"""
        key, val = point[:2], point[-1]
        if key in self.points:
            print(f"W: overriding old {key} value {self.points[key]} with {val} in the new set of points")
        newpoints = self.points.copy()
        newpoints[key] = val
        return Probings(newpoints)

    @classmethod
    def fromgrid(cls, side=4, f=lambda *_: 0):  # , simulated=None):
        """A new Probings object containing 'side'x'side' 2D points with z values given by function 'f'.

        Leave a margin of '1 / (2 * side)' around extreme points.

        Usage::
            >>> np.set_printoptions(precision=5)
            >>> probings = Probings.fromgrid(f=lambda x, y: x * y)
            >>> print(probings)
            [[0.01562 0.04688 0.07812 0.10938]
             [0.04688 0.14062 0.23438 0.32812]
             [0.07812 0.23438 0.39062 0.54688]
             [0.10938 0.32812 0.54688 0.76562]]
        """
        # if simulated is not None:
        #     raise NotImplementedError("'simulated' flag still not ready.")
        # true_value = not simulated        TODO
        points = {}
        margin = 1 / (2 * side)
        for x in ap[margin, 3 * margin, ..., 1]:
            for y in ap[margin, 3 * margin, ..., 1]:
                points[x, y] = f(x, y)  # , true_value
        return Probings(points)

    @classmethod
    def fromrandom(cls, size=16, f=lambda *_: 0, rnd=np.random.default_rng(0)):  # simulated=None,
        """A new Probings object containing 'size' 2D points with z values given by function 'f'."""
        # if simulated is not None:
        #     raise NotImplementedError("'simulated' flag still not ready.")
        # true_value = not simulated  TODO
        points = {}
        for i in range(size):
            x = rnd.random()
            y = rnd.random()
            points[x, y] = f(x, y)  # , true_value
        return Probings(points)

    @property
    def np(self):
        """Convert to a 2D numpy ndarray."""
        if self._np is None:
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
            for (x, y), z in self.points.items():
                arr[int((x - minx) / minw), int((y - miny) / minh)] = z
            self._np = arr
        return self._np

    # def items(self):
    #     return self.points.items()

    @cached_property
    def xy(self):
        return list(self.points.keys())

    @cached_property
    def z(self):
        return list(self.points.values())

    @cached_property
    def xy_z(self):
        return [np.array(pair) for pair in self.xy], self.z

    def __iter__(self):
        return iter(self.points)

    def __str__(self):
        return str(self.np)

    def __sub__(self, other):
        """Subtract one Probings object from another

        The points need to match. Otherwise, see Seabed.__sub__.

        Usage:
            >>> from seaexp.estimator import Estimator
            >>> np.set_printoptions(precision=10, suppress=True)
            >>> true_value = lambda a, b: a * b
            >>> real = Probings.fromgrid(side=5, f=true_value)
            >>> estimator = Estimator(real, "rbf")
            >>> estimated = Probings.fromgrid(side=5, f=estimator)
            >>> print(real - estimated)
            [[ 0.0000017569 -0.0000034972 -0.0000003108  0.00000417   -0.0000021208]
             [-0.0000034972  0.0000076777 -0.0000004205 -0.0000075265  0.000003735 ]
             [-0.0000003108 -0.0000004206 -0.0000002403  0.0000004872  0.0000005891]
             [ 0.00000417   -0.0000075264  0.0000004872  0.0000076428 -0.0000048833]
             [-0.0000021208  0.000003735   0.0000005891 -0.0000048834  0.0000027188]]
         """
        newpoints = {k: self[k] - other[k] for k in self.points}
        return Probings(newpoints)

    # def __divmod__(self, other):
    def __abs__(self):
        """
        Usage:
            >>> probings = Probings.fromgrid(f=lambda x, y: -x * y)
            >>> print(probings)
            [[-0.015625 -0.046875 -0.078125 -0.109375]
             [-0.046875 -0.140625 -0.234375 -0.328125]
             [-0.078125 -0.234375 -0.390625 -0.546875]
             [-0.109375 -0.328125 -0.546875 -0.765625]]
            >>> print(abs(probings))
            [[0.015625 0.046875 0.078125 0.109375]
             [0.046875 0.140625 0.234375 0.328125]
             [0.078125 0.234375 0.390625 0.546875]
             [0.109375 0.328125 0.546875 0.765625]]

        Returns
        -------

        """
        # # Only use np if it is already calculated.
        # if self._np is None:
        newpoints = {k: abs(v) for k, v in self.points.items()}
        return Probings(newpoints)
        # newnp = abs(self.np)
        # newprobings = Probings.fromnp(newnp)
        # newprobings._np = newnp  # Do not waste the calculation of np, caching just in case someone call it.
        # return newprobings

    @classmethod
    def fromnp(cls, np):
        """Create a Probings object from a 2D numpy array

        Usage:
            >>> array = np.array([[0.4, 0.3, 0.3],[0, 0, 0.5],[0.2, 0.3, 0.8]])
            >>> array
            array([[0.4, 0.3, 0.3],
                   [0. , 0. , 0.5],
                   [0.2, 0.3, 0.8]])
            >>> fromnp = Probings.fromnp(array)
            >>> fromnp
            Probings(points={(0, 0): 0.4, (2, 0): 0.2, (0, 1): 0.3, (2, 1): 0.3, (0, 2): 0.3, (1, 2): 0.5, (2, 2): 0.8})
            >>> (fromnp.np == array).all()
            True

        """
        return Probings(dict(csr_matrix(np).todok()))

    @cached_property
    def sum(self):
        return sum(self.z)

    @cached_property
    def abs(self):
        return abs(self)

    def __getitem__(self, item):
        return self.points[item]

    def __call__(self, x, y):
        return self[(x, y)]

    def __matmul__(self, f):
        """Replace z values according to the given function

        Usage:
            >>> zeroed = Probings.fromgrid(side=5)
            >>> print(zeroed)
            [[0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]]
            >>> print(zeroed @ (lambda x, y: x * y))
            [[0.01 0.03 0.05 0.07 0.09]
             [0.03 0.09 0.15 0.21 0.27]
             [0.05 0.15 0.25 0.35 0.45]
             [0.07 0.21 0.35 0.49 0.63]
             [0.09 0.27 0.45 0.63 0.81]]
        """
        return Probings({(x, y): f(x, y) for x, y in self.points})

    @property
    def n(self):
        return len(self.points)
