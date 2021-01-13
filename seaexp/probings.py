import operator
from dataclasses import dataclass
from functools import cached_property
from functools import reduce
from random import shuffle

import numpy
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
        >>> probings <<= 0.9, 1, 0.34  # Add point.
        >>> print(probings)  # NaN means 'unknown z value at that point'.
        [[0.12  nan  nan  nan  nan]
         [ nan 0.39  nan  nan  nan]
         [ nan  nan  nan  nan  nan]
         [ nan  nan  nan  nan  nan]
         [ nan  nan  nan  nan 0.34]]
    """
    points: dict
    name: str = None
    plots = []
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
    def fromgrid(cls, side=4, f=lambda *_: 0.0, name=None):  # , simulated=None):
        """A new Probings object containing 'side'x'side' 2D points with z values given by function 'f'.

        Leave a margin of '1 / (2 * side)' around extreme points.

        Usage::
            >>> probings = Probings.fromgrid(f=lambda x, y: x * y)
            >>> probings.show()
            [[0.015625 0.046875 0.078125 0.109375]
             [0.046875 0.140625 0.234375 0.328125]
             [0.078125 0.234375 0.390625 0.546875]
             [0.109375 0.328125 0.546875 0.765625]]
        """
        # if simulated is not None:
        #     raise NotImplementedError("'simulated' flag still not ready.")
        # true_value = not simulated        TODO
        points = {}
        margin = 1 / (2 * side)
        for x in ap[margin, 3 * margin, ..., 1]:
            for y in ap[margin, 3 * margin, ..., 1]:
                points[x, y] = f(x, y)  # , true_value
        return Probings(points, name=name)

    @classmethod
    def fromrandom(cls, size=16, f=lambda *_: 0.0, rnd=None, name=None):  # simulated=None,
        """A new Probings object containing 'size' 2D points with z values given by function 'f'."""
        # if simulated is not None:
        #     raise NotImplementedError("'simulated' flag still not ready.")
        # true_value = not simulated  TODO
        if rnd is None:
            rnd = numpy.random.default_rng(0)
        if isinstance(rnd, int):
            rnd = numpy.random.default_rng(rnd)
        points = {}
        for i in range(size):
            x = rnd.random()
            y = rnd.random()
            points[x, y] = f(x, y)  # , true_value
        return Probings(points, name=name)

    @property
    def np(self):
        """Convert to a spatially correct 2D numpy ndarray.

        Unknown points will be filled with NaN."""
        if self._np is None:
            xys = self.points.keys()
            xs, ys = zip(*xys)
            setx, sety = set(xs), set(ys)
            minx, miny = min(setx), min(sety)
            maxx, maxy = max(setx), max(sety)
            ordw, ordh = sorted(list(setx)), sorted(list(sety))
            difw = abs(numpy.array(ordw[1:] + ordw[0:1]) - numpy.array(ordw))
            difh = abs(numpy.array(ordh[1:] + ordh[0:1]) - numpy.array(ordh))
            minw, minh = min(difw), min(difh)
            w, h = int((maxx - minx) / minw) + 1, int((maxy - miny) / minh) + 1
            arr = numpy.empty((w, h))
            arr.fill(numpy.nan)
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
        return [numpy.array(pair) for pair in self.xy], self.z

    @cached_property
    def x_y_z(self):
        return tuple(zip(*self.xy)) + (self.z,)

    @cached_property
    def xyz(self):
        return [(x, y, z) for (x, y), z in self]

    def __iter__(self):
        yield from self.points.items()

    def __str__(self):
        return str(self.np)

    def __sub__(self, other):
        """Element-wise subtraction of a Probings object from another

        The points need to match. Otherwise, see Seabed.__sub__.

        Usage:
            >>> from seaexp.gpr import GPR
            >>> true_value = lambda a, b: a * b
            >>> real = Probings.fromgrid(side=5, f=true_value)
            >>> estimator = GPR("rbf")(real)
            >>> estimated = Probings.fromgrid(side=5, f=estimator)
            >>> (real - estimated).show()
            [[ 0.00000176 -0.0000035  -0.00000031  0.00000417 -0.00000212]
             [-0.0000035   0.00000768 -0.00000042 -0.00000753  0.00000373]
             [-0.00000031 -0.00000042 -0.00000024  0.00000049  0.00000059]
             [ 0.00000417 -0.00000753  0.00000049  0.00000764 -0.00000488]
             [-0.00000212  0.00000373  0.00000059 -0.00000488  0.00000272]]
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
    def fromnp(cls, np: numpy.ndarray):
        """Create a Probings object from a 2D numpy array

        Usage:
            >>> array = numpy.array([[0.4, 0.3, 0.3],[0, 0, 0.5],[0.2, 0.3, 0.8]])
            >>> array
            array([[0.4, 0.3, 0.3],
                   [0. , 0. , 0.5],
                   [0.2, 0.3, 0.8]])
            >>> fromnp = Probings.fromnp(array)
            >>> fromnp
            Probings(points={(0, 0): 0.4, (1, 0): 0, (2, 0): 0.2, (0, 1): 0.3, (1, 1): 0, (2, 1): 0.3, (0, 2): 0.3, (1, 2): 0.5, (2, 2): 0.8}, name=None)
            >>> (fromnp.np == array).all()
            True

        """
        m = np.copy()
        m[m == 0] = 8124567890123456  # Save zeros.
        m = csr_matrix(m).todok()  # Convert to sparse.
        points = {k: 0 if v == 8124567890123456 else v for k, v in dict(m).items()}  # Recover zeros.
        return Probings(points)

    @cached_property
    def sum(self):
        return sum(self.z)

    @cached_property
    def abs(self):
        return abs(self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Probings(dict(list(self)[item]))
        return self.points[item]

    def __call__(self, x, y):
        return self[(x, y)]

    # noinspection PyUnresolvedReferences
    def __xor__(self, f):
        """Replace z values according to the given function

        Usage:
            >>> zeroed = Probings.fromgrid(side=5)
            >>> print(zeroed)
            [[0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0.]]
            >>> (zeroed ^ (lambda x, y: x * y)).show()
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

    def show(self):
        """Print z values as a rounded float matrix"""
        with numpy.printoptions(suppress=True, linewidth=1000, precision=8):
            print(self)

    def shuffled(self, rnd=numpy.random.default_rng(0)):
        tuples = list(self.points.items())
        shuffle(tuples, rnd.random)
        return Probings(dict(tuples))

    def __and__(self, other):
        """Concatenation of two Probings

        The right operand will override the left operand in case of key collision.
        Usage:
            >>> true_value = lambda a, b: a * b
            >>> a = Probings.fromgrid(side=3, f=true_value)
            >>> b = Probings.fromgrid(side=2, f=true_value)
            >>> a.show()
            [[0.02777778 0.08333333 0.13888889]
             [0.08333333 0.25       0.41666667]
             [0.13888889 0.41666667 0.69444444]]
            >>> b.show()
            [[0.0625 0.1875]
             [0.1875 0.5625]]
            >>> (a & b).show()  # Note that the output granularity is adjusted to the minimum possible.
            [[0.02777778        nan        nan        nan 0.08333333        nan        nan        nan 0.13888889]
             [       nan 0.0625            nan        nan        nan        nan        nan 0.1875            nan]
             [       nan        nan        nan        nan        nan        nan        nan        nan        nan]
             [       nan        nan        nan        nan        nan        nan        nan        nan        nan]
             [0.08333333        nan        nan        nan 0.25              nan        nan        nan 0.41666667]
             [       nan        nan        nan        nan        nan        nan        nan        nan        nan]
             [       nan        nan        nan        nan        nan        nan        nan        nan        nan]
             [       nan 0.1875            nan        nan        nan        nan        nan 0.5625            nan]
             [0.13888889        nan        nan        nan 0.41666667        nan        nan        nan 0.69444444]]
        """

        return Probings(self.points | other.points)

    def plot(self, xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), name=None, block=True):
        """

        Usage:
            >>> from seaexp import Seabed
            >>> f = Seabed.fromgaussian()
            >>> g = Seabed.fromgaussian()
            >>> probings = Probings.fromrandom(20, f  + g)
            >>> probings.plot()

        Returns
        -------

        """
        from seaexp.plotter import Plotter
        plt = Plotter(xlim, ylim, zlim, name, inplace=False, block=block)
        name_ = self.name
        self.name = None
        plt << self
        self.name = name_
        self.plots.append(plt)  # Keeps a reference, so plt destruction (and  window creation) is delayed.


def cv(probings, k=5, rnd=None):
    """
    Usage:
        >>> probings = Probings({(0, 0): 0, (1, 1): 0, (2, 2): 0, (3, 3): 0, (4, 4): 0, (5, 5): 0})
        >>> probings.xy
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        >>> for run, (training, test) in enumerate(cv(probings, k=3)):
        ...     print(f"Run {run}:")
        ...     print("training:", training.xy)
        ...     print("test:", test.xy)
        Run 0:
        training: [(5, 5), (0, 0), (1, 1), (3, 3)]
        test: [(2, 2), (4, 4)]
        Run 1:
        training: [(2, 2), (4, 4), (1, 1), (3, 3)]
        test: [(5, 5), (0, 0)]
        Run 2:
        training: [(2, 2), (4, 4), (5, 5), (0, 0)]
        test: [(1, 1), (3, 3)]

    Parameters
    ----------
    probings
    k
    rnd

    Returns
    -------

    """
    if probings.n < k:
        raise Exception(f"Number of points ({probings.n}) is smaller than k ({k}).")
    if rnd is None:
        rnd = numpy.random.default_rng(0)
    min_fold_size, rem = divmod(probings.n, k)
    shuffled = probings.shuffled(rnd)
    folds = []
    i = 0
    while i < probings.n:
        fold_size = min_fold_size
        if i < rem:
            fold_size += 1
        folds.append(shuffled[i:i + fold_size])
        i += fold_size

    for i in range(k):
        tr = reduce(operator.and_, folds[0:(i + k) % k] + folds[i + 1: k])
        ts = folds[i]
        yield tr, ts
