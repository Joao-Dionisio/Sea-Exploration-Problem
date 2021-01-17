from dataclasses import dataclass
from dataclasses import dataclass
from functools import lru_cache
from typing import Union

import numpy
import numpy as np


# from typing import TYPE_CHECKING


# if TYPE_CHECKING:


@dataclass
class Probing:
    """Immutable set of n-D points (n>1) where the dimensions are called xy, xyz, xyza, xyzab... depending on n.

    Usage::
    >>> # From numpy.
    >>> import numpy as np
    >>> points_np = np.array([
    ...     [0.0, 0.1, 0.9],
    ...     [0.3, 0.1, 0.0],
    ...     [0.3, 0.2, 0.2],
    ...     [0.2, 0.3, 0.8],
    ...     [0.7, 0.4, 0.1],
    ...     [0.2, 0.3, 0.5]
    ... ])
    >>> probing = Probing(points_np, checkdups=False)
    >>> print(probing)
    [[0.  0.1 0.9]
     [0.3 0.1 0. ]
     [0.3 0.2 0.2]
     [0.2 0.3 0.8]
     [0.7 0.4 0.1]
     [0.2 0.3 0.5]]

    >>> # To avoid inconsistency, duplicates are checked by default at the cost of calling asdict() internally.
    >>> try:
    ...     probing = Probing(points_np)
    ... except Exception as e:
    ...     print(e)
    Duplicate items in provided array.

    >>> # From dict.
    >>> points = {
    ...     (0.0, 0.1): 0.9,
    ...     (0.3, 0.1): 0.0,
    ...     (0.3, 0.2): 0.2,
    ...     (0.2, 0.3): 0.8,
    ...     (0.7, 0.4): 0.1,
    ...     (0.2, 0.3): 0.5
    ... }

    >>> probing = Probing(points, checkdups=False)
    >>> # Note that 0.8 is overridden by 0.5, since they are on the same location.
    >>> probing
    Probing(points={(0.0, 0.1): 0.9, (0.3, 0.1): 0.0, (0.3, 0.2): 0.2, (0.2, 0.3): 0.5, (0.7, 0.4): 0.1}, name=None, eq_threshold=0, checkdups=False)

    >>> points_lst = [
    ...     [0.0, 0.1, 0.9],
    ...     [0.3, 0.1, 0.0],
    ...     [0.3, 0.2, 0.2],
    ...     [0.2, 0.3, 0.8],
    ...     [0.7, 0.4, 0.1],
    ...     [0.2, 0.3, 0.5]
    ... ]
    >>> try:
    ...     probing = Probing(points_lst)
    ... except Exception as e:
    ...     print(e)
    Duplicate items in provided list.

    >>> try:
    ...     probing = Probing(((1, 2), (3, 4)))
    ... except Exception as e:
    ...     print(e)
    ('Unknown type of probing points:', <class 'tuple'>)

    Parameters
    ----------
    points
        From 1 up until 10000 points, allocation time between 0.25us and 0.30us.
        From ~10000 up until 1300000, allocation time between 0.30us and 0.40us
        Above 1300000 points, allocation time near 4.0us.

    name

    eq_threshold

    checkdups

    """
    points: Union[dict, np.ndarray, list]
    name: str = None
    eq_threshold: float = 0
    checkdups: bool = True
    _asarray = None
    _asdict = None  # Data structure to, e.g., speed up pertinence check.
    _scatter = None

    def __post_init__(self):
        self.n = len(self.points)
        if isinstance(self.points, np.ndarray):
            self.d = self.points.shape[1] if self.n > 0 else None
            self._asarray = self.points
            if self.checkdups and self.n != len(self.asdict):
                raise Exception("Duplicate items in provided array.")
        elif isinstance(self.points, dict):
            self.d = len(next(iter(self.points))) + 1 if self.n > 0 else None
            self._asdict = self.points
        elif isinstance(self.points, list):
            self.d = len(self.points[0]) if self.n > 0 else None
            self._asarray = np.array(self.points)
            if self.checkdups and self.n != len(self.asdict):
                raise Exception("Duplicate items in provided list.")
        else:
            raise Exception("Unknown type of probing points:", type(self.points))

    @property
    def asarray(self):
        """As numpy array

        Usage:
        >>> points = {
        ...     (0.0, 0.1): 0.9,
        ...     (0.3, 0.1): 0.0,
        ...     (0.3, 0.2): 0.2,
        ...     (0.7, 0.4): 0.1,
        ...     (0.2, 0.3): 0.5
        ... }
        >>> Probing(points).asarray
        array([[0. , 0.1, 0.9],
               [0.3, 0.1, 0. ],
               [0.3, 0.2, 0.2],
               [0.7, 0.4, 0.1],
               [0.2, 0.3, 0.5]])
        """
        if self._asarray is None:
            self._asarray = np.array([tup + (z,) for tup, z in self.asdict.items()])
        return self._asarray

    @property
    def asdict(self):
        """As dict

        Usage:
        >>> import numpy as np
        >>> points_np = np.array([
        ...     [0.0, 0.1, 0.9],
        ...     [0.3, 0.1, 0.0],
        ...     [0.3, 0.2, 0.2],
        ...     [0.7, 0.4, 0.1],
        ...     [0.2, 0.3, 0.5]
        ... ])
        >>> Probing(points_np).asdict
        {(0.0, 0.1): 0.9, (0.3, 0.1): 0.0, (0.3, 0.2): 0.2, (0.7, 0.4): 0.1, (0.2, 0.3): 0.5}
        """
        if self._asdict is None:
            self._asdict = (self.d or {}) and {tuple(row[:self.d - 1]): float(row[self.d - 1]) for row in self.asarray}
        return self._asdict

    @lru_cache
    def __getattr__(self, item: str):
        """
        Usage::
        >>> points = {
        ...     (0.0, 0.1): 0.9,
        ...     (0.3, 0.1): 0.0,
        ...     (0.3, 0.2): 0.2,
        ...     (0.7, 0.4): 0.1,
        ...     (0.2, 0.3): 0.3
        ... }

        >>> probing = Probing(points)
        >>> probing.x
        array([[0. ],
               [0.3],
               [0.3],
               [0.7],
               [0.2]])

        >>> probing.xy
        array([[0. , 0.1],
               [0.3, 0.1],
               [0.3, 0.2],
               [0.7, 0.4],
               [0.2, 0.3]])

        >>> probing.z
        array([[0.9],
               [0. ],
               [0.2],
               [0.1],
               [0.3]])

        >>> points = {
        ...     (0.0, 0.1, 2.0, 1.2, 5.3): 0.9,
        ...     (0.3, 0.1, 1.3, 0.2, 0.3): 0.0,
        ...     (0.3, 0.2, 0.4, 3.5, 7.7): 0.2,
        ... }

        >>> probing = Probing(points)
        >>> # After 'x, y, z' comes 'a, b, c...'.
        >>> probing.zab
        array([[2. , 1.2, 5.3],
               [1.3, 0.2, 0.3],
               [0.4, 3.5, 7.7]])
        """
        if item in self.__dict__ or self.d is None:
            return self.__getattribute__(item)  # -pragma: no cover

        # Check if item is a sequence of coordinate letters.
        # TODO: fazer x, xy e xyz separado pra ver se fica mais rápido?
        coordsn = []
        for c in map(ord, item):
            c -= 120
            if c < 0:
                c += 26
            if not (0 <= c < self.d):
                # Coordinate letter outside number of dimensions.
                return self.__getattribute__(item)  # -pragma: no cover
            coordsn.append(c)
        return self.asarray[:, coordsn]

    def __lshift__(self, other: Union[tuple, np.ndarray]) -> 'Probing':
        """
        Add a point or points

        Usage::
        >>> import numpy as np
        >>> points = np.array([
        ...     [0.0, 0.1, 0.9],
        ...     [0.3, 0.1, 0.0],
        ...     [0.3, 0.2, 0.2],
        ...     [0.7, 0.4, 0.1],
        ...     [0.2, 0.3, 0.5]
        ... ])
        >>> probing = Probing(points)
        >>> print(probing)
        [[0.  0.1 0.9]
         [0.3 0.1 0. ]
         [0.3 0.2 0.2]
         [0.7 0.4 0.1]
         [0.2 0.3 0.5]]

        >>> print(probing << (3, 3, 3))
        [[0.  0.1 0.9]
         [0.3 0.1 0. ]
         [0.3 0.2 0.2]
         [0.7 0.4 0.1]
         [0.2 0.3 0.5]
         [3.  3.  3. ]]

        >>> # Insertion of an already existent item raises an exception.
        >>> try:
        ...     probing <<= 3, 3, 3
        ...     probing <<= 3, 3, 3
        ... except Exception as e:
        ...     print(e)
        Attempt to override old '(3, 3)' value '3' with '3' while appending a new point.

        >>> # Insertion of duplicate items raises an exception.
        >>> probing = Probing(points)
        >>> try:
        ...     print(probing << np.array([[2, 2, 2], [2, 2, 2]]))
        ... except Exception as e:
        ...     print(e)
        Duplicate items in provided array.


        >>> # Unsafe insertion of duplicate items is ignored by checkdups=False.
        >>> probing = Probing(points, checkdups=False)
        >>> print(probing << np.array([[2, 2, 2], [2, 2, 2]]))
        [[0.  0.1 0.9]
         [0.3 0.1 0. ]
         [0.3 0.2 0.2]
         [0.7 0.4 0.1]
         [0.2 0.3 0.5]
         [2.  2.  2. ]
         [2.  2.  2. ]]

        Returns
        -------
        Probing object extended by the given set of points.
        """
        if isinstance(other, tuple):
            other = [other]

        # Safe append.
        if self.checkdups:
            newdict = self.asdict.copy()
            for row in other:
                k, v = tuple(row[:-1]), row[-1]
                if k in self.asdict:
                    val = self.asdict[k]
                    msg = f"Attempt to override old '{k}' value '{v}' with '{val}' while appending a new point."
                    raise Exception(msg)
                newdict[k] = v
            if len(newdict) - self.n != len(other):
                raise Exception("Duplicate items in provided array.")
            return Probing(newdict)

        # Unsafe append.
        # Creating an array from scratch is faster and simpler than keeping memory allocated for future extensions.
        newarray = np.vstack([self.asarray, other])
        return Probing(newarray, checkdups=False)

    def __str__(self):
        """
        >>> print(str(Probing([[1,2,3],[4,5,6]])))
        [[1 2 3]
         [4 5 6]]
        """
        return str(self.asarray)

    def __hash__(self):  # doctest: +SKIP
        return id(self)

    @property
    def scatter(self):
        """Convert to a spatially correct 2D numpy ndarray.

        Usage::
        >>> points = {
        ...     (0.0, 0.1): 0.9,
        ...     (0.3, 0.1): 0.0,
        ...     (0.3, 0.2): 0.2,
        ...     (0.7, 0.4): 0.1,
        ...     (0.2, 0.3): 0.3
        ... }

        >>> probing = Probing(points)
        >>> probing.scatter
        array([[0.9, nan, nan, nan],
               [nan, nan, nan, nan],
               [nan, nan, 0.3, nan],
               [0. , 0.2, nan, nan],
               [nan, nan, nan, nan],
               [nan, nan, nan, nan],
               [nan, nan, nan, nan],
               [nan, nan, nan, 0.1]])

        Unknown points will be filled with NaN."""
        # TODO: generalize to any number of dimensions
        if self._scatter is None:
            xs, ys = self.asarray[:, 0], self.asarray[:, 1]
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
            for x, y, z in self.asarray:
                arr[int((x - minx) / minw), int((y - miny) / minh)] = z
            self._scatter = arr
        return self._scatter

#
#     @classmethod
#     def fromgrid(cls, side=4, f=lambda *_: 0.0, name=None):  # , simulated=None):
#         """A new Probings object containing 'side'x'side' 2D points with z values given by function 'f'.
#
#         Leave a margin of '1 / (2 * side)' around extreme points.
#
#         Usage::
#             >>> probings = Probings.fromgrid(f=lambda x, y: x * y)
#             >>> probings.show()
#             [[0.015625 0.046875 0.078125 0.109375]
#              [0.046875 0.140625 0.234375 0.328125]
#              [0.078125 0.234375 0.390625 0.546875]
#              [0.109375 0.328125 0.546875 0.765625]]
#         """
#         # if simulated is not None:
#         #     raise NotImplementedError("'simulated' flag still not ready.")
#         # true_value = not simulated        TODO
#         points = {}
#         margin = 1 / (2 * side)
#         for x in ap[margin, 3 * margin, ..., 1]:
#             for y in ap[margin, 3 * margin, ..., 1]:
#                 points[x, y] = f(x, y)  # , true_value
#         return Probings(points, name=name)
#
#     @classmethod
#     def fromrandom(cls, size=16, f=lambda *_: 0.0, rnd=None, name=None):  # simulated=None,
#         """A new Probings object containing 'size' 2D points with z values given by function 'f'."""
#         # if simulated is not None:
#         #     raise NotImplementedError("'simulated' flag still not ready.")
#         # true_value = not simulated  TODO
#         if rnd is None:
#             rnd = numpy.random.default_rng(0)
#         if isinstance(rnd, int):
#             rnd = numpy.random.default_rng(rnd)
#         points = {}
#         for i in range(size):
#             x = rnd.random()
#             y = rnd.random()
#             points[x, y] = f(x, y)  # , true_value
#         return Probings(points, name=name)
#
#
#     # def items(self):
#     #     return self.points.items()
#
#     @cached_property
#     def xy(self):
#         """2D points as ndarray
#
#         Usage:
#             >>> from seaexp import Probings
#             >>> Probings.fromrandom(3).xy
#             array([[0.63696169, 0.26978671],
#                    [0.04097352, 0.01652764],
#                    [0.81327024, 0.91275558]])
#         """
#         return numpy.array(list(self.points.keys()))
#
#     @cached_property
#     def z(self):
#         return list(self.points.values())
#
#     @cached_property
#     def xy_z(self):
#         return [numpy.array(pair) for pair in self.xy], self.z
#
#     @cached_property
#     def x_y_z(self):
#         return tuple(zip(*self.xy)) + (self.z,)
#
#     @cached_property
#     def xyz(self):
#         return [(x, y, z) for (x, y), z in self]
#
#     def __iter__(self):
#         yield from self.points.items()
#
#     def __sub__(self, other):
#         """Element-wise subtraction of a Probings object from another
#
#         The points need to match. Otherwise, see Seabed.__sub__.
#
#         Usage:
#             >>> from seaexp.gpr import GPR
#             >>> true_value = lambda a, b: a * b
#             >>> real = Probings.fromgrid(side=5, f=true_value)
#             >>> estimator = GPR("rbf")(real)
#             >>> estimated = Probings.fromgrid(side=5, f=estimator)
#             >>> (real - estimated).show()
#             [[ 0.00000176 -0.0000035  -0.00000031  0.00000417 -0.00000212]
#              [-0.0000035   0.00000768 -0.00000042 -0.00000753  0.00000373]
#              [-0.00000031 -0.00000042 -0.00000024  0.00000049  0.00000059]
#              [ 0.00000417 -0.00000753  0.00000049  0.00000764 -0.00000488]
#              [-0.00000212  0.00000373  0.00000059 -0.00000488  0.00000272]]
#         """
#         newpoints = {k: self[k] - other[k] for k in self.points}
#         return Probings(newpoints)
#
#     # def __divmod__(self, other):
#     def __abs__(self):
#         """
#         Usage:
#             >>> probings = Probings.fromgrid(f=lambda x, y: -x * y)
#             >>> print(probings)
#             [[-0.015625 -0.046875 -0.078125 -0.109375]
#              [-0.046875 -0.140625 -0.234375 -0.328125]
#              [-0.078125 -0.234375 -0.390625 -0.546875]
#              [-0.109375 -0.328125 -0.546875 -0.765625]]
#             >>> print(abs(probings))
#             [[0.015625 0.046875 0.078125 0.109375]
#              [0.046875 0.140625 0.234375 0.328125]
#              [0.078125 0.234375 0.390625 0.546875]
#              [0.109375 0.328125 0.546875 0.765625]]
#
#         Returns
#         -------
#
#         """
#         # # Only use np if it is already calculated.
#         # if self._np is None:
#         newpoints = {k: abs(v) for k, v in self.points.items()}
#         return Probings(newpoints)
#         # newnp = abs(self.np)
#         # newprobings = Probings.fromnp(newnp)
#         # newprobings._np = newnp  # Do not waste the calculation of np, caching just in case someone call it.
#         # return newprobings
#
#     @classmethod
#     def fromnp(cls, np: numpy.ndarray):
#         """Create a Probings object from a 2D numpy array
#
#         Usage:
#             >>> array = numpy.array([[0.4, 0.3, 0.3],[0, 0, 0.5],[0.2, 0.3, 0.8]])
#             >>> array
#             array([[0.4, 0.3, 0.3],
#                    [0. , 0. , 0.5],
#                    [0.2, 0.3, 0.8]])
#             >>> fromnp = Probings.fromnp(array)
#             >>> fromnp
#             Probings(points={(0, 0): 0.4, (1, 0): 0, (2, 0): 0.2, (0, 1): 0.3, (1, 1): 0, (2, 1): 0.3, (0, 2): 0.3, (1, 2): 0.5, (2, 2): 0.8}, name=None)
#             >>> (fromnp.np == array).all()
#             True
#
#         """
#         m = np.copy()
#         m[m == 0] = 8124567890123456  # Save zeros.
#         m = csr_matrix(m).todok()  # Convert to sparse.
#         points = {k: 0 if v == 8124567890123456 else v for k, v in dict(m).items()}  # Recover zeros.
#         return Probings(points)
#
#     @cached_property
#     def sum(self):
#         return sum(self.z)
#
#     @cached_property
#     def abs(self):
#         return abs(self)
#
#     def __getitem__(self, item):
#         if isinstance(item, slice):
#             return Probings(dict(list(self)[item]))
#         return self.points[item]
#
#     def __call__(self, x, y):
#         return self[(x, y)]
#
#     # noinspection PyUnresolvedReferences
#     def __xor__(self, f):
#         """Replace z values according to the given function
#
#         Usage:
#             >>> zeroed = Probings.fromgrid(side=5)
#             >>> print(zeroed)
#             [[0. 0. 0. 0. 0.]
#              [0. 0. 0. 0. 0.]
#              [0. 0. 0. 0. 0.]
#              [0. 0. 0. 0. 0.]
#              [0. 0. 0. 0. 0.]]
#             >>> (zeroed ^ (lambda x, y: x * y)).show()
#             [[0.01 0.03 0.05 0.07 0.09]
#              [0.03 0.09 0.15 0.21 0.27]
#              [0.05 0.15 0.25 0.35 0.45]
#              [0.07 0.21 0.35 0.49 0.63]
#              [0.09 0.27 0.45 0.63 0.81]]
#         """
#         return Probings({(x, y): f(x, y) for x, y in self.points})
#
#     @property
#     def n(self):
#         return len(self.points)
#
#     def show(self):
#         """Print z values as a rounded float matrix"""
#         with numpy.printoptions(suppress=True, linewidth=1000, precision=8):
#             print(self)
#
#     def shuffled(self, rnd=numpy.random.default_rng(0)):
#         tuples = list(self.points.items())
#         shuffle(tuples, rnd.random)
#         return Probings(dict(tuples))
#
#     def __and__(self, other):
#         """Concatenation of two Probings
#
#         The right operand will override the left operand in case of key collision.
#         Usage:
#             >>> true_value = lambda a, b: a * b
#             >>> a = Probings.fromgrid(side=3, f=true_value)
#             >>> b = Probings.fromgrid(side=2, f=true_value)
#             >>> a.show()
#             [[0.02777778 0.08333333 0.13888889]
#              [0.08333333 0.25       0.41666667]
#              [0.13888889 0.41666667 0.69444444]]
#             >>> b.show()
#             [[0.0625 0.1875]
#              [0.1875 0.5625]]
#             >>> (a & b).show()  # Note that the output granularity is adjusted to the minimum possible.
#             [[0.02777778        nan        nan        nan 0.08333333        nan        nan        nan 0.13888889]
#              [       nan 0.0625            nan        nan        nan        nan        nan 0.1875            nan]
#              [       nan        nan        nan        nan        nan        nan        nan        nan        nan]
#              [       nan        nan        nan        nan        nan        nan        nan        nan        nan]
#              [0.08333333        nan        nan        nan 0.25              nan        nan        nan 0.41666667]
#              [       nan        nan        nan        nan        nan        nan        nan        nan        nan]
#              [       nan        nan        nan        nan        nan        nan        nan        nan        nan]
#              [       nan 0.1875            nan        nan        nan        nan        nan 0.5625            nan]
#              [0.13888889        nan        nan        nan 0.41666667        nan        nan        nan 0.69444444]]
#         """
#
#         return Probings(self.points | other.points)
#
#     def plot(self, xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), name=None, block=True):
#         """
#
#         Usage:
#             >>> from seaexp import Seabed
#             >>> f = Seabed.fromgaussian()
#             >>> g = Seabed.fromgaussian()
#             >>> probings = Probings.fromrandom(20, f  + g)
#             >>> probings.plot()
#
#         Returns
#         -------
#
#         """
#         from seaexp.plotter import Plotter
#         plt = Plotter(xlim, ylim, zlim, name, inplace=False, block=block)
#         name_ = self.name
#         self.name = None
#         plt << self
#         self.name = name_
#         self.plots.append(plt)  # Keeps a reference, so plt destruction (and  window creation) is delayed.
#
#     @cached_property
#     def max(self):
#         """Return maximum z."""
#         return max(self.z)
#
#     @cached_property
#     def argmax(self):
#         """Return list of x,y points with maximum z."""
#         return [(x, y) for z, (x, y) in zip(self.z, self.xy) if abs(z - self.max) <= self.eq_threshold]
#
#     @cached_property
#     def min(self):
#         return min(self.z)
#
#
# def cv(probings, k=5, rnd=None):
#     """
#     Usage:
#         >>> probings = Probings({(0, 0): 0, (1, 1): 0, (2, 2): 0, (3, 3): 0, (4, 4): 0, (5, 5): 0})
#         >>> probings.xy
#         [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
#         >>> for run, (training, test) in enumerate(cv(probings, k=3)):
#         ...     print(f"Run {run}:")
#         ...     print("training:", training.xy)
#         ...     print("test:", test.xy)
#         Run 0:
#         training: [(5, 5), (0, 0), (1, 1), (3, 3)]
#         test: [(2, 2), (4, 4)]
#         Run 1:
#         training: [(2, 2), (4, 4), (1, 1), (3, 3)]
#         test: [(5, 5), (0, 0)]
#         Run 2:
#         training: [(2, 2), (4, 4), (5, 5), (0, 0)]
#         test: [(1, 1), (3, 3)]
#
#     Parameters
#     ----------
#     probings
#     k
#     rnd
#
#     Returns
#     -------
#
#     """
#     if probings.n < k:
#         raise Exception(f"Number of points ({probings.n}) is smaller than k ({k}).")
#     if rnd is None:
#         rnd = numpy.random.default_rng(0)
#     min_fold_size, rem = divmod(probings.n, k)
#     shuffled = probings.shuffled(rnd)
#     folds = []
#     i = 0
#     while i < probings.n:
#         fold_size = min_fold_size
#         if i < rem:
#             fold_size += 1
#         folds.append(shuffled[i:i + fold_size])
#         i += fold_size
#
#     for i in range(k):
#         tr = reduce(operator.and_, folds[0:(i + k) % k] + folds[i + 1: k])
#         ts = folds[i]
#         yield tr, ts
