from dataclasses import dataclass, replace
from functools import lru_cache, reduce
import operator
from typing import Union

import numpy
import numpy as np
# if TYPE_CHECKING:
from lange import ap


# from typing import TYPE_CHECKING


@dataclass
class Probing:
    """Immutable set of n-D points (n>1) where the dimensions are called xy, xyz, xyza, xyzab... depending on n.

    Usage:
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

    >>> probing.lastcoord
    'z'

    >>> try:
    ...     probing = Probing(((1, 2), (3, 4)))
    ... except Exception as e:
    ...     print(e)
    ('Unknown type of probing points:', <class 'tuple'>)

    >>> # Numpy operations
    >>> points = {
    ...     (0.0, 0.1): 1,
    ...     (0.3, 0.1): 2,
    ...     (0.3, 0.2): 3,
    ...     (0.2, 0.3): 4,
    ...     (0.7, 0.4): 5
    ... }
    >>> a = Probing(points)
    >>> a
    Probing(points={(0.0, 0.1): 1, (0.3, 0.1): 2, (0.3, 0.2): 3, (0.2, 0.3): 4, (0.7, 0.4): 5}, name=None, eq_threshold=0, checkdups=True)

    >>> b = Probing(dict(reversed(list(points.items()))))
    >>> b
    Probing(points={(0.7, 0.4): 5, (0.2, 0.3): 4, (0.3, 0.2): 3, (0.3, 0.1): 2, (0.0, 0.1): 1}, name=None, eq_threshold=0, checkdups=True)

    >>> print(a + b)
    [[0.  0.1 6. ]
     [0.3 0.1 6. ]
     [0.3 0.2 6. ]
     [0.2 0.3 6. ]
     [0.7 0.4 6. ]]


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
    d = None
    lastcoord = None
    _asarray = None
    _asdict = None  # Data structure to, e.g., speed up pertinence check.
    _scatter = None
    _memo = {}
    _min = None
    _max = None

    def __new__(cls, *args, **kwargs):
        """
        >>> true_value = lambda ab: ab[:, 0] * ab[:, 1]
        >>> a = Probing.fromgrid(2, 2, f=true_value)
        >>> a
        Probing(points=array([[0.25  , 0.25  , 0.0625],
               [0.75  , 0.25  , 0.1875],
               [0.25  , 0.75  , 0.1875],
               [0.75  , 0.75  , 0.5625]]), name=None, eq_threshold=0, checkdups=True)

        >>> b = Probing.fromgrid(2, 1, f=true_value)
        >>> b
        Probing(points=array([[0.25 , 0.5  , 0.125],
               [0.75 , 0.5  , 0.375]]), name=None, eq_threshold=0, checkdups=True)

        >>> print(a & b)
        [[0.25   0.25   0.0625]
         [0.75   0.25   0.1875]
         [0.25   0.75   0.1875]
         [0.75   0.75   0.5625]
         [0.25   0.5    0.125 ]
         [0.75   0.5    0.375 ]]

        """
        # Numpy ops.  # TODO: complete lists
        unary_target = {"abs": 0, "sum": 0}
        binary_target = {"divmod": 0, "add": 0, "mul": "multiply", "sub": "subtract"}
        binary_all = {"and": "vstack"}
        for magicm, npm in (unary_target | binary_target | binary_all).items():
            def f(name):
                h = getattr(np, name)
                if magicm in unary_target:
                    def g(self):
                        if name not in self._memo:
                            newpys = np.column_stack([self.input, h(self.target).reshape(self.n, 1)])
                            self._memo[name] = replace(self, points=newpys)
                        return self._memo[(name)]
                elif magicm in binary_target:
                    def g(self, other):
                        if (name, other) not in self._memo:
                            lst = [self.input, h(self.target, other.target).reshape(self.n, 1)]
                            newpts = np.column_stack(lst)
                            self._memo[(name, other)] = replace(self, points=newpts)
                        return self._memo[(name, other)]
                else:
                    def g(self, other):
                        if (name, other) not in self._memo:
                            self._memo[(name, other)] = replace(self, points=h([self.asarray, other.asarray]))
                        return self._memo[(name, other)]
                return g

            setattr(cls, f"__{magicm}__", f(npm or magicm))
        instance = object.__new__(cls)
        return instance

    def __post_init__(self):
        self.n = len(self.points)
        if isinstance(self.points, np.ndarray):
            if len(self.points.shape) < 2 or self.points.shape[1] < 2:
                raise Exception(f"Invalid shape: {self.points.shape}.")
            self.d = self.points.shape[1] if self.n > 0 else None
            self._asarray = self.points
            if self.checkdups and self.n != len(self.asdict):
                raise Exception("Duplicate items in provided array.")
        elif isinstance(self.points, dict):
            if self.n > 0:
                fstkey = next(iter(self.points))
                if isinstance(fstkey, int):
                    self._asdict = {(k,): v for k, v in self.points.items()}
                    fstkey = [None]
                else:
                    self._asdict = self.points
                self.d = len(fstkey) + 1
        elif isinstance(self.points, list):
            # TODO: check invalid key
            self.d = len(self.points[0]) if self.n > 0 else None
            self._asarray = np.array(self.points)
            if self.checkdups and self.n != len(self.asdict):
                raise Exception(f"Duplicate items in provided list.")
        else:
            raise Exception("Unknown type of probing points:", type(self.points))
        if self.d is not None:
            self.lastcoord = self.num2coord(self.d - 1)

    @staticmethod
    def num2coord(i):
        asc = i + 120
        if asc > 122:
            asc -= 26
        return chr(asc)

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

    @property
    @lru_cache
    def sum(self):
        return np.sum(self.target)

    @property
    @lru_cache
    def abs(self):
        return Probing(np.column_stack([self.input, np.abs(self.target).reshape(self.n, 1)]))

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
            other = np.hstack((other[0], other[1]))

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
                raise Exception("Duplicate items in provided array:", other)
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

    # @property
    # def scatter(self):
    #     """Convert to a spatially correct 2D numpy ndarray.
    #
    #     Usage::
    #     >>> points = {
    #     ...     (0.0, 0.1): 0.9,
    #     ...     (0.3, 0.1): 0.0,
    #     ...     (0.3, 0.2): 0.2,
    #     ...     (0.7, 0.4): 0.1,
    #     ...     (0.2, 0.3): 0.3
    #     ... }
    #
    #     >>> probing = Probing(points)
    #     >>> probing.scatter
    #     array([[0.9, nan, nan, nan],
    #            [nan, nan, nan, nan],
    #            [nan, nan, 0.3, nan],
    #            [0. , 0.2, nan, nan],
    #            [nan, nan, nan, nan],
    #            [nan, nan, nan, nan],
    #            [nan, nan, nan, nan],
    #            [nan, nan, nan, 0.1]])
    #
    #     Unknown points will be filled with NaN."""
    #     # TODO: generalize to any number of dimensions
    #     if self._scatter is None:
    #         axes = []
    #         for c in range(self.d - 1):
    #             xs = self.asarray[:, c]
    #             setx = set(xs)
    #             minx = min(setx)
    #             maxx = max(setx)
    #             ordw = sorted(list(setx))
    #             difw = set(abs(numpy.array(ordw[1:] + ordw[0:1]) - numpy.array(ordw)))

    # np.diff

    #             minw = min(difw)
    #             n = int((maxx - minx) / minw)
    #             marks = list(xs) + [x * minw + minx for x in range(n) if all([abs(x * minw + minx - x0) > self.eq_threshold for x0 in xs])]
    #             difw2 = set(abs(numpy.array(marks[1:] + marks[0:1]) - numpy.array(marks)))
    #             minw = min(difw2)
    #             if minw <= self.eq_threshold:
    #                 minw = sorted(difw2)[1]
    #             n = int((maxx - minx) / minw)
    #             marks = list(xs) + [x * minw + minx for x in range(n) if all([abs(x * minw + minx - x0) > self.eq_threshold for x0 in xs])]
    #             axes.append(sorted(marks))
    #         meshgrid = np.meshgrid(*axes)
    #
    #         def f(*key):
    #             return self.asdict[key] if tuple(key) in self.asdict else numpy.nan
    #
    #         fvec = np.vectorize(f)
    #         self._scatter = fvec(*meshgrid)
    #     return self._scatter

    @classmethod
    def fromgrid(cls, *sides, f=0, name=None):
        """A new Probing object containing 'n-1'-D grid points with values of 'n' coordinate given by function 'f'.

        Leave a margin of '1 / (2 * side)' around extreme points.

        2D usage:
            >>> probing = Probing.fromgrid(3, 3)
            >>> print(probing)  # As sequence of zero-valued points.
            [[0.16666667 0.16666667 0.        ]
             [0.5        0.16666667 0.        ]
             [0.83333333 0.16666667 0.        ]
             [0.16666667 0.5        0.        ]
             [0.5        0.5        0.        ]
             [0.83333333 0.5        0.        ]
             [0.16666667 0.83333333 0.        ]
             [0.5        0.83333333 0.        ]
             [0.83333333 0.83333333 0.        ]]

            >>> probing = Probing.fromgrid(3, 3, f=lambda xy: xy[:, 0] * xy[:, 1])
            >>> print(probing)  # As sequence of points.
            [[0.16666667 0.16666667 0.02777778]
             [0.5        0.16666667 0.08333333]
             [0.83333333 0.16666667 0.13888889]
             [0.16666667 0.5        0.08333333]
             [0.5        0.5        0.25      ]
             [0.83333333 0.5        0.41666667]
             [0.16666667 0.83333333 0.13888889]
             [0.5        0.83333333 0.41666667]
             [0.83333333 0.83333333 0.69444444]]

            # >>> probing.show()  # As scatter matrix.
            # [[0.02777778 0.08333333 0.13888889]
            #  [0.08333333 0.25       0.41666667]
            #  [0.13888889 0.41666667 0.69444444]]

        Parameters
        ----------
        sides
            iterable with the dimensions the the grid, or an integer for a 2D sidesXsides grid
        f
            Function providing values for the last coordinate.
        rnd
            Seed.
        name
            Identifier, e.g., for plots.

        """
        axes = [ap[1 / (2 * side), 3 / (2 * side), ..., 1] for side in sides]  # margin = 1 / (2 * side)
        input = np.column_stack([d.ravel() for d in np.meshgrid(*axes)])  # xs,ys
        f_ = (lambda m: np.zeros(len(m))) if isinstance(f, int) else f
        array = np.column_stack([input, f_(input)])  # xs,ys,zs
        return Probing(array)

    @classmethod
    def fromrandom(cls, size, f=0, dims=2, rnd=0, name=None):  # simulated=None,
        """A new Probing object containing 'n-1'-D random points with values of 'n' coordinate given by function 'f'.

        2D usage:
            >>> probing = Probing.fromrandom(2)
            >>> print(probing)  # As sequence of zero-valued points.
            [[0.63696169 0.26978671 0.        ]
             [0.04097352 0.01652764 0.        ]]

            >>> probing = Probing.fromrandom(2, f=lambda xy: xy[:, 0] * xy[:, 1])
            >>> print(probing)  # As sequence of points.
            [[0.63696169 0.26978671 0.1718438 ]
             [0.04097352 0.01652764 0.0006772 ]]

            # >>> probing.show()  # As scatter matrix.
            # [[0.0006772       nan]
            #  [      nan 0.1718438]]

        Parameters
        ----------
        size
            Number of points.
        dims
            Dimension of the space
        f
            Function providing values for the last coordinate.
        rnd
            Seed.
        name
            Identifier, e.g., for plots.
        """
        if isinstance(rnd, int):
            rnd = numpy.random.default_rng(rnd)
        input = rnd.random((size, dims))
        f_ = (lambda m: np.zeros(len(m))) if isinstance(f, int) else f
        array = np.column_stack([input, f_(input)])  # xs,ys,zs
        return Probing(array)

    def __iter__(self):
        yield from self.asarray

    @property
    @lru_cache
    def target(self):
        return getattr(self, self.lastcoord).reshape(self.n)

    @property
    @lru_cache
    def input(self):
        return getattr(self, self.allcoords[:-1])

    @property
    @lru_cache
    def allcoords(self):
        return "".join(map(self.num2coord, range(self.d)))

    @property
    @lru_cache
    def xy_z(self):
        return self.xy, self.z

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
    def __getitem__(self, item):
        if isinstance(item, slice):
            return Probing(self.asarray[item])
        elif isinstance(item, int):
            return self.asarray[item]
        return self.asdict[item]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            raise Exception("Not implemented yet.")
        return self << (key + (value,))

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
    # def show(self):
    #     """Print rounded z values as a scatter matrix"""
    #     with numpy.printoptions(suppress=True, linewidth=1000, precision=8):
    #         print(self.scatter)
    #
    def shuffled(self, rnd=numpy.random.default_rng(0)):
        """
        Usage:
        >>> probing = Probing.fromrandom(4)
        >>> print(probing)
        [[0.63696169 0.26978671 0.        ]
         [0.04097352 0.01652764 0.        ]
         [0.81327024 0.91275558 0.        ]
         [0.60663578 0.72949656 0.        ]]

        >>> print(probing.shuffled())
        [[0.81327024 0.91275558 0.        ]
         [0.63696169 0.26978671 0.        ]
         [0.04097352 0.01652764 0.        ]
         [0.60663578 0.72949656 0.        ]]

        Parameters
        ----------
        rnd

        Returns
        -------

        """
        arr = self.asarray.copy()
        rnd.shuffle(arr)
        return Probing(arr)

    #
    #
    def plot(self, xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), name=None, block=True):
        """

        Usage:
            >>> from seaexp import Seabed
            >>> f = Seabed.fromgaussian()
            >>> g = Seabed.fromgaussian()
            >>> probings = Probing.fromrandom(20, f=f + g)
            >>> probings.plot()

        Returns
        -------

        """
        from seaexp.plotter import Plotter
        plt = Plotter(self.name, xlim, ylim, zlim, inplace=False, block=block)
        name_ = self.name
        self.name = None
        plt << self
        self.name = name_
        # self.plots.append(plt)  # Keeps a reference, so plt destruction (and  window creation) is delayed.

    def __len__(self):
        return self.n

    @property
    def max(self):
        """Return maximum target value."""
        if self._max is None:
            self._max = np.max(self.target)
        return self._max

    @property
    def min(self):
        """Return minimum target value."""
        if self._min is None:
            self._min = np.min(self.target)
        return self._min

    @property
    @lru_cache  # TODO: check if caching/Noning is worth the cost
    def argmax(self):
        """Return list of points with maximum target value."""
        dif = self.max - self.target
        mask = dif <= self.eq_threshold
        return self.input[np.where(mask)]


def cv(probings, k=5, rnd=None):
    """
    Usage:
    >>> probings = Probing({(0, 0): 0, (1, 1): 0, (2, 2): 0, (3, 3): 0, (4, 4): 0, (5, 5): 0})
    >>> probings.xy
    array([[0, 0],
           [1, 1],
           [2, 2],
           [3, 3],
           [4, 4],
           [5, 5]])

    >>> for run, (training, test) in enumerate(cv(probings, k=3)):
    ...     print(f"Run {run}:")
    ...     print("training:\\n", training.xy)
    ...     print("test:\\n", test.xy)
    ...     print("---")
    Run 0:
    training:
     [[5 5]
     [4 4]
     [0 0]
     [1 1]]
    test:
     [[3 3]
     [2 2]]
    ---
    Run 1:
    training:
     [[3 3]
     [2 2]
     [0 0]
     [1 1]]
    test:
     [[5 5]
     [4 4]]
    ---
    Run 2:
    training:
     [[3 3]
     [2 2]
     [5 5]
     [4 4]]
    test:
     [[0 0]
     [1 1]]
    ---

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
    arrshuffled = probings.asarray.copy()
    rnd.shuffle(arrshuffled)
    folds = []

    i = 0
    while i < probings.n:
        fold_size = min_fold_size
        if i < rem:
            fold_size += 1
        folds.append(arrshuffled[i:i + fold_size])
        i += fold_size

    for i in range(k):
        tr = reduce(lambda a, b: np.vstack([a, b]), folds[0:(i + k) % k] + folds[i + 1: k])
        ts = folds[i]
        yield Probing(tr, checkdups=False), Probing(ts, checkdups=False)
