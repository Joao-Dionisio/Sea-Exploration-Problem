import numpy as np
import math
from dataclasses import dataclass, replace
from functools import reduce
from typing import Union

from seaexp import GPR
from seaexp.abs.mixin.withPairCheck import withPairCheck
# if TYPE_CHECKING:
from seaexp.probing import Probing


@dataclass
class Seabed(withPairCheck):
    """Composable 2D function representing the ocean floor."""
    functions: Union[list, callable]

    def __post_init__(self):
        if not isinstance(self.functions, list):
            self.functions = [self.functions]

    def __call__(self, xy):
        """
        True value at the point(s) given as a 2d ndarray, or as a probing object.

        Usage:
            >>> from seaexp import Probing
            >>> f = Seabed.fromgaussian()
            >>> print(f(Probing.fromrandom(3)))
            (0.63696 0.26979 0.78722) (0.04097 0.01653 0.99902) (0.81327 0.91276 0.47366)

        Parameters
        ----------
        xy
            2d ndarray or probings object
        Returns
        -------
            True z value(s).
        """
        probings = isinstance(xy, Probing) and xy
        if probings:
            xy = xy.xy
        z = reduce(lambda m, f: m + f(xy), [0] + self.functions)
        newpoints = np.column_stack([xy, z])
        return replace(probings, points=newpoints) if probings else z.reshape(len(z))

    def __add__(self, other):
        """Create a new seabed by adding another."""
        functions_from_other = [other] if not isinstance(other, Seabed) else other.functions

        return Seabed(self.functions + functions_from_other)

    @classmethod
    def fromgaussian(cls, x=0, y=0, z=0, s=1, a=1, center=None):
        """
        Callable Gaussian function.

        Usage:
            >>> f = Seabed.fromgaussian(a=10)
            >>> xy = Probing.fromgrid(3, 3)
            >>> print(xy)
            (0.16667 0.16667 0.0) (0.5 0.16667 0.0) (0.83333 0.16667 0.0) (0.16667 0.5 0.0) (0.5 0.5 0.0) (0.83333 0.5 0.0) (0.16667 0.83333 0.0) (0.5 0.83333 0.0) (0.83333 0.83333 0.0)

            # >>> xy.show()
            # [[0. 0. 0.]
            #  [0. 0. 0.]
            #  [0. 0. 0.]]

            >>> print(f(xy))
            (0.16667 0.16667 9.72604) (0.5 0.16667 8.70325) (0.83333 0.16667 6.96902) (0.16667 0.5 8.70325) (0.5 0.5 7.78801) (0.83333 0.5 6.23615) (0.16667 0.83333 6.96902) (0.5 0.83333 6.23615) (0.83333 0.83333 4.99352)

        Parameters
        ----------
        s
            sigma
        a
            amplitude

        Returns
        -------

        """
        center = cls._check_pair(x, y, center)

        def f(xy):
            r = ((xy - center) / s) ** 2 / 2
            return z + a * np.exp(-r[:, 0] - r[:, 1])

        return Seabed(f)

    def __sub__(self, other):
        """Compose this Seabed object with a callable object (function, Seabed, Estimator, ...)

        Usage:
            >>> real_seabed = Seabed(lambda ab: ab[:, 0] * ab[:, 1])
            >>> training_set = Probing.fromgrid(5, 5, f=real_seabed)
            >>> estimated_seabed = GPR("rbf")(training_set)
            >>> diff = Probing.fromgrid(5, 5, f=real_seabed - estimated_seabed)
            >>> diff.show()
            (0.1 0.1 0.0) (0.3 0.1 -0.0) (0.5 0.1 -0.0) (0.7 0.1 0.0) (0.9 0.1 -0.0) (0.1 0.3 -0.0) (0.3 0.3 1e-05) (0.5 0.3 -0.0) (0.7 0.3 -1e-05) (0.9 0.3 0.0) (0.1 0.5 -0.0) (0.3 0.5 -0.0) (0.5 0.5 -0.0) (0.7 0.5 0.0) (0.9 0.5 0.0) (0.1 0.7 0.0) (0.3 0.7 -1e-05) (0.5 0.7 0.0) (0.7 0.7 1e-05) (0.9 0.7 -0.0) (0.1 0.9 -0.0) (0.3 0.9 0.0) (0.5 0.9 0.0) (0.7 0.9 -0.0) (0.9 0.9 0.0)

            >>> error = diff.abs.sum
            >>> round(error, 5)
            8e-05
            """
        if not isinstance(other, Seabed):
            other = Seabed(other)
        return self + -other

    def __neg__(self):
        return Seabed(lambda ab: -self(ab))
