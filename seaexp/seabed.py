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
            [[0.63696169 0.26978671 0.78721606]
             [0.04097352 0.01652764 0.99902448]
             [0.81327024 0.91275558 0.47365995]]

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
        z.shape = len(z), 1
        newpoints = np.column_stack([xy, z])
        return replace(probings, points=newpoints) if probings else z

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
            >>> xy = Probing.fromgrid(3)
            >>> print(xy)
            [[0.16666667 0.        ]
             [0.5        0.        ]
             [0.83333333 0.        ]]

            # >>> xy.show()
            # [[0. 0. 0.]
            #  [0. 0. 0.]
            #  [0. 0. 0.]]

            >>> print(f(xy))
            [[0.16666667 0.         9.86207117]
             [0.5        0.         8.82496903]
             [0.83333333 0.         7.06648278]]

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
            >>> real_seabed = Seabed(lambda a, b: a * b)
            >>> training_set = Probing.fromgrid(side=5, f=real_seabed)
            >>> estimated_seabed = GPR("rbf")(training_set)
            >>> diff = Probing.fromgrid(side=5, f=real_seabed - estimated_seabed)
            >>> diff.show()
            [[ 0.00000176 -0.0000035  -0.00000031  0.00000417 -0.00000212]
             [-0.0000035   0.00000768 -0.00000042 -0.00000753  0.00000373]
             [-0.00000031 -0.00000042 -0.00000024  0.00000049  0.00000059]
             [ 0.00000417 -0.00000753  0.00000049  0.00000764 -0.00000488]
             [-0.00000212  0.00000373  0.00000059 -0.00000488  0.00000272]]
            >>> error = diff.abs.sum
            >>> error
            7.551744725834558e-05
            """
        if not isinstance(other, Seabed):
            other = Seabed(other)
        return self + -other

    def __neg__(self):
        return Seabed(lambda x, y: -self(x, y))
