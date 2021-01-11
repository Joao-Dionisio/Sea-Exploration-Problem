import math
import numpy as np
from dataclasses import dataclass
from typing import Union, TYPE_CHECKING

from seaexp.abs.mixin.withPairCheck import withPairCheck

# if TYPE_CHECKING:
from seaexp.estimator import Estimator
from seaexp.probings import Probings


@dataclass
class Seabed(withPairCheck):
    """Composable 2D function representing the ocean floor."""
    functions: Union[list, callable]

    def __post_init__(self):
        if not isinstance(self.functions, list):
            self.functions = [self.functions]

    def __call__(self, x_tup, y=None):
        """
        True value at the given point.
        Parameters
        ----------
        x_tup
            x value or a tuple (x, y)
        y
            y value or None
        Returns
        -------
            True value z.
        """
        x, y = self._check_pair(x_tup, y)
        value = 0
        for f in self.functions:
            value += f(x, y)
        return value

    def __add__(self, other):
        """Create a new seabed by adding another."""
        functions_from_other = [other] if isinstance(other, Estimator) else other.functions

        return Seabed(self.functions + functions_from_other)

    @classmethod
    def fromgaussian(cls, sigma=1, ampl=1, xcenter_tup=(0, 0), ycenter=None):
        """
        Callable Gaussian function.

        Parameters
        ----------
        sigma
        ampl
        xcenter_tup
            x center coordinate or a tuple (x, y)
        ycenter
            y center coordinate or None

        Returns
        -------

        """
        center = cls._check_pair(xcenter_tup, ycenter)
        return Seabed(
            lambda x, y: ampl * math.exp(- ((x - center[0]) / sigma) ** 2 / 2 - ((y - center[1]) / sigma) ** 2 / 2)
        )

    def __sub__(self, other):
        """Compose this Seabed object with a callable object (function, Seabed, Estimator, ...)

        Usage:
            >>> np.set_printoptions(precision=10, suppress=True)
            >>> real_seabed = Seabed(lambda a, b: a * b)
            >>> training_set = Probings.fromgrid(side=5, f=real_seabed)
            >>> estimated_seabed = Seabed(Estimator(training_set, "rbf"))
            >>> diff = Probings.fromgrid(side=5, f=real_seabed - estimated_seabed)
            >>> print(diff)
            [[ 0.0000017569 -0.0000034972 -0.0000003108  0.00000417   -0.0000021208]
             [-0.0000034972  0.0000076777 -0.0000004205 -0.0000075265  0.000003735 ]
             [-0.0000003108 -0.0000004206 -0.0000002403  0.0000004872  0.0000005891]
             [ 0.00000417   -0.0000075264  0.0000004872  0.0000076428 -0.0000048833]
             [-0.0000021208  0.000003735   0.0000005891 -0.0000048834  0.0000027188]]
            >>> error = diff.abs.sum
            >>> error
            7.551744725834558e-05
            """
        if isinstance(other, Estimator):
            other = Seabed(other)
        return self + -other

    def __neg__(self):
        return Seabed(lambda x, y: -self(x, y))
