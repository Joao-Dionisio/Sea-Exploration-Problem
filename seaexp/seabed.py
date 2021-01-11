import math
import numpy as np
from dataclasses import dataclass
from typing import Union

from seaexp.abs.mixin.withPairCheck import withPairCheck
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
        return Seabed(self.functions + other.functions)

    @classmethod
    def fromgaussian(cls, sigma, ampl, xcenter_tup, ycenter=None):
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
        """Subtract one Seabed object from another

        The points don't need to match.

        Usage:
            >>> np.set_printoptions(precision=10, suppress=True)
            >>> real_f = Seabed(lambda a, b: a * b)
            >>> training_set = Probings.fromgrid(real_f, side=5)
            >>> estimated_f = Seabed(Estimator(training_set, "rbf"))
            >>> diff = Probings.fromgrid(real_f - estimated_f, side=5)
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
        return self + -other

    def __neg__(self):
        return Seabed(lambda x, y: -self(x, y))
