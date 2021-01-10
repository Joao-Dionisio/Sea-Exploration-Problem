import math
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
            >>> real_f = Seabed(lambda a, b: a * b)
            >>> training_set = Probings.fromgrid(real_f, side=5)
            >>> estimated_f = Seabed(Estimator(training_set, "rbf"))
            >>> diff = Probings.fromgrid(real_f - estimated_f, side=5)
            >>> print(diff)
            [[ 1.75689071e-06 -3.49719514e-06 -3.10842734e-07  4.16997529e-06
              -2.12081810e-06]
             [-3.49718059e-06  7.67765581e-06 -4.20537253e-07 -7.52645341e-06
               3.73498871e-06]
             [-3.10824544e-07 -4.20566357e-07 -2.40339432e-07  4.87225043e-07
               5.89093543e-07]
             [ 4.16996802e-06 -7.52644613e-06  4.87228681e-07  7.64279524e-06
              -4.88333273e-06]
             [-2.12081810e-06  3.73497416e-06  5.89104457e-07 -4.88338730e-06
               2.71880577e-06]]
            >>> error = diff.abs.sum
            >>> error
            7.551744725834558e-05
            """
        return self + -other

    def __neg__(self):
        return Seabed(lambda x, y: -self(x, y))
