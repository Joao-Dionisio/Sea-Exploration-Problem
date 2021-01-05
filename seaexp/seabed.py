import math
from dataclasses import dataclass

from seaexp.abs.mixin.withPairCheck import withPairCheck


@dataclass
class Seabed(withPairCheck):
    """Composable 2D function representing the ocean floor."""
    function: callable

    def __post_init__(self):
        self._functions = [self.function]

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
        for f in self._functions:
            value += f(x, y)
        return value

    def __add__(self, seabed):
        """Create a new seabed by adding another."""
        newseabed = Seabed(self.function)
        newseabed._functions.append(seabed.function)
        return newseabed

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
