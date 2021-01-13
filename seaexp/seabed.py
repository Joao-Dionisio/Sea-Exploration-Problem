import math
from dataclasses import dataclass
from typing import Union

from seaexp import GPR
from seaexp.abs.mixin.withPairCheck import withPairCheck
# if TYPE_CHECKING:
from seaexp.probings import Probings


@dataclass
class Seabed(withPairCheck):
    """Composable 2D function representing the ocean floor."""
    functions: Union[list, callable]

    def __post_init__(self):
        if not isinstance(self.functions, list):
            self.functions = [self.functions]

    def __call__(self, x, y=None):
        """
        True value at the given point, or probings.

        Parameters
        ----------
        x
            x value or a tuple (x, y)
        y
            y value or None
        Returns
        -------
            True value z.
        """
        if y is None:
            if isinstance(x, Probings):
                return x ^ self
            elif isinstance(x, tuple):
                x, y = x
            else:
                raise Exception(f"Missing exactly one of: Probings, tuple or x,y float values.")
        value = 0
        for f in self.functions:
            value += f(x, y)
        return value

    def __add__(self, other):
        """Create a new seabed by adding another."""
        functions_from_other = [other] if not isinstance(other, Seabed) else other.functions

        return Seabed(self.functions + functions_from_other)

    @classmethod
    def fromgaussian(cls, x=0, y=0, z=0, s=1, a=1, center=None):
        """
        Callable Gaussian function.

        Parameters
        ----------
        s

        Returns
        -------

        """
        center = cls._check_pair(x, y, center)
        return Seabed(
            lambda x, y: z + a * math.exp(- ((x - center[0]) / s) ** 2 / 2 - ((y - center[1]) / s) ** 2 / 2)
        )

    def __sub__(self, other):
        """Compose this Seabed object with a callable object (function, Seabed, Estimator, ...)

        Usage:
            >>> real_seabed = Seabed(lambda a, b: a * b)
            >>> training_set = Probings.fromgrid(side=5, f=real_seabed)
            >>> estimated_seabed = GPR("rbf")(training_set)
            >>> diff = Probings.fromgrid(side=5, f=real_seabed - estimated_seabed)
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
