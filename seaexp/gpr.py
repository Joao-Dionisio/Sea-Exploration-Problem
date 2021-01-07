from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, RBF, Matern, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor

from seaexp.abs.mixin.withPairCheck import withPairCheck


class GPR:
    """
    Composable Gaussian Process Regressor

    Parameters
    ----------
    kernel_alias
        available abbreviations: quad, rbf, matern, expsine, white
    params
        sklearn parameters for both kernel and GPR
        available abbreviations: lsb, ab, nlb, restarts
    """

    def __init__(self, kernel_alias, **params):
        self.kernel_alias, self.params = kernel_alias, params

        if "lsb" in self.params:
            self.params["length_scale_bounds"] = self.params.pop("lsb")
        if "ab" in self.params:
            self.params["alpha_bounds"] = self.params.pop("ab")
        if "nlb" in self.params:
            self.params["noise_level_bounds"] = self.params.pop("nlb")
        self.restarts = self.params.pop("restarts") if "restarts" in self.params else 10

        if self.kernel_alias == "quad":
            self.kernel = RationalQuadratic(**self.params)
        elif self.kernel_alias == "rbf":
            self.kernel = RBF(**self.params)
        elif self.kernel_alias == "matern":
            self.kernel = Matern(**self.params)
        elif self.kernel_alias == "expsine":
            self.kernel = ExpSineSquared(**self.params)
        elif self.kernel_alias == "white":
            self.kernel = WhiteKernel(**self.params)
        else:
            raise Exception("Unknown kernel:", self.kernel_alias)
        self.gpr_func = lambda: GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=self.restarts, copy_X_train=True
        )

    @property
    def gpr(self):
        """A new GaussianProcessRegressor object already configured."""
        return self.gpr_func()

    def model(self, probings):
        """Return the induced model according to the provided probings."""
        gpr = self.gpr
        gpr.fit(*probings.xy_z)
        return GPRModel(gpr)


@dataclass
class GPRModel(withPairCheck):
    def __call__(self, x_tup, y=None):
        """
        Estimated value at the given point, list or Probings obj.
        Parameters
        ----------
        x_tup
            x value or a tuple (x, y)
        y
            y value or None
        Returns
        -------
            Estimated value z'.
        """
        if not isinstance(x_tup, (list, )):
            x_tup = [self._check_pair(x_tup, y)]
        elif y:
            raise Exception(f"Cannot provide both x_tup as list and y={y}.")
        return self.gpr.predict(x_tup)
