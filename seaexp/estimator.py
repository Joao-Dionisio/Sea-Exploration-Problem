import numpy as np

from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, RBF, Matern, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor

from seaexp.abs.mixin.withPairCheck import withPairCheck


class Estimator(withPairCheck):
    """
    Composable Gaussian Process Regressor

    Parameters
    ----------
    known_points
        for training
    kernel_alias
        available abbreviations: quad, rbf, matern, expsine, white
    params
        sklearn parameters for both kernel and GPR
        available abbreviations: lsb, ab, nlb, restarts
    """

    def __init__(self, known_points, kernel_alias, **params):
        self.known_points, self.kernel_alias, self.params = known_points, kernel_alias, params

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

        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.restarts, copy_X_train=True)
        self.gpr.fit(*known_points.xy_z)
        # self.gpr.fit(np.array(*known_points.xy_z))

    def __add__(self, gpr, restarts=None):
        """Compose the kernels of two estimators to create a new one."""
        if restarts:
            self.params["restarts"] = restarts
        elif self.restarts != gpr.restarts:
            raise Exception(f"Number of restarts should be provided explicitly when both estimators disagree:\n"
                            f"{self.restarts} != {gpr.restarts}.")

        newgpr = Estimator(self.kernel_alias, **self.params)
        newgpr.kernel += gpr.kernel
        return newgpr

    def __call__(self, x_tup, y=None):
        """
        Estimated value at the given point.
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
