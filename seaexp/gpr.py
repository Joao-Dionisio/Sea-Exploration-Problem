from dataclasses import dataclass
from typing import TYPE_CHECKING
from seaexp.probings import Probings

if TYPE_CHECKING:
    from seaexp.seabed import Seabed

import math
from copy import copy
from typing import TYPE_CHECKING

import numpy as np
from hyperopt import fmin, tpe, space_eval, hp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor as SkGPR
from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, RBF, Matern, ExpSineSquared
from sklearn.utils._testing import ignore_warnings

from seaexp.abs.mixin.withPairCheck import withPairCheck

if TYPE_CHECKING:
    from seaexp.probings import Probings
    from seaexp.seabed import Seabed

import json


def rf(o, ndigits=2):
    """Round floats inside collections

    https://stackoverflow.com/a/53798633/9681577
    """
    if isinstance(o, float):
        return round(o, ndigits)
    if isinstance(o, dict):
        return {k: rf(v, ndigits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rf(x, ndigits) for x in o]
    return o


class GPR:
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

    @ignore_warnings(category=ConvergenceWarning)
    def __init__(self, kernel_alias, seed=0, signal=1, **params):
        self.kernel_alias = kernel_alias
        self.seed = seed
        self.signal = signal
        self.params = params

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
        self.skgpr_func = lambda: SkGPR(kernel=self.kernel, n_restarts_optimizer=self.restarts, copy_X_train=True,
                                        random_state=self.seed)

    @property
    def skgpr(self):
        """A new (unfit) sklearn GaussianProcessRegressor object already configured."""
        return self.skgpr_func()

    @ignore_warnings(category=ConvergenceWarning)
    def __call__(self, probings):
        """Return the induced estimator according to the provided probings."""
        return Estimator(self.skgpr, probings)

    def __add__(self, gpr, restarts=None):
        """Compose two kernels to create a new GPR."""
        params = self.params.copy()
        if restarts:
            params["restarts"] = restarts
        elif self.restarts != gpr.restarts:
            raise Exception(f"Number of restarts should be provided explicitly when both estimators disagree:\n"
                            f"{self.restarts} != {gpr.restarts}.")

        estimator = GPR(self.kernel_alias, **params)
        estimator.kernel += gpr.kernel
        return estimator

    @classmethod
    def fromoptimizer(cls, known_points: 'Probings', testing_points: 'Probings', seed=0,
                      param_space=None, algo=None, max_evals=10, verbosity=1):
        """
        Hyperopt error minimizer for kernel search

        Usage:
            # >>> from seaexp.probings import Probings
            # >>> from seaexp.seabed import Seabed
            # >>> points = {
            # ...     (0.0, 0.1): 0.12,
            # ...     (0.2, 0.3): 0.39
            # ... }
            # >>> known_points = Probings(points)
            # >>> testing_points = Probings.fromrandom(f=Seabed.fromgaussian())
            # >>> estimator = Estimator.fromoptimizer(known_points, testing_points)
            # >>> print(estimator)  # doctest: +NORMALIZE_WHITESPACE
            # Estimator(
                points: 2, kernel: matern, seed: 0, signal: 1
                params: {'nu': 1.353602, 'length_scale_bounds': [0.1, 10]}
            )

        Parameters
        ----------
        known_points
        testing_points
        seed
        param_space
        max_evals
        algo
            default = tpe.suggest
        verbosity
            0, 1 or 2

        Returns
        -------

        """
        if algo is None:
            algo = tpe.suggest
        minerror = [math.inf]

        @ignore_warnings(category=ConvergenceWarning)
        def objective(kwargs):
            estimator = GPR(seed=seed, **kwargs)(known_points)
            errors = testing_points - testing_points @ estimator
            error = errors.abs.sum
            if verbosity > 1:
                k = kwargs.pop("kernel_alias")
                print(f"{round(error, 1)} \t{k} \t{json.dumps(kwargs, sort_keys=True)}")
            if error < minerror[0]:
                minerror[0] = error
            return error

        if param_space is None:
            bounds = [(0.00001, 0.001), (0.001, 0.1), (0.1, 10), (10, 1000), (1000, 100000)]
            param_space = hp.choice('kernel', [
                {"kernel_alias": 'quad', "lsb": hp.choice("lsb_qua", bounds), "ab": hp.choice("ab", bounds)},
                {"kernel_alias": 'rbf', "lsb": hp.choice("lsb_rbf", bounds), },
                {"kernel_alias": 'matern', "lsb": hp.choice("lsb_mat", bounds), "nu": hp.uniform("nu", 0.5, 2.5)}
                # ('expsine', hp.loguniform("lsb_l", 0.00001, 1000), hp.loguniform("lsb_l", 0.001, 100000)),
                # ('white', hp.loguniform("lsb_l", 0.00001, 1000), hp.loguniform("lsb_l", 0.001, 100000))
            ])

        # Set random number generator.
        rnd = np.random.RandomState(seed=0)  # rnd = np.random.default_rng(seed)

        # Select minimum error config.
        best = fmin(objective, param_space, algo=algo, max_evals=max_evals, rstate=rnd, show_progressbar=verbosity > 0)
        cfg = space_eval(param_space, best)
        if verbosity > 0:
            print("Lowest error:", minerror[0])

        return GPR(seed=seed, **cfg)

    def __str__(self):
        return f"GPR(\n\t" \
               f"kernel: {self.kernel_alias}, seed: {self.seed}, signal: {self.signal}" \
               f"\n\tparams: {rf(self.params, 6)}\n)"


@dataclass
class Estimator(withPairCheck):
    skgpr: SkGPR
    probings: Probings
    scale: float = 1

    def __post_init__(self):
        self.skgpr.fit(*self.probings.xy_z)

    def __call__(self, x_tup, y=None):
        """
        Estimated value(s) at the given point, list or Probings obj.
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
        if not isinstance(x_tup, (list,)):
            x_tup = [self._check_pair(x_tup, y)]
        elif y:
            raise Exception(f"Cannot provide both x_tup as list and y={y}.")
        return self.scale * float(self.skgpr.predict(x_tup))

    def __neg__(self):
        newestimator = copy(self)
        newestimator.scale = -1
        return newestimator
