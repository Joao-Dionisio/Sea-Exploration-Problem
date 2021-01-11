import math
from copy import copy
from typing import TYPE_CHECKING

import numpy as np
from hyperopt import fmin, tpe, space_eval, hp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
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

    def __init__(self, known_points, kernel_alias, seed=0, signal=1, **params):
        self.known_points = known_points
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

        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=self.restarts, copy_X_train=True, random_state=self.seed
        )
        self.gpr.fit(*known_points.xy_z)
        # self.gpr.fit(np.array(*known_points.xy_z))

    def __add__(self, gpr, restarts=None):
        """Compose the kernels of two estimators to create a new one."""
        if restarts:
            self.params["restarts"] = restarts
        elif self.restarts != gpr.restarts:
            raise Exception(f"Number of restarts should be provided explicitly when both estimators disagree:\n"
                            f"{self.restarts} != {gpr.restarts}.")

        estimator = Estimator(self.kernel_alias, **self.params)
        estimator.kernel += gpr.kernel
        return estimator

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
        if not isinstance(x_tup, (list,)):  # TODO: accept Probings?
            x_tup = [self._check_pair(x_tup, y)]
        elif y:
            raise Exception(f"Cannot provide both x_tup as list and y={y}.")
        return self.signal * float(self.gpr.predict(x_tup))

    @classmethod
    def fromoptimization(cls,
                         known_points: 'Probings', testing_points: 'Probings', seed=0,
                         param_space=None, algo=tpe.suggest, max_evals=10, verbose=False):
        """
        Usage:
            >>> from seaexp.probings import Probings
            >>> from seaexp.seabed import Seabed
            >>> points = {
            ...     (0.0, 0.1): 0.12,
            ...     (0.2, 0.3): 0.39
            ... }
            >>> known_points = Probings(points)
            >>> testing_points = Probings.fromrandom(f=Seabed.fromgaussian())
            >>> estimator = Estimator.fromoptimization(known_points, testing_points)
            >>> print(estimator)  # doctest: +NORMALIZE_WHITESPACE
            Estimator(
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
        verbose

        Returns
        -------

        """
        from seaexp.probings import Probings
        minerror = [math.inf]

        @ignore_warnings(category=ConvergenceWarning)
        def objective(kwargs):
            estimator = Estimator(known_points, seed=seed, **kwargs)
            errors = testing_points - testing_points @ estimator
            error = errors.abs.sum
            if verbose:
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
        best = fmin(objective, param_space, algo=algo, max_evals=max_evals, rstate=rnd, show_progressbar=verbose)
        cfg = space_eval(param_space, best)
        if verbose:
            print("Lowest error:", minerror[0])

        return Estimator(known_points, seed=seed, **cfg)

    def __neg__(self):
        newestimator = copy(self)
        newestimator.signal = -1
        return newestimator

    def __str__(self):
        return f"Estimator(\n\t" \
               f"points: {self.known_points.n}, kernel: {self.kernel_alias}, seed: {self.seed}, signal: {self.signal}" \
               f"\n\tparams: {rf(self.params, 6)}\n)"
