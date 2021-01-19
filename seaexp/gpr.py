from functools import lru_cache
from typing import TYPE_CHECKING, Tuple, Union

from seaexp.probing import Probing, cv

if TYPE_CHECKING:
    from seaexp import Seabed

import math
from typing import TYPE_CHECKING

import numpy as np
from hyperopt import fmin, tpe, space_eval, hp
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor as SkGPR
from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic, RBF, Matern, ExpSineSquared
from sklearn.utils._testing import ignore_warnings

if TYPE_CHECKING:
    from seaexp.probing import Probing

import json

mem = {}


def memo(skgpr, id, xy):
    """Memoizer to avoid calling predict twice to get both mean and stdev"""
    key = skgpr, id
    if key not in mem:
        mem[key] = skgpr.predict(xy, return_std=True)
    return mem[key]


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
    Composable Gaussian Processes Regressor

    Parameters
    ----------
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
        self.skgpr_func = lambda: SkGPR(
            kernel=self.kernel,  # + WhiteKernel(noise_level_bounds=(1e-5, 1e-2)),
            n_restarts_optimizer=self.restarts,
            copy_X_train=True,
            random_state=self.seed
        )

    @property
    def skgpr(self):
        """A new (unfit) sklearn GaussianProcessRegressor object already configured."""
        return self.skgpr_func()

    @ignore_warnings(category=ConvergenceWarning)
    def __call__(self, probing, stdev=False):
        """Return two estimators: one for the mean and other for the stdev

        Both estimators shared the same calculations through memoization."""
        skgpr = self.skgpr
        skgpr.fit(*probing.xy_z)
        from seaexp import Seabed
        if stdev:
            return (Seabed(lambda xy: memo(skgpr, id(xy), xy)[0]),
                    Seabed(lambda xy: memo(skgpr, id(xy), xy)[1]))
        else:
            return Seabed(lambda xy: skgpr.predict(xy))

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
    def fromoptimizer(cls, known_points: 'Probing', seed=0, k=3, param_space=None, algo=None, max_evals=10,
                      verbosity=1):
        """
        Hyperopt error minimizer for kernel search

        Usage:
            >>> from seaexp.probing import Probing
            >>> from seaexp.seabed import Seabed
            >>> points = {
            ...     (0.0, 0.1): 0.12,
            ...     (0.2, 0.3): 0.39
            ... }
            >>> known_points = Probing(points)
            >>> estimator = GPR.fromoptimizer(known_points, k=2, verbosity=0)
            >>> print(estimator)  # doctest: +NORMALIZE_WHITESPACE
            GPR(kernel:rbf, seed:0, signal:1, params:{'length_scale_bounds': [0.001, 0.1]})

        Parameters
        ----------
        k
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

        # Set random number generator.
        rnd = np.random.RandomState(seed=0)

        @ignore_warnings(category=ConvergenceWarning)
        def objective(kwargs):
            # Select kernel+params for estimator.
            # ! I opted for k-fold CV because using all known points for both training and test is prone to overfitting.
            gpr = GPR(seed=seed, **kwargs)
            error = 0
            rnd = np.random.RandomState(seed=0)
            for training, test in cv(known_points, k=k, rnd=rnd):
                mean_estimator = gpr(training)
                errors = test - mean_estimator(test)
                error += errors.abs.sum
            error /= k

            if verbosity > 1:
                kernel = kwargs.pop("kernel_alias")
                print(f"{round(error, 1)} \t{kernel} \t{json.dumps(kwargs, sort_keys=True)}")
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

        # Select minimum error config.
        best = fmin(objective, param_space, algo=algo, max_evals=max_evals, rstate=rnd, show_progressbar=verbosity > 0)
        cfg = space_eval(param_space, best)
        if verbosity > 0:
            print("Lowest error:", minerror[0])

        return GPR(seed=seed, **cfg)

    def __str__(self):
        return f"GPR(kernel:{self.kernel_alias}, seed:{self.seed}, signal:{self.signal}, params:{rf(self.params, 6)})"
