"""
Grid Maximum-Variance Experiment

Select the point of maximum variance on the grid (G on paper) and recalculate variances.
Proceed until no improvement is made. This results in the surface maximum variance point being added to the trip.

Proceed adding points to the trip until the budget (T on paper) is exhausted.
"""
import numpy as np
from hyperopt import fmin, tpe, space_eval, hp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from seaexp.estimator import Estimator
from seaexp.probings import Probings
from seaexp.seabed import Seabed

# Set random number generator.
rnd = np.random.RandomState(seed=0)  # rnd = np.random.default_rng(seed)

# Create initial known points.
seabed1 = (Seabed.fromgaussian(0.1, 100, 0.35113, 0.070836)
           + Seabed.fromgaussian(0.05, 75, 0.48802, 0.28421)
           + Seabed.fromgaussian(0.1, 50, 0.032971, 0.20382)
           + Seabed.fromgaussian(0.05, 75, 0.52266, 0.19859)
           + Seabed.fromgaussian(0.15, 25, 0.24493, 0.7871))
seabed2 = (Seabed.fromgaussian(0.49985, 5.3545, 0.19934, 0.50696)
           + Seabed.fromgaussian(0.011509, 80.985, 0.63317, 0.34842)
           + Seabed.fromgaussian(0.12154, 78.089, 0.97123, 0.63791)
           + Seabed.fromgaussian(0.10809, 67.355, 0.9706, 0.27782)
           + Seabed.fromgaussian(0.40523, 0.28157, 0.029318, 32.567))
initially_known = Probings.fromgrid(seabed1, 4)

# Tweak for printing numpy arrays.
np.set_printoptions(suppress=True, linewidth=3000)
print(initially_known)

# Setup kernel optimization.
bounds = [(0.00001, 0.001), (0.001, 0.1), (0.1, 10), (10, 1000), (1000, 100000)]
space = hp.choice('kernel', [
    {
        "kernel_alias": 'quad',
        "lsb": hp.choice("lsb_qua", bounds),
        "ab": hp.choice("ab", bounds),
    },
    {
        "kernel_alias": 'rbf',
        "lsb": hp.choice("lsb_rbf", bounds),
    },
    {
        "kernel_alias": 'matern',
        "lsb": hp.choice("lsb_mat", bounds),
        "nu": hp.uniform("nu", 0.5, 2.5)
    }
    # ('expsine', hp.loguniform("lsb_l", 0.00001, 1000), hp.loguniform("lsb_l", 0.001, 100000)),
    # ('white', hp.loguniform("lsb_l", 0.00001, 1000), hp.loguniform("lsb_l", 0.001, 100000))
])


@ignore_warnings(category=ConvergenceWarning)
def objective(kwargs):
    estimator = Estimator(initially_known, **kwargs)
    estimated_seabed = Seabed(estimator)
    error = Probings.fromgrid(seabed1 - estimated_seabed, 10).abs.sum
    return error


# Select the point of minimum error.
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, rstate=rnd)

print(space_eval(space, best))

# err = -1 * cross_val_score(gpr, xys, zs, scoring='neg_mean_absolute_error', cv=5).mean()
