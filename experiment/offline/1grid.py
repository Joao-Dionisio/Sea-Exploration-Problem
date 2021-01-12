"""
Grid Maximum-Variance Experiment

Select the point of maximum variance on the grid (G on paper) and recalculate variances.
Proceed until no improvement is made. This results in the surface maximum variance point being added to the trip.

Proceed adding points to the trip until the budget (T on paper) is exhausted.
"""
import numpy as np

from seaexp.estimator import Estimator
from seaexp.probings import Probings, cv
from seaexp.seabed import Seabed

seed = 0
rnd = np.random.default_rng(seed)

# Create initial known points and testing set.
sfg = Seabed.fromgaussian
f1, f2, f3 = sfg(0.1, 100, 0.35113, 0.070836), sfg(0.05, 75, 0.48802, 0.28421), sfg(0.1, 50, 0.032971, 0.20382)
f4, f5, f6 = sfg(0.05, 75, 0.52266, 0.19859), sfg(0.15, 25, 0.24493, 0.7871), sfg(0.49985, 5.3545, 0.19934, 0.50696)
# f7, f8 = sfg(0.011509, 80.985, 0.63317, 0.34842), sfg(0.12154, 78.089, 0.97123, 0.63791)
# f9, f10 = sfg(0.10809, 67.355, 0.9706, 0.27782), sfg(0.40523, 0.28157, 0.029318, 32.567)
initial_known_points = Probings.fromgrid(6, f1 + f2 + f3 + f4 + f5)
initial_known_points.show()

# Select kernel+params for estimator.
# ! I opted for k-fold CV because using 'initially_known_points' for both training and test is prone to overfitting.
for training, test in cv(initial_known_points, rnd=rnd):
    estimator = Estimator.fromoptimizer(training, test, seed=seed, verbosity=1, max_evals=10)

print(estimator)

# Select point of maximum variance.


"""














err = -1 * cross_val_score(gpr, xys, zs, scoring='neg_mean_absolute_error', cv=5).mean()
"""
