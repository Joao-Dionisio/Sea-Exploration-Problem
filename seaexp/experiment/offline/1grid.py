"""
Grid Maximum-Variance Experiment

Select the point of maximum variance on the grid (G on paper) and recalculate variances.
Proceed until no improvement is made. This results in the surface maximum variance point being added to the trip.

Proceed adding points to the trip until the budget (T on paper) is exhausted.
"""
import numpy as np

from seaexp.estimator import Estimator
from seaexp.probings import Probings
from seaexp.seabed import Seabed

seed = 0

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
initially_known_points = Probings.fromgrid(4, seabed1)
testing_points = Probings.fromgrid(10, seabed1)

# Tweak for printing numpy arrays.
np.set_printoptions(suppress=True, linewidth=3000)
print(initially_known_points)

# Select kernel+params for estimator.
estimator = Estimator.fromoptimization(initially_known_points, testing_points, seed, verbose=True, max_evals=100)

print(estimator)
# err = -1 * cross_val_score(gpr, xys, zs, scoring='neg_mean_absolute_error', cv=5).mean()
