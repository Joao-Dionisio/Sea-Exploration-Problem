"""
Grid Maximum-Variance Experiment

Select the point of maximum variance on the grid (G on paper) and recalculate variances.
Proceed until no improvement is made. This results in the surface maximum variance point being added to the trip.

Proceed adding points to the trip until the budget (T on paper) is exhausted.
"""
from seaexp.estimator import Estimator
from seaexp.probings import Probings
from seaexp.seabed import Seabed
import numpy as np

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

np.set_printoptions(suppress=True, linewidth=3000)
print(initially_known)

# Select the point of maximum variance.
estimator = Estimator(initially_known, "rbf", lsb=(0.08, 100))
simulated_probings = Probings.fromgrid(estimator, 4, simulated=True)  # apenas para teste, usamos 100 no código anterior

print(simulated_probings)

from sklearn.model_selection import cross_val_score

# err = -1 * cross_val_score(gpr, xys, zs, scoring='neg_mean_absolute_error', cv=5).mean()


print("\n\n====================")
print("experimentando hyperopt...")
np.random.seed(0)


# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2


# define a search space
from hyperopt import hp

space = hp.choice('a', [
    ('case 1', 1 + hp.loguniform('c1', 0, 1)),
    ('case 2', hp.uniform('c2', -10, 10))
])

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval

best = fmin(objective, space, algo=tpe.suggest, max_evals=100, rstate=np.random)

print(best)
print(space_eval(space, best))
