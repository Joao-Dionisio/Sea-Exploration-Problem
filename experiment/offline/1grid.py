"""
Grid Maximum-Variance Experiment

[   lembrete: talvez esse exp. ainda seja pré-artigo, pois me parece que o primeiro
    algoritmo do novo paper já fala em amostrar de uma gaussiana sobre o ponto do grid. ]

Select the point of maximum variance on the grid (G on paper),
adopt estimated value as a real probing, refit GPR and recalculate variances.
This results in the grid maximum variance point being added to the trip.

Proceed adding points to the trip until the budget (T on paper) is exhausted.
"""
import numpy as np

from seaexp import GPR, Trip
from seaexp.plotter import Plotter
from seaexp.probing import Probing
from seaexp.seabed import Seabed
from seaexp.trip import BudgetExhausted

seed = 0
rnd = np.random.default_rng(seed)

with Plotter(zlim=(0, 100), inplace=True) as plt:
    # Create (continuous and discrete) seabed true function.
    gaus = Seabed.fromgaussian
    f1, f2 = gaus(0.35113, 0.070836, s=0.1, a=100), gaus(0.48802, 0.28421, s=0.05, a=75)
    f3, f4 = gaus(0.032971, 0.20382, s=0.1, a=50), gaus(0.52266, 0.19859, s=0.05, a=75)
    f5, f6 = gaus(0.24493, 0.7871, s=0.15, a=25), gaus(0.19934, 0.50696, s=0.49985, a=5.3545)
    f7, f8 = gaus(0.63317, 0.34842, s=0.011509, a=80.985), gaus(0.97123, 0.63791, s=0.12154, a=78.089)
    f9, f10 = gaus(0.9706, 0.27782, s=0.10809, a=67.355), gaus(0.40523, 0.28157, s=0.029318, a=32.567)
    true_f = f1 + f2 + f3 + f4 + f5  # + f6 + f7 + f8 + f9 + f10)
    true_discrete = Probing.fromgrid(33, 33, f=true_f, name="true")
    plt << true_discrete

    # Known points from past trips.
    known = Probing.fromgrid(3, 3, f=true_f, name="known")
    plt << known

    # Select kernel+params for estimator.
    gpr = GPR.fromoptimizer(known, seed=seed, verbosity=2, max_evals=10)
    print(f"Selected kernel/config: {gpr}\n")

    # Loop
    current = known
    trip = Trip()
    feasible = True
    while feasible:
        # Fit GPR and get both functions from estimator: mean and std.
        fmean, fstd = gpr(current, stdev=True)

        # Create a zeroed grid, and replace the zeros by variances (z=stdev).
        candidates = Probing.fromgrid(20, 20)
        stds = fstd(candidates)
        stds.name = "stdev"
        plt(zlim=(0, 2), color="gray", delay=0) << stds

        # Add newpoint of maximum variance (with simulated z value) to the extended set (training points).
        maxs = stds.argmax
        newpoint = np.array([rnd.choice(maxs)])
        extended = current << (newpoint, fmean(newpoint))
        print(f"Maxpoints {maxs}\n\tmaxstd: {stds.max}\tminstd: {stds.min}\tchosen: {newpoint}\tnew size: {extended.n}")

        # Try to add newpoint to the trip in three ways: raw, heuristic #and exhaustive.
        bkp = trip  # Save state before insertion attempt.
        feasible = False
        try:
            trip <<= newpoint
            feasible = True
            print(f"Successful insertion:\n\tsize: {trip.n}\tcost: {trip.cost}\ttrip:{trip}")
        except BudgetExhausted as e:
            print(e)
            # try:
            # except Exception as e:
            #     print(e)
            # Confirm or revert insertion according to feasibility.
        if feasible:
            current = extended
        else:
            trip = bkp

    # trip.plot()

"""














err = -1 * cross_val_score(gpr, xys, zs, scoring='neg_mean_absolute_error', cv=5).mean()
"""
