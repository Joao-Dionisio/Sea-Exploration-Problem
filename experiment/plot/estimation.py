"""
Some plots of means and stdevs for a selected kernel
"""
import numpy as np

from seaexp import GPR
from seaexp.plotter import Plotter
from seaexp.probings import Probings
from seaexp.seabed import Seabed

seed = 0
rnd = np.random.default_rng(seed)

with Plotter(zlim=(0, 100)) as plt:
    # Create (continuous and discrete) seabed true function.
    gaus = Seabed.fromgaussian
    f1, f2 = gaus(0.35113, 0.070836, s=0.1, a=100), gaus(0.48802, 0.28421, s=0.05, a=75)
    f3, f4 = gaus(0.032971, 0.20382, s=0.1, a=50), gaus(0.52266, 0.19859, s=0.05, a=75)
    f5, f6 = gaus(0.24493, 0.7871, s=0.15, a=25), gaus(0.19934, 0.50696, s=0.49985, a=5.3545)
    f7, f8 = gaus(0.63317, 0.34842, s=0.011509, a=80.985), gaus(0.97123, 0.63791, s=0.12154, a=78.089)
    f9, f10 = gaus(0.9706, 0.27782, s=0.10809, a=67.355), gaus(0.40523, 0.28157, s=0.029318, a=32.567)
    true_f = f1 + f2 + f3 + f4 + f5  # + f6 + f7 + f8 + f9 + f10)
    true_discrete = Probings.fromgrid(side=33, f=true_f, name="true")
    plt << true_discrete

    # Known points from past trips.
    initially_known = Probings.fromgrid(side=6, f=true_f, name="known")
    plt << initially_known

    # Select kernel+params for estimator.
    gpr = GPR.fromoptimizer(initially_known, seed=seed, verbosity=2, max_evals=10)
    print(f"Selected kernel/config: {gpr}\n")

    # Select point of maximum variance.
    mean_estimator, std_estimator = gpr(initially_known, stdev=True)
    candidates = Probings.fromgrid(side=10)  # create a zeroed grid, and replace the zeros by variances (z=std)
    stds = std_estimator(candidates)
    stds.name = "stdev"
    plt(zlim=(0, 2), color="gray") << stds
    print(f"Maximum stdev: {stds.max} ; min: {stds.min}")

    # Plot means out of curiosity
    means = mean_estimator(candidates)
    means.name = "mean"
    plt << means
