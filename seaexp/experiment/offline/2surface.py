"""
Surface Maximum-Variance Experiment

Selected the point of maximum variance on the grid (G on paper);
then recursively sample the maximum variance point from a Gaussian distribution set around the previous selected point.
Proceed until no improvement is made. This results in the surface maximum variance point being added to the trip.

Proceed recalculating variances and adding points to the trip until the budget (T on paper) is exhausted.
"""
