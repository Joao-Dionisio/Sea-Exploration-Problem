from aux import evalu_var


def custom_distortion(trip, TSxy, nb, distortionf, max_failures=9999999):
    """Distort one city at a time. Allow temporary max_failures setbacks."""
    print('# Distorting one city at a time...')
    trip_var_min = sum(trip.stds_simulated(TSxy))
    trip.store3()
    failures = 0
    for b in range(0, nb):
        trip.distort1b(distortionf)
        trip.calculate_tour()
        if trip.feasible: trip_var = sum(trip.stds_simulated(TSxy))
        if trip.feasible and trip_var < trip_var_min:
            failures = 0
            trip_var_min = trip_var
            trip.store3()
        else:
            failures += 1
            if failures > max_failures: break
            trip.restore3()
    trip.restore3()
    return trip_var_min


def custom_distortion2(trip, TSxy, nb, max_failures, distortionf):
    """Distort one city at a time. Maximize variance over the tour."""
    print('# Distorting one city at a time...')
    trip_var_max = evalu_var(trip, trip.xys)
    trip.store3()
    failures = 0
    for b in range(0, nb):
        trip.distort1(distortionf)
        trip.calculate_tour()
        if trip.feasible: trip_var = evalu_var(trip, trip.xys)
        if trip.feasible and trip_var > trip_var_max:
            failures = 0
            trip_var_max = trip_var
            trip.store3()
        else:
            failures += 1
            if failures > max_failures: break
            trip.restore3()
    trip.restore3()


def custom_distortion3(trip, TSxy, nb, distortionf):
    """Distort one city at a time. Not greedy."""
    print('# Distorting one city at a time...')
    trip_var_min = sum(trip.stds_simulated(TSxy))
    trip.store3()
    for b in range(0, nb):
        trip.distort1(distortionf)
        trip.calculate_tour()
        if trip.feasible: trip_var = sum(trip.stds_simulated(TSxy))
        if trip.feasible and trip_var < trip_var_min:
            trip_var_min = trip_var
            trip.store3()
    trip.restore3()


def custom_distortion4(trip, TSxy, nb, distortionf, max_failures=9999999):
    """Distort one city at a time. Allow temporary max_failures setbacks. Cheaply, only check tour at the end, backtracking."""
    print('# Finding better variance, without checking tour feasibility...')
    trip_var_min = sum(trip.stds_simulated(TSxy))
    queue = [(trip.xys.copy(), trip_var_min)]
    failures = 0
    for b in range(0, nb):
        trip.distort1b(distortionf)
        trip_var = sum(trip.stds_simulated(TSxy))
        if trip_var < trip_var_min:
            failures = 0
            trip_var_min = trip_var
            queue.append((trip.xys.copy(), trip_var_min))
        else:
            failures += 1
            trip.xys = queue[-1][0]
            if failures > max_failures:
                break

    print('# Selecting best feasible solution...')
    while True:
        trip.xys, trip_var_min = queue.pop()
        trip.calculate_tour()
        if trip.feasible: break

    return trip_var_min
