#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
from sys import argv
from _aux import train_data, test_data, evalu_var, evalu_sum, probe, random_distortion, current_milli_time, median_distortion, fo
from numpy.random import seed, randint, uniform
from functions import *
from plotter import Plotter
from trip import Trip
from swarm import swarm_distortion
from ga import ga_distortion
from custom_distortion import custom_distortion, custom_distortion4
import ast
import time

# Process arguments.
if len(argv) > 1 and argv[1] == "view":
    if len(argv) is not 6:
        print('5 arguments expected: python3 ocean.py view [function number] [grid side] "[list of points]" [file name]\nExample: python3 ocean.py view 5 7 "[(1, 2), (3, 4)]" plot.png')
        print('Alternatively,\nSee file run.sh for a different usage like below.')
        print("9 arguments expected: python3 ocean.py [plot mode] [seed] [time limit] [iteractions] [grid side] [budget] [function number] [algorithm] [on/off line mode]\nSee file run.sh for examples of use.")
        exit(0)
else:
    if len(argv) is not 10:
        print("9 arguments expected: python3 ocean.py [plot mode] [seed] [time limit] [iteractions] [grid side] [budget] [function number] [algorithm] [on/off line mode]\nSee file run.sh for examples of use.")
        print('Alternatively,\n5 arguments expected: python3 ocean.py view [function number] [grid side] "[list of points]" [file name]\nExample: python3 ocean.py view 5 7 "[(1, 2), (3, 4)]" plot.png')
        exit(0)
fidx, sideidx = (7, 5) if argv[1] != "view" else (2, 3)
fnumber = int(argv[fidx])
switcher = {1: f1, 2: f2, 3: f3, 4: f4, 5: f5, 6: f6, 7: f7, 8: f8, 9: f9, 10: f10}
f, side = switcher.get(fnumber), int(argv[sideidx])
(first_xys, first_zs) = train_data(side, f, False)
if argv[1] == "view":
    points = first_xys + ast.literal_eval(argv[4])
    trip = Trip(f, (0, 0), points, probe(f, points), 100, Plotter('surface'), 0)
    trip.select_kernel()
    trip.fit(n_restarts_optimizer=100)
    trip.plot_pred(argv[5])
    exit(0)
plot, seedval, time_limit, nb, budget, alg, online = argv[1].startswith('plot'), int(argv[2]), float(argv[3]), int(argv[4]), int(argv[6]), argv[8], argv[9] == 'on'

# Initial settings.
seed(seedval)
(TSxy, TSz), depot = test_data(f), (-0.00001, -0.00001)
plotter = Plotter('surface') if plot else None

# Create initial model with kernel selected by cross-validation.
trip = Trip(f, depot, first_xys, first_zs, budget, plotter)
print('# selecting kernel...')
trip.select_kernel()  # TOD
print('# fitting...')
trip.fit()

# Main loop. Report stddev and error...: first iteration -> ...using only known points; (conta == 0)
#                                        second iteration -> ...after orienteering; (conta == 1)
#                                        third iteration -> ...after first distortion; (conta == 2)
#                                        next iterations -> ...after probing [only for online mode] and next distortions; (conta > 2)
conta, acctime = 0, 0
trip_var = sum(trip.stds_simulated(TSxy))
if argv[1] == 'plotvar': trip.plotvar = True
while acctime < time_limit * 3600000:
    trip.tour_time, trip.model_time, trip.pred_time = 0, 0, 0
    start = current_milli_time()

    if conta > 0: trip.add_while_possible(trip.add_maxvar_point(TSxy))
    if conta == 1:
        trip_var = sum(trip.stds_simulated(TSxy))
    elif conta > 1:
        if online and conta > 2: trip.probe_next()
        if len(trip.xys) > 0:
            if alg == '1c': trip_var = custom_distortion(trip, TSxy, nb, random_distortion)
            if alg == 'sw': trip_var = swarm_distortion(trip, TSxy, time_limit * 3600000 - acctime - (current_milli_time() - start))
            if alg == 'ga': trip_var = ga_distortion(trip, TSxy)
            if alg == 'a4': trip_var = custom_distortion4(trip, TSxy, nb, random_distortion)
    conta += 1

    # Logging.
    now = current_milli_time()
    print("# Inducing with real data to evaluate error...")
    erroron = -1
    if online:
        erroron = 0
        errors = []
        for i in range(10):
            trip4 = Trip(f, depot, trip.first_xys + trip.fixed_xys, trip.first_zs + probe(f, trip.fixed_xys), trip.budget, plotter, seedval + i)
            trip4.select_kernel()
            trip4.fit(n_restarts_optimizer=100)
            err = evalu_sum(trip4.model, TSxy, TSz)
            erroron += err
            errors.append(err)
        erroron /= 10.0
        if argv[1] == 'plotpred': trip4.plot_pred()

    trip2 = Trip(f, depot, trip.first_xys + trip.fixed_xys + trip.xys, trip.first_zs + probe(f, trip.fixed_xys + trip.xys), trip.budget, plotter, seedval)
    # trip2.select_kernel()
    trip2.fit(trip.kernel, 100)
    if not online and argv[1] == 'plotpred': trip2.plot_pred()
    erroroff = evalu_sum(trip2.model, TSxy, TSz)

    total = now - start
    acctime += total
    other = total - trip.model_time - trip.pred_time - trip.tour_time
    error = erroron if online else erroroff
    print('res:', acctime, fo(trip_var), fo(error), trip.model_time, trip.pred_time, trip.tour_time, other, errors, len(trip.tour), str(trip2.kernel).replace(' ', '_'), fo(erroroff), fo(trip.cost), trip.fixed_xys + trip.xys, trip.tour, sep='\t')

    # Plotting.
    if plot:
        trip.plot_path()
    if len(trip.xys) == 0 and conta > 2: break

print('# res:', trip.first_xys, trip.first_zs, sep='\t')
print('# res:', trip.xys, sep='\t')
print('# res:', trip.cost, trip.tour, sep='\t')
for i in range(10):
    trip2 = Trip(f, depot, trip.first_xys + trip.fixed_xys + trip.xys, trip.first_zs + probe(f, trip.fixed_xys + trip.xys), trip.budget)
    trip2.select_kernel()
    trip2.fit(n_restarts_optimizer=100)
    error = evalu_sum(trip2.model, TSxy, TSz)
    print('# res:', sum(trip.stds_simulated(TSxy)), error, sep='\t')
