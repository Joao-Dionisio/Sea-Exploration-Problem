#!/bin/bash
export seed=0         # seed to start pseudo number generation
export timelimit="$3" # total execution time for the program in hours
export logafter="$2"  # number of iterations of algorithm 1c (1-city-at-a-time distortion/VOCAL paper) between logs
export budget=100     # total time available for the trip (includes probing time)
export mode="$1"      # off (static) or on (dynamic) - line version

for algorithm in `echo 1c sw`; do    # sw = Particle Swarm Optmization (PSO parameters should be choosen inside swarm.py; currently 4000 iterations and pop. size 100)
  for function in `seq 5 10; seq 1 4`; do
    for gridsize in `echo 7 4 10`; do
      echo f$function-$seed-${gridsize}x$gridsize-$budget-$algorithm-$timelimit.$mode.log ...
      time python3 -u ocean.py dontplot $seed $timelimit $logafter $gridsize $budget $function $algorithm $mode 2> >(grep --line-buffered -v gaussian_process | grep --line-buffered -v np.newaxis | grep --line-buffered -v convergence_dict | grep --line-buffered -v warnings.warn)
    done
  done
done

