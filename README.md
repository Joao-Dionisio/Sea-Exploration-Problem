![test](https://github.com/Joao-Dionisio/Sea-Exploration-Problem/workflows/test/badge.svg)
[![codecov](https://codecov.io/gh/Joao-Dionisio/Sea-Exploration-Problem/branch/main/graph/badge.svg)](https://codecov.io/gh/Joao-Dionisio/Sea-Exploration-Problem)

# surface
Optimization and machine learning for sea bed exploration

<!---

# install
```
sudo apt-get install python3-tk
git clone https://github.com/oceanbed/surface
cd surface
```
Copy gurobi folder (gurobi752 ?) to the current folder, if exact search is needed.


# usage example (linux bash)
Other parameters are set and documented inside the script. 
```
    ./run.sh off 100 1    # on = online (dynamic) mode. 100 = log after 100 iterations*. 1 = time limit of 1h.
    ./run.sh on 2000 9999 # off = offline (static) mode. 2000 = probe and log after 2000 iterations*. 9999 = we hope the program will end by itself.
```
* -> A new iteration starts after 1/3 of failures.
The program will output the total variance value, the total error among other values as a function of time.
The kernel is selected by k-fold CV.

The plot of the path or the predicted surface of functions can be requested at the command line in the first argument
[other arguments are explained inside run.sh]:
```
python3 -u ocean.py plotvar ...
python3 -u ocean.py plotpred ...
python3 -u ocean.py plotpath ...
python3 -u ocean.py dontplot ...
```
--->

# Code usage

The algorithms are in static.py and evaluate.py is the file to run the functions. At the end of evaluate.py, you can see the "mode" variable. Set it to 0 for the original algorithm, 1 for Local Search, 2 for Multimodal and 3 for Multimodal + Local Search. In here, you also have the "grid" variable. Set it to True to have the initial probings on a grid and False to have them uniformly disitributed. 

Note: The exact TSP algorithm requires the Gurobi solver.

