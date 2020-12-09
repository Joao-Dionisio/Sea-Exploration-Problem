# surface
Optimization and machine learning for sea bed exploration

# install
```
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

