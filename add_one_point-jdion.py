# Function to add just one point 


# required function: route planning
def get_new_pointç(X, z, f, pos, N, randomize, dynam=False):
    """planner: decide list of points to visit based on:
        - X: list of coordinates [(x1,y1), ...., (xN,yN)]
        - z: list of evaluations of "true" function [z1, ..., zN]
        - f: useless in static version
        - pos: previous route that we're trying to extend
        - gpr: gaussian process regressor that we'll fit now
    """
    X = list(X)  # work with local copies
    z = list(z)
    l = len(X)
    from tsp import solve_tsp, sequence  # exact
    from tsp import multistart_localsearch  # heuristic

    x0, y0 = (inst.x0, inst.y0)  # initial position (depot)
    pos = [(x0, y0)]
    N = 1
    c = {}

    # retorna lista ordenada por variância
    points = max_var(gpr, n=100)  # n x n grid !!!!!

    # pega ponto com maior variância
    (x, y), (z_new_std, z_new) = points.pop()

    # descarta previamente já selecionados, passando para a próxima maior variância
    while (x, y) in pos:
        (x, y), (z_new_std, z_new) = points.pop()

    # Now that we have the candidate cell, we can do the random search
    if randomize:
        (x,y), (z_new_std, z_new) = search_around_point(x,y,pos,gpr)
    print('found point nº ', N)
  

    # evita TSP em menos de 3 'cidades', falta checar limite de tempo
    if N < 3:
        pos.append((x, y)) # be careful with this, for some reason it might be worsening the solution
        X.append((x,y))
        z.append(z_new)
        return [(x,y), z_new]
    else:
        # estende matriz de distâncias para ambos solvers
        for i in range(N):
            c[i, N] = dist(*pos[i], x, y)/inst.s # We divide by the speed to reflect the time needed to travel
            c[N, i] = c[i, N]
        sol_, cost = multistart_localsearch(100, N + 1, c, cutoff=inst.T - (N) * inst.t)  # heuristic
        if cost <= inst.T - (N) * inst.t:
            print("heuristic solution")
            idx = sol_.index(0)
            sol = sol_[idx:] + sol_[:idx]  # heuristic
            # print(obj + (N)*inst.t, "TSP solution:", obj, N, inst.T, sol)
            # print("appending", (x,y), z_new_std, z_new, "orient.len:", obj + (N)*inst.t)
            N += 1
            assert (x, y) not in pos
            pos.append((x, y))
            X.append((x, y))
            if (dynam):
                z.append(f(x, y))  # !!!!! with PROBING
            else:
                z.append(z_new)  # !!!!! with average of the prediction as an estimate
            gpr.fit(X, z)
            return [(x,y), z_new]
        else:
            # attempt exact solution:
            print("attempting exact solution")
            cost, edges = solve_tsp(range(N + 1), c)  # exact
            if cost <= inst.T - (N) * inst.t:
                sol = sequence(range(N + 1), edges)  # exact
                # print(obj + (N) * inst.t, "TSP EXACT:", obj, N, inst.T, sol)
                # print("appending", (x, y), z_new_std, z_new, "orient.len:", obj + (N) * inst.t)
                N += 1
                pos.append((x, y))
                X.append((x, y))
                if (dynam):
                    z.append(f(x, y))  # !!!!! with PROBING
                else:
                    z.append(z_new)  # !!!!! with average of the prediction as an estimate
                gpr.fit(X, z)
                print("found; continue")
                return [(x,y), z_new]
            print("heuristic and exact solution exceeds time limit")

            # print("testing next interesting points")

            print("Couldn't find another point")
            return

    
    
    
