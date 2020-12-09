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
'''
TO-DO List:

- Maybe try to do something a bit more interesting with z? Instead of commiting z to what the last point provides, maybe do some sort of average, that still prioritizes 
information collected more recently.

'''
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic
import numpy as np
from random import random


# object for keeping instance data
class INST:
    pass


inst = INST()
inst.t = 1  # time for probing
inst.s = 1  # traveling speed
inst.T = 100  # time limit for a route
inst.x0 = 0  # depot coordinates
inst.y0 = 0  # depot coordinates


# auxiliary function: euclidean distance
def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# find max variance in a grid
# retorna lista ordenada por variância
def max_var(gpr, n):
    # prepare data
    x = np.arange(0, 1.01, 1. / n)
    y = np.arange(0, 1.01, 1. / n)
    X = [(x[i], y[j]) for i in range(n + 1) for j in range(n + 1)]
    z, z_std = gpr.predict(X, return_std=True)
    points = [(x, v) for (v, x) in sorted(zip(zip(z_std, z), X))]
    return points


def search_around_point(x,y,pos,gpr, n=100):
    '''
    Given a point, searches in a small neighborhood of it for points with higher variance. Repeats the search until no improvement can be found. 
    Converges towards local maxima.  

    Improvements:
                - Add momentum to converge more quickly and also, hopefuly, exit local maxima?
    '''

    possible_points = np.random.normal((x,y),0.01,size=(n,2)) # using small variance precisely because we want to search in a small neighbourhood, in this case

    z, z_std = gpr.predict(possible_points,return_std=True)
    
    best_z, best_points = (z_std[0], z[0]), (x,y)
    ######################################
    # Initializing the variables
    improv = True
    while improv == True:
        improv = False
        for index, cur_std in enumerate(z_std):
            cur_z = (z_std[index], z[index])
            cur_coor = (possible_points[index][0], possible_points[index][1])
            if cur_std > best_z[0]:
                if all(i >= 0 and i <= 1 for i in cur_coor):
                    best_z = cur_z
                    best_points = possible_points[index]
                    improv = True

        if improv == True:
            #possible_points = [best_points]+[(best_points[0] + (2*random()-1)/100, best_points[1] + (2*random()-1)/100) for _ in range(n)] # uniform yields worse results
            possible_points = np.random.normal(best_points, 0.01, (n,2))
            z, z_std = gpr.predict(possible_points,return_std=True)
    #######################################
    return (best_points, best_z)

    

def get_random_points(points,n):
    '''
    Returns a n random points following a moltimodal distribution, where the peaks are the provided points, proportional to their variance
    '''
    # I think this is essentially launching a swarm 
    # I think this is taking a considerable ammount of time, we need to see how we can speed this up
    z_std = [i[1][0] for i in points]
    norm = sum(z_std)
    z_std = [i/norm for i in z_std] # Look up how to do this in numpy, probably faster
    
    #norm = np.linalg.norm(z_std) # <- this is faster
    #z_std/=norm

    fst = True
    for _ in range(n):
        x = random()
        for i, v in enumerate(z_std): #(it's ordered)
            x-=v
            if x < 0:
                coor = points[i][0]
                chosen_z_std = v
                if fst:
                    random_points = np.random.normal(coor,chosen_z_std,size=(1,2))
                    #random_points = np.random.normal(coor,0.1,size=(1,2))                    
                break 
        if not fst:
            random_points = np.append(random_points,np.random.normal(coor,chosen_z_std, size=(1,2)),axis=0)
            #random_points = np.append(random_points,np.random.normal(coor,0.1, size=(1,2)),axis=0)
        else:
            fst = False
    
    
    # chosen_normals = np.random.choice(np.arange(len(points)), size=(1,2,n), p=z_std)
    # chosen_means = [points[i] for i in chosen_normals]
    # chosen_std = [std[i] for i in chosen_normals]
    # random_points = np.random.normal(mean = chosen_means, sd = chosen_std, size=(1,2,n))
    
    #chosen_normals = np.random.choice(np.arange(len(points)), size=n, p=z_std)
    #chosen_means = [points[i][0] for i in chosen_normals]
    #chosen_std = [2*[points[i][1][0]] for i in chosen_normals] 
    #random_points = np.random.normal(loc = chosen_means, scale = chosen_std, size=(n,2))
    
    return random_points



def get_next_point(points, pos, gpr, local_search, n=100):
    '''
    Main function for multimodal search. Keeps lauching random points and picking the best until no improvement can be found.
    '''
    # We need to decide what the scalars will be - for now we'll go with the standard deviation
    first = True
    improv = True
    while improv:     
        random_points = get_random_points(points,n) # Right now this is just repeating, not particularly smart. Take advantage of the improvements. Maybe a priority queue, that prioritases chosen elements?
        #random_points = [search_around_point(*i,pos,gpr,n=10) for i in random_points] # we now have swarm, I think. we'll have to try it later, but I think it's too expensive
        z, z_std = gpr.predict(random_points,return_std=True)
        if first:
            best_z, best_points = (z_std[0], z[0]), random_points[0]
            first = False
        improv = False
        for index, cur_std in enumerate(z_std):
            cur_z = (z_std[index], z[index])
            cur_coor = (random_points[index][0], random_points[index][1])
            if cur_std > best_z[0]:
               # if cur_std == best_z[0]: # In the case of a tie, we're introducing randomness to decide the outcome
               #     if random() > 0.5:   # Maybe we want randomness even if we have a better point. By improving every time we're just doing greedy
               #         continue
                if (0,0) < cur_coor and cur_coor < (1,1) and cur_coor not in pos:
                    best_z = cur_z
                    best_points = random_points[index]
                    improv = True
   
    if local_search:
        return search_around_point(*best_points,pos,gpr) # This will get better results quickly, but won't benefict as much from the randomness. Just improves greedy
    else:
        return (best_points, best_z) 





from tsp import solve_tsp, sequence  # exact
from tsp import multistart_localsearch  # heuristic
# required function: route planning
def planner(X, z, f, mode, dynam=False):
    """planner: decide list of points to visit based on:
        - X: list of coordinates [(x1,y1), ...., (xN,yN)]
        - z: list of evaluations of "true" function [z1, ..., zN]
        - f: useless in static version
    """
    X = list(X)  # work with local copies
    z = list(z)
    l = len(X)
    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    #gpr.fit(X,z) <- we should be fitting but for some reason it often worsens the solution 
    # # plot preliminary GP
    # from functions import plot
    # from functions import f1
    # plot(f1,100)
    # # end of plot
    # # # plot posteriori GP
    # GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # GPR.fit(X, z)
    # def GP(x_,y_):
    #     return GPR.predict([(x_,y_)])[0]
    # plot(GP,100)
    # # # end of plot

    x0, y0 = (inst.x0, inst.y0)  # initial position (depot)
    pos = [(x0, y0)]
    N = 1
    c = {}

    randomize = False if mode == 0 else True
    local_search = False if mode % 2 == 0 else True
    multimodal = False if mode < 2 else True   
    
    
    while True:
        # retorna lista ordenada por variância
        points = max_var(gpr, n=100)  # n x n grid !!!!!
        z_std = [i[1][0] for i in points]
        if randomize: 
            if not multimodal:
                (x, y), (z_new_std, z_new) = points.pop()
                while (x, y) in pos: 
                    (x, y), (z_new_std, z_new) = points.pop()
                (x,y), (z_new_std, z_new) = search_around_point(x,y,pos,gpr, n=100)
            else:
                (x,y), (z_new_std, z_new) = get_next_point(points, pos, gpr, local_search, n=100)
            #(x,y), (z_new_std, z_new) = find_next_candidate_point(pos, gpr, 0.01, 2)
        else:
            # pega ponto com maior variância
            (x, y), (z_new_std, z_new) = points.pop()

            # descarta previamente já selecionados, passando para a próxima maior variância
            while (x, y) in pos: 
                (x, y), (z_new_std, z_new) = points.pop()

            
        
        print('Trying to probe point nº ', N)
        # estende matriz de distâncias para ambos solvers
        for i in range(N):
            c[i, N] = dist(*pos[i], x, y)/inst.s # We divide by the speed to reflect the time needed to travel
            c[N, i] = c[i, N]

        # evita TSP em menos de 3 'cidades', falta checar limite de tempo
        if N < 3:
            N += 1
            pos.append((x, y)) 
            X.append((x,y))
            z.append(z_new)
            #gpr.fit(X,z)
            continue  # !!!!!

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
                continue
            print("heuristic and exact solution exceeds time limit")

            # print("testing next interesting points")

            print("break")
            break

    #print("Found original points. Checking them")
    #print()
    #result = check_chosen_points(X, pos, l, gpr, z, c, sol, N, n=100)
    #pos, sol = result[0], result[1]
    
    print(cost + (N) * inst.t, "TSP solution:", cost, N, inst.T, sol)
    return [pos[i] for i in sol[1:]]


# required function: route planning with probing
def explorer(X, z, plan):
    """planner: decide list of points to visit based on:
        - X: list of coordinates [(x1,y1), ...., (xN,yN)]
        - z: list of evaluations of "true" function [z1, ..., zN]
        - plan: list of coordinates of initial probing plan [(x1,y1), ...., (xP,yP)]
                this plan may be changed; the only important movement is (x1,y1)
    """
    X = list(X)  # work with local copies
    z = list(z)
    from tsp import solve_tsp, sequence
    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

    x0, y0 = (inst.x0, inst.y0)  # initial position (depot)
    x1, y1 = plan[0]
    pos = [(x0, y0), (x1, y1)]
    N = 2
    c = {}
    c[0, 1] = dist(*pos[0], *pos[1])
    while True:
        # attempt adding Nth "city" to the tour
        points = max_var(gpr, n=100)  # n x n grid !!!!!
        (x, y), (z_new_std, z_new) = points.pop()
        for i in range(N):
            c[i, N] = dist(*pos[i], *(x, y))

        if N < 3:
            N += 1
            pos.append((x, y))
            continue  # !!!!!
        obj, edges = solve_tsp(range(N + 1), c)
        if obj <= inst.T - (N) * inst.t:
            sol = sequence(range(N + 1), edges)
            print("TSP solution:", obj, N, inst.T, sol)
            print("appending", (x, y), z_new_std, z_new)
            N += 1
            pos.append((x, y))
            X.append((x, y))
            z.append(z_new)  # !!!!! with average of the prediction as an estimate
            gpr.fit(X, z)
        else:
            # maybe some other elements of 'points' should be attempted here
            break

    return [pos[i] for i in sol[1:]]


def estimator(X, z, mesh):
    """estimator: treina em X, z e retorna valores preditos para os pontos no mesh
        evaluate z at points [(x1,y1), ...., (xK,yK)] based on M=n+N known points:
        - X: list of coordinates [(x1,y1), ...., (xM,yM)]
        - z: list of evaluations of "true" function [z1, ..., zM]
    """
    # from tools import trainGP
    # gpr = trainGP(X, z)

    kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
    gpr.fit(X, z)

    z = []
    for (x_, y_) in mesh:
        val = gpr.predict([(x_, y_)])
        z.append(val)

    return z
