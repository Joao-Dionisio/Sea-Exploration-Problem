from aux_functions import dist, remove_from_dist_matrix, update_dist_matrix 


def candidate_point_is_better(pos, sol, point_index, cur_point, candidate_point):
    """Given a solution to the problem, the point we're currently considering, and a candidate point, heuristically determines which point will lead to a 
    smaller TSP tour"""

    # we don't update the distance matrix, need to do it outside function
    from copy import copy
    #removed_point = sol[point_index]
    removed_point = sol.index(point_index)
    chosen_index = removed_point
    
    cur_coor = cur_point
    candidate_coor = candidate_point
    l_sol = copy(sol) # have to work with local copies, because pop will affect the variables even outside the function
    l_pos = copy(pos)
    l_sol.pop(removed_point)
    ##l_sol.remove(point_index)
    l_pos.pop(point_index)

    prev_point = pos[sol[removed_point-1]]
    next_point = pos[sol[(removed_point+1)%len(sol)]]
    cur_dist = dist(*prev_point, *cur_coor) + dist(*cur_coor, *next_point) # We can't get rid of sol because of this - we need to know the order in order to know the distances 
    old_dist = cur_dist
    for i in range(len(l_sol)):
        candidate_distance = dist(*pos[l_sol[i]], *candidate_coor)
        if candidate_distance < cur_dist:
            candidate_distance += dist(*candidate_coor, *pos[l_sol[(i+1)%len(l_sol)]])
            if candidate_distance < cur_dist:
                cur_dist = candidate_distance
                cur_coor = candidate_coor
                chosen_index = i+1 # When we get to this point, it means that the best arrangement (until now) is i -> candidate_point -> i+1. So it should take i+1's place and push the other forward

    l_sol = l_sol[:chosen_index] + [point_index] + l_sol[chosen_index:]
    l_pos = l_pos[:point_index] + [tuple(cur_coor)] + l_pos[point_index:]

    # sol[chosen_index] = point_index
    # pos[point_index] = cur_coor
    # we are getting points that are too close

    return l_sol, l_pos, cur_dist - old_dist


def check_chosen_points(X, pos, l, gpr, z, c, sol, N, n): # sol needs to be here, trust
    '''
    - Implementation of a sort of Simmulated Annealing, maybe? Tries to move the points at the end of the tour in order to
    both improve the colective variance and to reduce the distance of the tour, so that we can maybe fit another point

    - Takes what would be a solution to the problem and point by point searches around it for a more appropriate point.

    Improvements:
            Save the original tour, so that if we can't add more points we return it instead. Points that are closer together
            are generally worse and are only worth if we manage to fit more of them. 
            When this is the case, that we can't fit another point, maybe try the other way around and try to pick points that 
            are farther away, while also not breaking the time limit?
            We are going around only once, we can probably do it more times, but we'll need to be careful as it can take quite some 
            time, almost as long as constructing the original tour.
    '''
    from copy import copy
    c = remove_from_dist_matrix(c, len(pos)) # we need to remove the last point (that isn't added) from the dist matrix 
    improvement = True
    first_time = True
    old_pos = copy(pos) # if we can't fit another point, just return what we got initially?
    old_sol = copy(sol)
    managed_to_fit_another_point = False
    while improvement:
    # While n_improvements > 0: <- this would take a long ass time
    #for _ in range(1):
        improvement = False
        n_improvement = 0
        cost, edges = solve_tsp(range(N), c)
        sol = sequence(range(N), edges)
        for index, cur_point in enumerate(X[l:]): # not entire pos. Here we're going through the points we chose and try to "improve" them
            print("checking point nr ", index+1)
            improvement, *best_point, c, pos, sol, dist_dif = try_to_reduce_TSP(pos, sol, index+1, cur_point, gpr, c, n=100) # we use index+1 because pos has the origin and X does not 
            n_improvement+=improvement # True -> 1 ; False -> 0
            X[index+l] = tuple(best_point[0])
            z[index+l] = float(best_point[1][1]) # Coverting from numpy array to tuple/float to avoid annoyances
            cost += dist_dif
            print("Decreased by roughly ", dist_dif) # we say roughly because we need to TSP again to make sure
            gpr.fit(X, z) # Updating the landscape, since we commited to another point (or not, the variables may not change, if it's not worth it)

        # we're getting a negative cost, so the loop never ends
        if cost < 0:
            print('assa')
            pass
        # Trying to fit another point
        points = max_var(gpr, n=100)
        (x,y), (z_new_std, z_new) = get_next_point(points, pos, gpr, n=100)
        temp_c = copy(c) # We need a temporary distance matrix because we might not be able to fit another point, and then c would go to the next iteration with an extra 
        for i in range(N): # point, causing the other functions that loop through c to break
            temp_c[i, N] = dist(*pos[i], x, y)/inst.s 
            temp_c[N, i] = temp_c[i, N]
    
        prev_cost = cost
        cost, edges = solve_tsp(range(N), temp_c) # moving past heuristic # need to be careful 
        print("Managed to decrease TSP by ", cost - prev_cost)
        if cost <= inst.T - (N) * inst.t:
            first_time = True
            c = temp_c
            try:
                sol = sequence(range(N + 1), edges)
            except:
                return (old_pos, old_sol) # sometimes we get an error, for some reason. Getting tour length infeasible when we reach here
            N += 1
            pos.append((x, y))
            X.append((x, y))
            z.append(z_new)  
            gpr.fit(X, z)
            managed_to_fit_another_point = True
            print("Managed to fit another point")
        if improvement:
            print("Found ", n_improvement, " improvements, continuing search")
        
    return (pos, sol) # if managed_to_fit_another_point else (old_pos, old_sol)
