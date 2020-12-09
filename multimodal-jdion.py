def remove_close_points(points):
    seen = {}
    index = 0
    while index < len(points):
        val = points[index]
        x, y = round(val[0][0]), round(val[0][1]) # we're rounding now, should do something else
        if (x,y) not in seen:
            seen[(x,y)] = 1
        else:
            points.pop(index)
    return points


import numpy as np

def get_next_point(points, pos, grp, n=10):
    # points is a list of points sorted by variance
    # n is the number of points we want to sample
    # We need to decide what the scalars will be - for now we'll go with the standard deviation
    points = remove_close_points(points) # To dilute a bit the more important areas
    random_points = n*[0]
    for i in points:
        coor = points[0]
        z_std = points[1][0] 
        random_points += z_std*np.random.normal(coor,z_std,n) # don't know if the standard deviation should be z_std or 1/z_st
    z, z_std = gpr.predict(random_points,return_std=True)

    improv = True
    while improv:
        improv = False
        for index, cur_std in enumerate(z_std):
            cur_z = (z_std[index], z[index])
            cur_coor = (random_points[index][0], random_points[index][1])
            if cur_std > best_z[0]:
                if all(i >= 0 and i <= 1 for i in cur_coor):
                    best_z = cur_z
                    best_points = random_points[index]
                    improv = True
                else:
                    random_points.append(np.random.normal(*best_points,1))
            elif cur_std == best_z[0]:
                temp_distance = min_distance(*cur_coor, pos)    
                if temp_distance < best_distance:
                    if all(i >= 0 and i <= 1 for i in cur_coor):
                        best_distance, best_z, best_points = temp_distance, cur_z, cur_coor
                        improv = True
                    else:
                        random_points.append(np.random.normal(*best_points,1))
        if improv == True:
            random_points = np.random.normal((x,y), 1, n)
            z, z_std = gpr.predict(random_points,return_std=True)
    
    
    return (best_points, best_z)
