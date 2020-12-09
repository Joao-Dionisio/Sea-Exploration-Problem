# A bunch of useful functions
from random import random
import numpy as np
import math 

# auxiliary function: euclidean distance
def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    

def get_random_points(points,n):
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
                break 
        if not fst:
            random_points = np.append(random_points,np.random.normal(coor,chosen_z_std, size=(1,2)),axis=0)
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



def add_to_dist_matrix(c, pos, new_point):
    # We need to push every point bigger than index 1 point to the right!
    x, y = new_point[0]
    for i in range(len(pos)-1):
        c[i, len(pos)-1] = dist(*pos[i], x,y)
        c[len(pos)-1, i] = c[i, len(pos)-1]
    return c


def update_dist_matrix(c, index, pos, new_point):
    # we need to remove from the dist_matrix the last point that was added!
    for i in c:
        if i[0] == index:
            c[i] = dist(*pos[i[1]], *new_point[0])/inst.s
        elif i[1] == index:
            c[i] = dist(*pos[i[0]], *new_point[0])/inst.s
    return c


def remove_from_dist_matrix(c, index):
    # We have to push every point bigger than index 1 point to the left!
    for i in list(c): # has to be list because c changes size during iteration
        if index in i:
            del c[i]
            continue
        if i[0] > index:
            if i[1] > index:
                c[(i[0]-1,i[1]-1)] = c.pop(i)
            else:
                c[(i[0]-1,i[1])] = c.pop(i)
        elif i[1] > index:
            c[(i[0],i[1]-1)] = c.pop(i)
    return c
