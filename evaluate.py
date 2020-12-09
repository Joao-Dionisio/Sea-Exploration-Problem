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
import math
import random
import numpy as np
from static import inst

EPS = 1.e-6  # for floating point comparisons
INF = float('Inf')


# auxiliary function: euclidean distance
def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def evaluate(f, planner, estimator, mesh, mode=0, N = 4, grid = True, dynam=False):
    # prepare data
    X, z = [], []

    # grid
    if grid:
        for x in [i / N for i in range(1, N+1)]: 
            for y in [i / N for i in range(1, N+1)]:
                X.append((x, y))
                z.append(f(x, y))
    else: 
        # alternative to grid: random points
        for x in [i/N for i in range(1,N+1)]:
            for y in [i/N for i in range(1,N+1)]:
                x = random.random()
                y = random.random()
                X.append((x,y))
                z.append(f(x,y))
                print("{}\t&{}\t&{}\\\\".format(x,y,z[-1]))
        
    # test preliminary forecasting part
    #if mesh == []:
    for i in range(101):
        for j in range(101):
            x, y = i / 100., j / 100.
            mesh.append((x, y))

    # treina em X, z e retorna valores preditos para os pontos no mesh
    z0 = estimator(X, z, mesh)

    # calcula erro preliminar no mesh
    prelim = 0
    for i in range(len(mesh)):
        (x, y) = mesh[i]
        prelim += abs(f(x, y) - float(z0[i]))

    # test planning part

    # calcula rota
    route = planner(X, z, f, mode, dynam)

    # calcula duração da rota
    tsp_et = 0  # elapsed time
    (xt, yt) = (inst.x0, inst.y0)
    for (x, y) in route:
        tsp_et += dist(xt, yt, x, y) / inst.s + inst.t
        xt, yt = x, y
        X.append((x, y))
        z.append(f(x, y))
        # print("probing at ({:8.5g},{:8.5g}) --> \t{:8.5g}".format(x, y, z[-1]))
        print("{:8.5g}\t{:8.5g}\t{:8.5g}".format(x, y, z[-1]))

    # # # plot posteriori GP
    # from sklearn.gaussian_process import GaussianProcessRegressor
    # from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, DotProduct, RationalQuadratic
    # kernel = RationalQuadratic(length_scale_bounds=(0.08, 100)) + WhiteKernel(noise_level_bounds=(1e-5, 1e-2))
    # from functions import plot
    # GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # GPR.fit(X, z)
    # def GP(x_,y_):
    #     return GPR.predict([(x_,y_)])[0]
    # plot(GP,100)
    # # # end of plot
    pass

    # acrescenta tempo de retorno ao porto
    tsp_et += dist(xt, yt, inst.x0, inst.y0) / inst.s

    if tsp_et > inst.T + EPS:
        print("tour length infeasible:", tsp_et)
        # return INF    # route takes longer than time limit
    

    # test forecasting part
    #if mesh == []: # grid test favours grid search
    mesh = []
    for i in range(101):
        for j in range(101):
            x, y = i / 100., j / 100.
            mesh.append((x, y))
    
    z = estimator(X, z, mesh)
    final = 0
    for i in range(len(mesh)):
        (x, y) = mesh[i]
        final += abs(f(x, y) - float(z[i]))

    return prelim, tsp_et, final


if __name__ == "__main__":
    from static import planner, estimator

    '''
    random.seed(0)
    prelim, tsp_len, final = evaluate(f, planner, estimator, False)
    print("student's evaluation:\t{:8.7g}\t[TSP:{:8.7g}]\t{:8.7g}".format(prelim, tsp_len, final))

    random.seed(0)
    prelim, tsp_len, final = evaluate(f, planner, estimator, True)
    print("student's evaluation:\t{:8.7g}\t[TSP:{:8.7g}]\t{:8.7g}".format(prelim, tsp_len, final))
    '''
    total_result = []
    random.seed(0)
    mesh = []
    '''
    for i in range(101):
        for j in range(101):
            x, y = random.random(), random.random()
            mesh.append((x, y))
    '''
    for _ in range(1):
        import functions as fs
        result = []
        # for mode in range(4): # 0 - Grid; 1 - Local; 2 - Multimodal; 3 - Multimodal + Local
        for i in [4,7,10]: 
            print("Starting to work with ", i**2, " initial points and randomize = ", 'a')
            partial_result = []
            for f in [fs.f1,fs.f2,fs.f3,fs.f4,fs.f5,fs.f6,fs.f7,fs.f8,fs.f9,fs.f10]:
                
                print("Working with function ", f)
                print()       
                random.seed(0)
                np.random.seed(0)
                prelim, tsp_len, final = evaluate(f, planner, estimator, mesh, mode=3, N=i, grid=True)
                print("student's evaluation:\t{:8.7g}\t[TSP:{:8.7g}]\t{:8.7g}".format(prelim, tsp_len, final))
                partial_result.append(final) 
            print(partial_result)
            result.append(partial_result)
        print(result)
        total_result.append(result)
    print(total_result)

    # # for reading trend from csv file:
    # import csv
    # import gzip
    # with gzip.open("data_2017_newsvendor.csv.gz", 'rt') as f:
    #     reader = csv.reader(f)
    #     data = [int(t) for (t,) in reader]
