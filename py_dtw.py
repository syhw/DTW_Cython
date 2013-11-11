import numpy as np
import time

def euclidian_distance(x, y):
    d = x-y
    return np.sqrt(np.dot(d, d))


def manhattan_distance(x, y):
    return np.sum(np.abs(x-y))


#@profile
def fill_dist_array(x, y, d_f):
    tmp_d_a = np.ndarray((x.shape[0], y.shape[0]))
    for i in xrange(x.shape[0]):
        for j in xrange(y.shape[0]):
            tmp_d_a[i,j] = d_f(i,j)
    return tmp_d_a


#@profile
def DTW(x, y, dist_function=None, dist_array=None):
    """
    Default is euclidian distance, otherwise provide (in order of priority):
     - a distance function as dist_function
     - a distance array as dist_array[x_ind][y_ind]
    """

    d_f = None
    d_a = dist_array
    if dist_function == None and dist_array == None:
        d_f = lambda i,j: euclidian_distance(x[i], y[j])
    else:
        if dist_function != None:
            d_f = lambda i,j: dist_function(x[i], y[j])
        elif dist_array != None:
            d_f = lambda i,j: dist_array[i][j]
    #if d_a == None:
    #    d_a = fill_dist_array(x, y, d_f)

    cost = np.empty((x.shape[0], y.shape[0]), dtype=np.float)
    cost[0,0] = d_f(0,0)
    N = x.shape[0]
    for i in xrange(1, N):
        cost[i,0] = d_f(i,0)
    M = y.shape[0]
    for j in xrange(1, M):
        cost[0,j] = d_f(0,j)
    # the dynamic programming loop
    for i in xrange(1, N):
        for j in xrange(1, M):
            #cost[i,j] = d_a[i,j] + np.min([cost[i-1,j], cost[i-1,j-1],
            cost[i,j] = d_f(i,j) + np.min((cost[i-1,j], cost[i-1,j-1],
                                                         cost[i,j-1]))
    # here, cost[x.shape[0]-1,y.shape[0]-1] is the best possible distance
    path_x = None
    path_y = None
    return cost[N-1,M-1], cost, (path_x, path_y)


def test():
    np.random.seed(42)
    a = np.random.random((170, 10))
    b = np.random.random((130, 10))
    t = time.time()
    for k in xrange(10):
        d = DTW(a, b, euclidian_distance)
    print d
    print "took:", ((time.time() - t) / k),  "seconds per run"


if __name__ == '__main__':
    test()

