#cython: profile=True
#TODO set to False
#cython: boundscheck=False
#cython: wraparound=False


cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport sqrt
#from cython.view cimport array as cvarray
import time

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


#def e_dist(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
def e_dist(DTYPE_t[:] x, DTYPE_t[:] y):
    """ Equivalent to: d=x-y, return sqrt(dot(d,d)) """
    # In [11]: %timeit dist(a,b) # numpy version as above
    # 100000 loops, best of 3: 6.49 µs per loop
    # In [12]: %timeit e_dist(a,b)
    # 1000000 loops, best of 3: 1.78 µs per loop
    cdef DTYPE_t d, tmp
    cdef int K = x.shape[0]
    d = 0.0
    for k in range(K):
        tmp = x[k] - y[k]
        d += tmp * tmp
    return sqrt(d)


#def euclidian_distance(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
def euclidian_distance(DTYPE_t[:,:] x, DTYPE_t[:,:] y):
    cdef DTYPE_t d, tmp
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int K = y.shape[1]
    #cdef double[:, ::1] D = np.empty((N,M), dtype=np.float64)
    cdef double[:,:] D = np.empty((N,M), dtype=np.float64)
    #cdef np.ndarray[np.float_t, ndim=2] D = np.empty((N,M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            d = 0.0
            for k in range(K):
                tmp = x[i,k] - y[j,k]
                d += tmp * tmp
            D[i,j] = sqrt(d)
    return np.asarray(D)


#def manhattan_distance(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
def manhattan_distance(DTYPE_t[:,:] x, DTYPE_t[:,:] y):
    cdef DTYPE_t d, tmp
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int K = y.shape[1]
    #cdef double[:, ::1] D = np.empty((N,M), dtype=np.float64)
    cdef double[:,:] D = np.empty((N,M), dtype=np.float64)
    #cdef np.ndarray[np.float_t, ndim=2] D = np.empty((N,M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            d = 0.0
            for k in range(K):
                d += abs(x[i,k] - y[j,k])
            D[i,j] = d
    return np.asarray(D)


def DTW(x, y, dist_function=None, dist_array=None):
    """ Python wrapper does the array format checks + asserts """
    xx = x
    if len(x.shape) == 1:
        xx = np.reshape(x, (x.shape[0], 1))
    yy = y
    if len(y.shape) == 1:
        yy = np.reshape(y, (y.shape[0], 1))
    assert(xx.shape[1] == yy.shape[1])
    if dist_array != None:
        assert(dist_array.shape == (x.shape[0], y.shape[0]))
    return DTW_cython(xx, yy, dist_function=dist_function, dist_array=dist_array)


#def DTW_cython(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y, dist_function=None, dist_array=None):
def DTW_cython(DTYPE_t[:,:] x, DTYPE_t[:,:] y, dist_function=None, dist_array=None):
    """
    Default is euclidian distance, otherwise provide (in order of priority):
     - a distance function as dist_function
     - a distance array as dist_array[x_ind][y_ind]
    """
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int i, j
    #cdef double[:, ::1] cost = np.empty((N, M), dtype=np.float64)
    cdef double[:, :] cost = np.empty((N, M), dtype=np.float64)
    #cdef np.ndarray[np.float_t, ndim=2] cost = np.empty((N,M), dtype=np.float64)

    if dist_array != None:
        cost[:,0] = dist_array[:,0]
        cost[0,:] = dist_array[0,:]
        # the dynamic programming loop
        for i in range(1, N):
            for j in range(1, M):
                cost[i,j] = dist_array[i,j] + min(cost[i-1,j], 
                                                  cost[i-1,j-1],
                                                  cost[i,j-1])
    else:
        if dist_function == None:
            dist_function = e_dist
        for i in range(N):
            cost[i,0] = dist_function(x[i], y[0])
        for j in range(M):
            cost[0,j] = dist_function(x[0], y[j])
        # the dynamic programming loop
        for i in range(1, N):
            for j in range(1, M):
                cost[i,j] = dist_function(x[i], y[j]) + min(cost[i-1,j], 
                                                            cost[i-1,j-1],
                                                            cost[i,j-1])
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
        d = DTW(a, b, e_dist)
    print d
    print "took:", ((time.time() - t) / k),  "seconds per run"


    import sys
    sys.exit(0)


    np.random.seed(42)
    a = np.random.random(900)
    b = np.random.random(1000)
    t = time.time()
    for k in xrange(10):
        DTW(a, b, euclidian_distance)
    print "took:", ((time.time() - t) / k),  "seconds per run"
    np.random.seed(42)
    idx = np.linspace(0, 2*np.pi, 1000)
    template = np.cos(idx)
    query = np.r_[np.sin(idx) + np.random.random(1000)/2., np.array([0 for i in range(20)])]
    t = time.time()
    print DTW(query, template, euclidian_distance)
    print "took:", ((time.time() - t) / k),  "seconds"


if __name__ == '__main__':
    test()

