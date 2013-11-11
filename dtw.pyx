#cython: profile=True
#TODO set to False
#cython: boundscheck=False
#cython: wraparound=False


cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport sqrt
import time

DTYPE = np.float
ctypedef np.float_t DTYPE_t


def min_3(np.float_t a, np.float_t b, np.float_t c):
    """ Faster than min(a, min(b, c)) """
    if a < b and a < c:
        return a
    elif b < a and b < c:
        return b
    else:
        return c


def euclidian_distance(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
    cdef DTYPE_t d, tmp
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int K = y.shape[1]
    #cdef double[:, ::1] D = np.empty((N,M), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=2] D = np.empty((N,M), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            d = 0.0
            for k in range(K):
                tmp = x[i,k] - y[j,k]
                d += tmp * tmp
            D[i,j] = sqrt(d)
    return np.asarray(D)


def manhattan_distance(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y):
    cdef DTYPE_t d, tmp
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int K = y.shape[1]
    #cdef double[:, ::1] D = np.empty((N,M), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=2] D = np.empty((N,M), dtype=np.float64)
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
    return DTW_cython(xx, yy, dist_function=dist_function, dist_array=dist_array)


def DTW_cython(np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=2] y, dist_function=None, dist_array=None):
    """
    Default is euclidian distance, otherwise provide (in order of priority):
     - a distance function as dist_function
     - a distance array as dist_array[x_ind][y_ind]
    """

    if dist_array == None:
        if dist_function == None:
            dist_array = euclidian_distance(x, y)
        else:
            dist_array = dist_function(x, y)

    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    #cdef double[:, ::1] cost = np.empty((N, M), dtype=np.float64)
    cdef np.ndarray[np.float_t, ndim=2] cost = np.empty((N,M), dtype=np.float64)
    cost[0,0] = dist_array[0,0]
    cost[:,0] = dist_array[:,0]
    cost[0,:] = dist_array[0,:]
    # the dynamic programming loop
    for i in range(1, N):
        for j in range(1, M):
            cost[i,j] = dist_array[i,j] + min(cost[i-1,j], cost[i-1,j-1],
                                                             cost[i,j-1])
    # here, cost[x.shape[0]-1,y.shape[0]-1] is the best possible distance
    path_x = None
    path_y = None
    return cost[N-1,M-1], cost, (path_x, path_y)


def test():
    np.random.seed(42)
    a = np.random.random((110, 10))
    b = np.random.random((130, 10))
    t = time.time()
    for k in xrange(10):
        d = DTW(a, b, euclidian_distance)
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

