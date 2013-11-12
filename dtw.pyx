#cython: profile=False
# set profile to True to see where we are spending time / what is Cythonized
#cython: boundscheck=False
# set boundscheck to True when debugging out of bounds errors
#cython: wraparound=False

# TODO maybe use DTYPE_t[:,::1] (C contiguous) memoryview instd of DTYPE_t[:,:]
# tried it -> no speed improvement
# TODO understand why cpdef is faster than cdef
# TODO see if we can add some "nogil" (cpdef foo(int a, int b) nogil:...)

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
import time

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cpdef e_dist(DTYPE_t[:] x, DTYPE_t[:] y):
    """ Euclidian distance, equivalent to: d=x-y, return sqrt(dot(d,d)) """
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


cpdef m_dist(DTYPE_t[:] x, DTYPE_t[:] y):
    """ Manhattan distance, equivalent to: d=x-y, return sum(abs(d)) """
    # In [11]: %timeit dist(a,b) # numpy version as above
    # 100000 loops, best of 3: 11.1 µs per loop
    # In [12]: %timeit m_dist(a,b)
    # 1000000 loops, best of 3: 1.9 µs per loop
    cdef DTYPE_t d
    cdef int K = x.shape[0]
    d = 0.0
    for k in range(K):
        d += abs(x[k] - y[k])
    return d


def DTW(x, y, dist_function=None, dist_array=None):
    """ Python wrapper does the array format checks + asserts + distance string
     - x and y should be numpy 2dim ndarrays of DTYPE.
     - dist_function should be a Cython/Python function,
       or "euclidian"/"manhattan" strings.
     - dist_array should be a x.shape[0], y.shape[0] 2dim ndarray of DTYPE.
    --> a little bit of overhead, maybe you want to call DTW_cython directly"""
    xx = x
    if len(x.shape) == 1:
        xx = np.reshape(x, (x.shape[0], 1))
    if xx.dtype != DTYPE:
        xx = np.asarray(xx, dtype=DTYPE)
    yy = y
    if len(y.shape) == 1:
        yy = np.reshape(y, (y.shape[0], 1))
    if yy.dtype != DTYPE:
        yy = np.asarray(yy, dtype=DTYPE)
    assert(xx.shape[1] == yy.shape[1])
    if dist_array != None:
        assert(dist_array.shape == (x.shape[0], y.shape[0]))
        if dist_array.dtype != DTYPE:
            dist_array = np.asarray(dist_array, dtype=DTYPE)
    if dist_function == "euclidian":
        dist_function = e_dist
    elif dist_function == "manhattan":
        dist_function = m_dist
    return DTW_cython(xx, yy, dist_function=dist_function, dist_array=dist_array)


cpdef DTW_cython(DTYPE_t[:,:] x, DTYPE_t[:,:] y, dist_function=None, dist_array=None):
    """
    Default is euclidian distance, otherwise provide (in order of priority):
     - a distance array as dist_array[x_ind][y_ind]
     - a distance function as dist_function
    """
    cdef int N = x.shape[0]
    cdef int M = y.shape[0]
    cdef int i, j
    cdef double[:,:] cost = np.empty((N, M), dtype=DTYPE)

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
    # now, compute the path from x to y: path_x[x_ind] = y_ind; & y to x path_y
    path = None

    return cost[N-1,M-1], cost, path


def test():
    np.random.seed(42)
    a = np.random.random((170, 10))
    b = np.random.random((130, 10))
    t = time.time()
    for k in xrange(10):
        d = DTW(a, b, e_dist)
    print d
    print "took:", ((time.time() - t) / k),  "seconds per run"
    np.testing.assert_almost_equal(d[0], 139.79425569811386)

    np.random.seed(42)
    a = np.random.random(900)
    b = np.random.random(1000)
    t = time.time()
    for k in xrange(5):
        d = DTW(a, b, e_dist)
    print d
    print "took:", ((time.time() - t) / k),  "seconds per run"
    np.testing.assert_almost_equal(d[0], 126.59496270135652)

    np.random.seed(42)
    idx = np.linspace(0, 2*np.pi, 1000)
    template = np.cos(idx)
    query = np.r_[np.sin(idx) + np.random.random(1000)/2., np.array([0 for i in range(20)])]
    t = time.time()
    d = DTW(query, template, e_dist)
    print d
    print "took:", ((time.time() - t) / k),  "seconds"
    np.testing.assert_almost_equal(d[0], 147.10538852640641)

    # R dtw align of f101_at/af : time: 0.101805925369, cost: 1586.29814585
    import htkmfc
    mfc1 = np.asarray(htkmfc.open("s_f101_at.mfc").getall(), dtype=DTYPE)
    mfc2 = np.asarray(htkmfc.open("s_f101_ar.mfc").getall(), dtype=DTYPE)
    t = time.time()
    d = DTW(mfc1, mfc2, e_dist)
    print d
    print "took:", ((time.time() - t) / k),  "seconds"

    # R dtw align of f113_xof_xok : time: 0.0426249504089, cost: 1730.2299737
    mfc1 = np.asarray(htkmfc.open("s_f113_xof.mfc").getall(), dtype=DTYPE)
    mfc2 = np.asarray(htkmfc.open("s_f113_xok.mfc").getall(), dtype=DTYPE)
    t = time.time()
    d = DTW(mfc1, mfc2, e_dist)
    print d
    print "took:", ((time.time() - t) / k),  "seconds"


if __name__ == '__main__':
    test()

