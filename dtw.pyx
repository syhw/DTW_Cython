#cython: profile=True
# set profile to True to see where we are spending time / what is Cythonized
#cython: boundscheck=False
# set boundscheck to True when debugging out of bounds errors
#cython: wraparound=False

# TODO maybe use DTYPE_t[:,::1] (C contiguous) memoryview instd of DTYPE_t[:,:]
# tried it -> no speed improvement
# TODO 
# see: http://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/
# tried it on e_dist -> no speed improvment???
# TODO understand why cpdef is faster than cdef
# TODO see if we can add some "nogil" (cpdef foo(int a, int b) nogil:...)
# for instance would have to cpdef double e_dist(...) nogil:...

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
import time

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
#ctypedef DTYPE_t (*dist_func_t)(DTYPE_t[:], DTYPE_t[:])
ctypedef DTYPE_t (*dist_func_t)(DTYPE_t[:,:], DTYPE_t[:,:], np.intp_t, np.intp_t, np.intp_t)


cdef DTYPE_t e2_dist_1d(DTYPE_t x, DTYPE_t y):
    cdef DTYPE_t tmp = x - y
    return tmp*tmp


cdef DTYPE_t e2_dist(DTYPE_t[:] x, DTYPE_t[:] y):
    """ Squared Euclidian distance, equivalent to: d=x-y, return dot(d,d) """
    cdef DTYPE_t d, tmp
    cdef np.intp_t K = x.shape[0]
    d = 0.0
    for k in range(K):
        tmp = x[k] - y[k]
        d += tmp * tmp
    return d


cdef DTYPE_t e_dist(DTYPE_t[:] x, DTYPE_t[:] y):
    """ Euclidian distance, equivalent to: d=x-y, return sqrt(dot(d,d)) """
    # In [11]: %timeit dist(a,b) # numpy version as above
    # 100000 loops, best of 3: 6.49 µs per loop
    # In [12]: %timeit e_dist(a,b)
    # 1000000 loops, best of 3: 1.78 µs per loop
    return sqrt(e2_dist(x, y))


cdef DTYPE_t e2_dist_noslicing(DTYPE_t[:,:] X, DTYPE_t[:,:] Y, np.intp_t i, np.intp_t j, np.intp_t K):
    """ Squared Euclidian distance, equivalent to: d=X[i]-Y[j], return dot(d,d) """
    cdef DTYPE_t d, tmp
    #cdef np.intp_t K = X.shape[1]
    d = 0.0
    for k in range(K):
        tmp = X[i,k] - Y[j,k]
        d += tmp * tmp
    return d


cdef DTYPE_t m_dist(DTYPE_t[:] x, DTYPE_t[:] y):
    """ Manhattan distance, equivalent to: d=x-y, return sum(abs(d)) """
    # In [11]: %timeit dist(a,b) # numpy version as above
    # 100000 loops, best of 3: 11.1 µs per loop
    # In [12]: %timeit m_dist(a,b)
    # 1000000 loops, best of 3: 1.9 µs per loop
    cdef DTYPE_t d
    cdef np.intp_t K = x.shape[0]
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
    cdef int with_d_array = 0
    cdef DTYPE_t[:,:] d_array = np.empty((0,0), dtype=DTYPE)
    if dist_array != None:
        with_d_array = 1
        assert(dist_array.shape == (x.shape[0], y.shape[0]))
        if dist_array.dtype != DTYPE:
            d_array = np.asarray(dist_array, dtype=DTYPE)
    #cdef dist_func_t cython_dist_func = e2_dist
    cdef dist_func_t cython_dist_func = e2_dist_noslicing
#    if dist_function == "euclidian":
#        cython_dist_func = e_dist
#    elif dist_function == "manhattan":
#        cython_dist_func = m_dist
    return DTW_cython(xx, yy, dist_function=cython_dist_func, dist_array=d_array, with_dist_array=with_d_array)


cdef DTW_1d(DTYPE_t[:] x, DTYPE_t[:] y):
    cdef np.intp_t N = x.shape[0]
    cdef np.intp_t M = y.shape[0]
    cdef np.intp_t i, j
    #cdef DTYPE_t tmp
    cdef DTYPE_t[:,:] cost = np.empty((N, M), dtype=DTYPE)
    for i in range(1, N):
        for j in range(1, M):
            #tmp = x[i] - y[j]
            #cost[i,j] = tmp*tmp + min(cost[i-1,j], 
            cost[i,j] = e2_dist_1d(x[i], y[j]) + min(cost[i-1,j], 
                                                     cost[i-1,j-1],
                                                     cost[i,j-1])
    return cost[N-1][M-1]


cdef DTW_cython(DTYPE_t[:,:] x, DTYPE_t[:,:] y, dist_func_t dist_function, DTYPE_t[:,:] dist_array, int with_dist_array):
    """
    Default is euclidian distance, otherwise provide (in order of priority):
     - a distance array as dist_array[x_ind][y_ind]
     - a distance function as dist_function
    """
    cdef np.intp_t N = x.shape[0]
    cdef np.intp_t M = y.shape[0]
    cdef np.intp_t K = x.shape[1]
    cdef np.intp_t i, j
    cdef DTYPE_t[:,:] cost = np.empty((N, M), dtype=DTYPE)

    if with_dist_array:
        cost[:,0] = dist_array[:,0]
        cost[0,:] = dist_array[0,:]
        # the dynamic programming loop
        for i in range(1, N):
            for j in range(1, M):
                cost[i,j] = dist_array[i,j] + min(cost[i-1,j], 
                                                  cost[i-1,j-1],
                                                  cost[i,j-1])
    else:
        for i in range(N):
            #cost[i,0] = dist_function(x[i], y[0])
            cost[i,0] = dist_function(x, y, i, 0, K)
        for j in range(M):
            #cost[0,j] = dist_function(x[0], y[j])
            cost[0,j] = dist_function(x, y, 0, j, K)
        # the dynamic programming loop
        for i in range(1, N):
            for j in range(1, M):
                #cost[i,j] = dist_function(x[i], y[j]) + min(cost[i-1,j], 
                cost[i,j] = dist_function(x, y, i, j, K) + min(cost[i-1,j], 
                                                            cost[i-1,j-1],
                                                            cost[i,j-1])
    # here, cost[x.shape[0]-1,y.shape[0]-1] is the best possible distance
    # now, compute the path from x to y: path_x[x_ind] = y_ind; & y to x path_y
    path = np.zeros((N, M), dtype=np.int)
    path_x = []
    path_y = []
    path[N-1,M-1] = 1
    i = N-1
    j = M-1
    cdef DTYPE_t c_i1_j
    cdef DTYPE_t c_i1_j1
    cdef DTYPE_t c_i_j1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # where does cost[i,j] come from?
            c_i1_j = cost[i-1,j]
            c_i1_j1 = cost[i-1,j-1]
            c_i_j1 = cost[i,j-1]
            if c_i1_j1 <= c_i1_j and c_i1_j1 <= c_i_j1:
                i -= 1
                j -= 1
            elif c_i1_j < c_i1_j1 and c_i1_j < c_i_j1:
                i -= 1
            else:
                j -= 1
        path[i,j] = 1
        path_x.append(i)
        path_y.append(j)
    path_x.reverse()
    path_y.reverse()

    return cost[N-1,M-1], cost, (path, path_x, path_y)


def test():
    a = np.array([[1,2,3],[3,4,5],[4,5,6]])
    b = np.array([[1,2,3],[1,2,4],[3,4,5],[4,4,6],[4,5,6]])
    d = DTW(a, b)
    print "cost:", d[0]
    print "alignment:"
    print d[2][0]

    np.random.seed(42)
    a = np.random.random((170, 30))
    b = np.random.random((130, 30))
    t = time.time()
    for k in xrange(5):
        d = DTW(a, b)
    print "took:", ((time.time() - t) / k),  "seconds per run"
    print "cost:", d[0]
    ##np.testing.

    np.random.seed(42)
    a = np.random.random(9000)
    b = np.random.random(10000)
    t = time.time()
    for k in xrange(3):
        d = DTW(a, b)
        #d = DTW_1d(a, b)
    print "took:", ((time.time() - t) / k),  "seconds per run"
    print "cost:", d[0]
    #print "cost:", d
    ##np.testing.assert_almost_equal(d[0], 26.0858110759)
    #np.testing.assert_almost_equal(d, 26.0858110759)

    np.random.seed(42)
    idx = np.linspace(0, 2*np.pi, 1000)
    template = np.cos(idx)
    query = np.r_[np.sin(idx) + np.random.random(1000)/2., np.array([0 for i in range(20)])]
    t = time.time()
    d = DTW(query, template)
    print "took:", (time.time() - t),  "seconds"
    print "cost:", d[0]
    ##np.testing.assert_almost_equal(d[0], 51.6351058322)

    # R dtw align of f101_at/af : time: 0.101805925369, cost: 1586.29814585
    import htkmfc
    mfc1 = np.asarray(htkmfc.open("s_f101_at.mfc").getall(), dtype=DTYPE)
    mfc2 = np.asarray(htkmfc.open("s_f101_ar.mfc").getall(), dtype=DTYPE)
    t = time.time()
    d = DTW(mfc1, mfc2)
    print "took:", (time.time() - t),  "seconds"
    print mfc1.shape, mfc2.shape
    print "cost:", d[0]
    import pylab as pl
    pl.imshow(d[2][0], interpolation="nearest", origin="lower")
    pl.savefig("path.png")
    pl.imshow(np.asarray(d[1]), interpolation="nearest", origin="lower")
    pl.savefig("cost.png")

    # R dtw align of f113_xof_xok : time: 0.0426249504089, cost: 1730.2299737
    mfc1 = np.asarray(htkmfc.open("s_f113_xof.mfc").getall(), dtype=DTYPE)
    mfc2 = np.asarray(htkmfc.open("s_f113_xok.mfc").getall(), dtype=DTYPE)
    t = time.time()
    d = DTW(mfc1, mfc2)
    print "took:", (time.time() - t),  "seconds"
    print "cost:", d[0]
    pl.imshow(d[2][0], interpolation="nearest", origin="lower")
    pl.savefig("path2.png")
    pl.imshow(np.asarray(d[1]), interpolation="nearest", origin="lower")
    pl.savefig("cost2.png")


if __name__ == '__main__':
    test()

