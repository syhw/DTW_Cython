#cython: profile=True
# set profile to True to see where we are spending time / what is Cythonized
#cython: boundscheck=False
# set boundscheck to True when debugging out of bounds errors
#cython: wraparound=False

# TODO maybe use DTYPE_t[:,::1] (C contiguous) memoryview instd of DTYPE_t[:,:]
# tried it -> no speed improvement
# TODO see if we can add some "nogil" (cpdef foo(int a, int b) nogil:...)
# for instance would have to cpdef double e_dist(...) nogil:...

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt
import time

DTYPE = np.float64 # features' type (could be int)
CTYPE = np.float64 # cost's type (should be float)
ctypedef np.float64_t DTYPE_t
ctypedef np.float64_t CTYPE_t
ctypedef np.intp_t IND_t 
ctypedef CTYPE_t (*dist_func_t)(DTYPE_t[:,:], DTYPE_t[:,:], IND_t, IND_t, IND_t)


cdef CTYPE_t e2_dist_1d(DTYPE_t x, DTYPE_t y):
    """ for debug/test purposes """
    cdef CTYPE_t tmp = x - y
    return tmp*tmp


cdef DTW_1d(DTYPE_t[:] x, DTYPE_t[:] y):
    """ for debug/test purposes """
    cdef IND_t N = x.shape[0]
    cdef IND_t M = y.shape[0]
    cdef IND_t i, j
    cdef CTYPE_t tmp
    cdef CTYPE_t[:,:] cost = np.empty((N, M), dtype=CTYPE)
    for i in range(1, N):
        for j in range(1, M):
            tmp = x[i] - y[j]
            cost[i,j] = tmp*tmp + min(cost[i-1,j], 
            #cost[i,j] = e2_dist_1d(x[i], y[j]) + min(cost[i-1,j], 
                                                     cost[i-1,j-1],
                                                     cost[i,j-1])
    return cost[N-1][M-1]


cdef CTYPE_t dummy_dist(DTYPE_t[:,:] x, DTYPE_t[:,:] y, IND_t i, IND_t j, IND_t K):
    """ to replace "None" so that it's not a Python object in DTW(...) """
    return 0.0


cdef CTYPE_t e_dist(DTYPE_t[:,:] x, DTYPE_t[:,:] y, IND_t i, IND_t j, IND_t K):
    """ Euclidian distance, equivalent to: d=x-y, return sqrt(dot(d,d)) """
    return sqrt(e2_dist(x, y, i, j, K))


cdef CTYPE_t e2_dist(DTYPE_t[:,:] x, DTYPE_t[:,:] y, IND_t i, IND_t j, IND_t K):# nogil:
    """ Squared Euclidian distance, equivalent to: d=x[i]-y[j], return dot(d,d) """
    # In this function, we avoid memoryview slicing for speed,
    # see: http://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/
    cdef CTYPE_t d, tmp
    d = 0.0
    for k in range(K):
        tmp = x[i,k] - y[j,k]
        d += tmp * tmp
    return d


cdef CTYPE_t m_dist(DTYPE_t[:,:] x, DTYPE_t[:,:] y, IND_t i, IND_t j, IND_t K):
    """ Manhattan distance, equivalent to: d=x-y, return sum(abs(d)) """
    cdef CTYPE_t d = 0.0
    for k in range(K):
        d += abs(x[i,k] - y[j,k])
    return d


cdef DTW(x, y, return_alignment=0, dist_func_t cython_dist_function=dummy_dist,
        dist_array=None, python_dist_function=None):
    """ Python wrapper does the array format checks + asserts + distance string
     - x and y should be numpy 2dim ndarrays of DTYPE.
     - return_alignment = 0/1 (False/True) if you want the DTW alignment.
     In order of priority:
         - cython_dist_function should be a Cython function as e2_dist,
         - dist_array should be a x.shape[0], y.shape[0] 2dim ndarray of CTYPE,
         - python_dist_function should be a Python function as e2_dist_python.
     The default distance used is e2_dist.
    --> a bit of overhead, maybe you want to call DTW_cython_(f|a) directly """

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
    assert xx.shape[1] == yy.shape[1], "X and Y do not have the same features dimensions"
    cdef int r_a = return_alignment
    cdef dist_func_t c_d_func = e2_dist
    ###cdef CTYPE_t[:,:] d_array = np.empty((0,0), dtype=CTYPE)
    if cython_dist_function == dummy_dist and dist_array == None and python_dist_function == None:
        return DTW_f(xx, yy, dist_function=c_d_func, return_alignment=r_a)
    elif cython_dist_function != dummy_dist:
        c_d_func = cython_dist_function
        return DTW_f(xx, yy, dist_function=c_d_func, return_alignment=r_a)
    else:
        d_array = np.empty((0,0), dtype=CTYPE) ###
        if dist_array != None:
            assert dist_array.shape == (x.shape[0], y.shape[0]), "dist_array is not of X.Y shape"
            if dist_array.dtype != CTYPE:
                d_array = np.asarray(dist_array, dtype=CTYPE)
            else:
                d_array = dist_array
        else:
            # TODO for/for apply python_dist_function
            # TODO parallelize?
            pass
        return DTW_a(xx, yy, dist_array=d_array, return_alignment=r_a)


cdef find_alignment(CTYPE_t[:,:] cost, IND_t N, IND_t M):
    """ Compute the alignment that achieves the smalled distance CTYPE_t[:,:] cost.
    Also, from x to y: path_x[x_ind] = y_ind; & y to x path_y """
    path = np.zeros((N, M), dtype=np.int)
    path_x = []
    path_y = []
    path[N-1,M-1] = 1
    cdef IND_t i = N-1
    cdef IND_t j = M-1
    cdef CTYPE_t c_i1_j
    cdef CTYPE_t c_i1_j1
    cdef CTYPE_t c_i_j1
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
    return path, path_x, path_y


cdef DTW_a(DTYPE_t[:,:] x, DTYPE_t[:,:] y, CTYPE_t[:,:] dist_array,
        int return_alignment):
    """ the optimised Cython DTW function for using a memoryview dist_array """
    cdef IND_t N = x.shape[0]
    cdef IND_t M = y.shape[0]
    cdef IND_t K = x.shape[1]
    cdef IND_t i, j
    cdef CTYPE_t[:,:] cost = np.empty((N, M), dtype=CTYPE)
    # initialization
    cost[:,0] = dist_array[:,0]
    cost[0,:] = dist_array[0,:]
    # the dynamic programming loop
    for i in range(1, N):
        for j in range(1, M):
            cost[i,j] = dist_array[i,j] + min(cost[i-1,j], 
                                              cost[i-1,j-1],
                                              cost[i,j-1])
    if return_alignment:
        return cost[N-1,M-1], cost, find_alignment(cost, N, M)
    else:
        return cost[N-1,M-1]


cdef DTW_f(DTYPE_t[:,:] x, DTYPE_t[:,:] y, dist_func_t dist_function,
        int return_alignment):
    """ the optimised Cython DTW function for using a Cython dist_function """
    cdef IND_t N = x.shape[0]
    cdef IND_t M = y.shape[0]
    cdef IND_t K = x.shape[1]
    cdef IND_t i, j
    cdef CTYPE_t[:,:] cost = np.empty((N, M), dtype=CTYPE)
    # initialization
    for i in range(N):
        cost[i,0] = dist_function(x, y, i, 0, K)
    for j in range(M):
        cost[0,j] = dist_function(x, y, 0, j, K)
    # the dynamic programming loop
    for i in range(1, N):
        for j in range(1, M):
            cost[i,j] = dist_function(x, y, i, j, K) + min(cost[i-1,j], 
                                                        cost[i-1,j-1],
                                                        cost[i,j-1])

    # here, cost[x.shape[0]-1,y.shape[0]-1] is the best possible distance
    if return_alignment:
        return cost[N-1,M-1], cost, find_alignment(cost, N, M)
    else:
        return cost[N-1,M-1]


def test():
    a = np.array([[1,2,3],[3,4,5],[4,5,6]])
    b = np.array([[1,2,3],[1,2,4],[3,4,5],[4,4,6],[4,5,6]])
    d = DTW(a, b, return_alignment=1)
    print "cost:", d[0]
    print "alignment:"
    print d[2][0]
    assert(np.all(d[2][0] == np.array([[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]])))

    np.random.seed(42)
    a = np.random.random((170, 30))
    b = np.random.random((130, 30))
    print a.shape, "x", b.shape
    t = time.time()
    for k in xrange(10):
        d = DTW(a, b)
    print "took:", ((time.time() - t) / k),  "seconds per run"
    print "cost:", d
    np.testing.assert_almost_equal(d, 533.437172504)

    np.random.seed(42)
    a = np.random.random(900)
    b = np.random.random(1000)
    print a.shape, "x", b.shape
    t = time.time()
    for k in xrange(10):
        d = DTW(a, b)
    print "took:", ((time.time() - t) / k),  "seconds per run"
    print "cost:", d
    np.testing.assert_almost_equal(d, 26.0858110759)
    print "same as above in 1D:"
    for k in xrange(10):
        d = DTW_1d(a, b)
    print "took:", ((time.time() - t) / k),  "seconds per run"
    print "cost:", d
    np.testing.assert_almost_equal(d, 26.0858110759)

    np.random.seed(42)
    idx = np.linspace(0, 2*np.pi, 1000)
    template = np.cos(idx)
    query = np.r_[np.sin(idx) + np.random.random(1000)/2., np.array([0 for i in range(20)])]
    print "cosinus + noise:", template.shape, "x", query.shape
    t = time.time()
    d = DTW(query, template)
    print "took:", (time.time() - t),  "seconds"
    print "cost:", d
    np.testing.assert_almost_equal(d, 51.6351058322)

    # R dtw align of f101_at/af : time: 0.101805925369, cost: 1586.29814585
    import htkmfc
    mfc1 = np.asarray(htkmfc.open("s_f101_at.mfc").getall(), dtype=DTYPE)
    mfc2 = np.asarray(htkmfc.open("s_f101_ar.mfc").getall(), dtype=DTYPE)
    print "MFCC now:"
    print mfc1.shape, "x", mfc2.shape
    t = time.time()
    d = DTW(mfc1, mfc2, return_alignment=1)
    print "took:", (time.time() - t),  "seconds"
    print "cost:", d[0]
    np.testing.assert_almost_equal(d[0], 21856.8021244)
    import pylab as pl
    pl.imshow(d[2][0], interpolation="nearest", origin="lower")
    pl.savefig("path.png")
    pl.imshow(np.asarray(d[1]), interpolation="nearest", origin="lower")
    pl.savefig("cost.png")

    # R dtw align of f113_xof_xok : time: 0.0426249504089, cost: 1730.2299737
    mfc1 = np.asarray(htkmfc.open("s_f113_xof.mfc").getall(), dtype=DTYPE)
    mfc2 = np.asarray(htkmfc.open("s_f113_xok.mfc").getall(), dtype=DTYPE)
    print mfc1.shape, "x", mfc2.shape
    t = time.time()
    d = DTW(mfc1, mfc2, return_alignment=1)
    print "took:", (time.time() - t),  "seconds"
    print "cost:", d[0]
    np.testing.assert_almost_equal(d[0], 26551.3192278)
    pl.imshow(d[2][0], interpolation="nearest", origin="lower")
    pl.savefig("path2.png")
    pl.imshow(np.asarray(d[1]), interpolation="nearest", origin="lower")
    pl.savefig("cost2.png")


if __name__ == '__main__':
    test()

