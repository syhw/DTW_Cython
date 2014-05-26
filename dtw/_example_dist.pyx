#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False

cdef CTYPE_t dist(DTYPE_t[:,:] x, DTYPE_t[:,:] y, IND_t i, IND_t j, IND_t K):
    """ In-between Manhattan and Euclidian distance, equivalent to: d=x[i]-y[j], return d^(1.5) """
    # In this function, we avoid memoryview slicing for speed,
    # see: http://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/
    cdef CTYPE_t d, tmp
    d = 0.0
    for k in range(K):
        tmp = abs(x[i,k] - y[j,k])
        d += pow(tmp, 1.5) # should be fast, according to http://docs.cython.org/src/userguide/language_basics.html
    return d


