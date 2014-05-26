import numpy as np
cimport numpy as np
cimport cython
include "_dtw_types.pxi"

cdef CTYPE_t dist(DTYPE_t[:,:] x, DTYPE_t[:,:] y, IND_t i, IND_t j, IND_t K)

