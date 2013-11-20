import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float64_t DTYPE_t
ctypedef np.float64_t CTYPE_t
ctypedef np.intp_t IND_t 
ctypedef CTYPE_t (*dist_func_t)(DTYPE_t[:,:], DTYPE_t[:,:], IND_t, IND_t, IND_t)
DTYPE = np.float64 # features' type (could be int)
CTYPE = np.float64 # cost's type (should be float)

