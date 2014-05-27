DTW in Cython
=============

Easy to use Dynamic Time Warping in Cython. You can set your own distance in
`_example_dist.pyx` by respecting the function name and the type.


Dependencies:
 - Cython
 - Numpy
 - (optional) Matplotlib


Use/test:

    (optional) cython -a dtw/_dtw.pyx && open dtw/_dtw.html
    python -c "import pyximport; import numpy as np;\
    pyximport.install(setup_args={'include_dirs':[np.get_include()]});\
    import dtw; dtw._dtw.test()"


Install:

    make install


See `dtw.DTW(...)` and `dtw._dtw.test()` for the usage of the DTW function.


I keep `DTW_a` and `DTW_f` separated to be able to restrict the search (by a 
maximum warp or a heuristic) in `DTW_f`, while I can experiment with parallel 
computing of the `dist_array` for `DTW_a` (on GPUs).

