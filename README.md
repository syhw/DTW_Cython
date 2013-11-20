DTW in Cython
=============

Easy to use Dynamic Time Warping in Cython. You can set your own distance in
`my_dist.pyx` by respecting the function name and the type.


Usage:
    make
    (optional) open dtw.html
    python -c "import dtw; dtw.test()"


See `DTW(...)` and `dtw.test()` for the usage of the DTW function.


I keep `DTW_a` and `DTW_f` separated to be able to restrict the search (by a 
maximum warp or a heuristic) in `DTW_f`, while I can experiment with parallel 
computing of the `dist_array` for `DTW_a` (on GPUs).

