all:
	python setup.py build_ext --inplace
	cython -a dtw.pyx
