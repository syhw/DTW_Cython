all:
	python setup.py build_ext --inplace
	cython -a dtw.pyx
	#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
	#	      -I/usr/include/python2.7 -o mydtw.so mydtw.c
