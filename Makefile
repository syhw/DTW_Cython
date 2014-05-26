all:
	python setup.py build_ext --inplace
	cython -a dtw/_dtw.pyx

install:
	python setup.py install

clean:
	rm dtw/*.c
