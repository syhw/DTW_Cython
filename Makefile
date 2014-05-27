cythonize:
	cython -a dtw/_dtw.pyx

install: clean
	python setup.py install

clean:
	rm -rf dtw/*.c dtw/__init__.pyc build/ dist/ _*.so
