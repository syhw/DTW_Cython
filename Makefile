build:
	cython -a dtw/_dtw.pyx
	python setup.py build_ext --inplace

install: clean
	python setup.py install

clean:
	rm dtw/*.c
	rm dtw/__init__.pyc
	rm -rf build/
	rm -rf dist/
	rm _*.so
