from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import *
from distutils.util import *
from Cython.Distutils import build_ext
import os, numpy

py_inc = [get_python_inc()]
np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]

setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("dtw", 
                                 ["dtw.pyx"],
                                 include_dirs=py_inc+np_inc,
                                 extra_compile_args=["-O3"])],
        include_dirs=[numpy.get_include(),
                      os.path.join(numpy.get_include(), 'numpy')]
)
