import os, sys

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

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
        name = 'DTW_Cython',
        version = '0.1.0',
        url = 'https://github.com/SnippyHolloW/DTW_Cython',
        author = 'Gabriel Synnaeve',
        author_email = 'gabriel.synnaeve@gmail.com',
        packages = ['dtw'],
        license = 'MIT (see the LICENSE file)',
        description = 'Dynamic Time Warping in Cython',
        long_description=open('README.md').read(),
        install_requires = ['Cython', 'numpy'],
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("_example_dist", 
                                 ["dtw/_example_dist.pyx"],
                                 include_dirs=py_inc+np_inc+['dtw'],
                                 extra_compile_args=["-O3"]),
                       Extension("_dtw", 
                                 ["dtw/_dtw.pyx"],
                                 include_dirs=py_inc+np_inc+['dtw'],
                                 extra_compile_args=["-O3"])],
        include_dirs=[numpy.get_include(),
                      os.path.join(numpy.get_include(), 'numpy')]
)

