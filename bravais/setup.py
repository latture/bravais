from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

extensions = [
    Extension("assign_major_axes.assign_major_axes", ["assign_major_axes/assign_major_axes.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"]),
    Extension('cpp_utils._cpp_utils',
              sources=['cpp_utils/cpp_utils_wrap.cxx', 'cpp_utils/cpp_utils.cpp'],
              swig_opts=['-c++'],
              language="c++",
              extra_compile_args=['-O3', '-std=gnu++11', '-fopenmp'],
              extra_link_args=['-O3', '-std=gnu++11', '-fopenmp'])
]

setup(
    name="extension_modules",
    ext_modules=cythonize(extensions, annotate=False),
)
