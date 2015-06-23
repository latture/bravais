"""
setup.py file for SWIG
"""

from distutils.core import setup, Extension

cpp_utils_module = Extension('_cpp_utils',
                             sources=['cpp_utils_wrap.cxx', 'cpp_utils.cpp'],
                             swig_opts=['-c++'],
                             language="c++",
                             extra_compile_args=['-O3', '-std=gnu++11', '-fopenmp'],
                             extra_link_args=['-O3', '-std=gnu++11', '-fopenmp'])

setup(name='cpp_utils',
      version='0.1',
      author="Ryan Latture",
      description="""Utility functions""",
      ext_modules=[cpp_utils_module],
      py_modules=["cpp_utils"],
)