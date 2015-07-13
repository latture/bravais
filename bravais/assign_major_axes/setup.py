from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("assign_major_axes", ["assign_major_axes.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"]),
]

setup(
    name="assign_major_axes",
    ext_modules=cythonize(extensions, annotate=False),
)
