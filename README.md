Bravais Package
===============

This package contains tools required to analyze truss structures based off of Bravais lattices using Abaqus.
It is divided into 6 sections:

1. `unit_cells` contains all of the elementary unit cells that can be meshed in three dimensions.
2. `mesher` takes the unit cell and creates a 3D mesh of specified dimensions.
3. `bcs` finds nodes and returns information regarding the necessary boundary conditions for a given analysis.
4. `inp_creator` creates .inp files to be analyzed via Abaqus.
5. `abaqus` executes Abaqus using the command line to analyze and extract data.
6. `post_process` takes nodal dispalcement and force data and calculates elastic properties.

Usage
-----
First compile the cpp_utils extension module: Navigate to the ~/bravais/cpp_utils directory and execute `python setup.py build_ext --inplace`.
The module can be used by importing the required tools, i.e. `from bravais import unit_cells, mesher` if a mesh needs to be created. A simple
example that creates an inp file on Linux then runs the FE calculation on Windows is provided in the examples folder.

Contributor(s)
------------
 * Ryan Latture
