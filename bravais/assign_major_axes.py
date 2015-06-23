import numpy as np
from math import cos, sin, sqrt, acos
from numba import jit, float64

__all__ = ["assign_major_axes"]

@jit
def get_yz_rotation(vec):
    """
    Returns the rotations around the z and y axes required such that the local x-axis is aligned along the vector's
    length.

    :param vec : Numpy array with in the form of [x,y,z] corresponding to vector's scale values along the global
    coordinate axes.
    """
    # get the length of the x-y components of the vector
    xy_length = sqrt(vec[0] * vec[0] + vec[1] * vec[1])
    # get the length of the entire vector
    vec_length = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    # calculate the rotations about the y and z axes:
    # if there is no xyLength then the element is already aligned in the z direction
    if xy_length == 0:
        zrot = 0.0
        yrot = np.pi / 2.0
        # check if vector along z points up or down
        if vec[0] < 0:
            yrot *= -1.0
    # if not aligned with z-axis calculate the rotations normally
    else:
        zrot = -acos(vec[0] / xy_length)
        val = (vec[0] * vec[0] + vec[1] * vec[1]) / (xy_length * vec_length)
        # check that the value is in the valid range of acos (mainly for rounding errors when value approaches +/-1)
        val = min(1, max(val, -1))
        yrot = acos(val)
    # adjust rotation to correct sign based on quadrant the vector is in
    if vec[1] < 0.0:
        zrot *= -1.0
    if vec[2] < 0.0:
        yrot *= -1.0
    # put results in numpy array
    return yrot, zrot

@jit(float64(float64, float64, float64))
def rotate_coords(xRot, yRot, zRot):
    """
    Returns the local x, y, and z coordinate vectors that result from rotating the global x,y, and z vectors by the
    amounts xRot, yRot, and zRot, respectively.

    :param xRot : Clockwise rotation in radians to apply about the x-axis
    :param yRot : Clockwise rotation in radians to apply about the y-axis
    :param zRot : Clockwise rotation in radians to apply about the z-axis
    """
    # define the unit vectors along the global x,y,z directions
    globalCoords = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # get rotation matrices
    Rx = np.matrix([[1.0, 0.0, 0.0],
                    [0.0, cos(xRot), -sin(xRot)],
                    [0.0, sin(xRot), cos(xRot)]])
    Ry = np.matrix([[cos(yRot), 0.0, sin(yRot)],
                    [0.0, 1.0, 0.0],
                    [-sin(yRot), 0, cos(yRot)]])
    Rz = np.matrix([[cos(zRot), -sin(zRot), 0.0],
                    [sin(zRot), cos(zRot), 0.0],
                    [0.0, 0.0, 1.0]])
    # combine rotation matrices and apply to global coordinates
    return (Rx * Ry * Rz) * globalCoords


def get_major_axis(pt1, pt2, xRot):
    """
    A vector is defined from pt1 to pt2, and a local coordinate system is defined where the x-axis runs parallel to the
    vector and the z-axis is coplanar with the global z-axis. The variable xRot will the apply a rotation about the
    x-axis by the specified amount and return the resultant z-axis unit vector in the local rotated coordinate system.

    :param pt1  : Point in global (x,y,z) space
    :param pt2  : Point in global (x,y,z) space
    :param xRot : Clockwise rotation about the x-axis that should be applied in local coordinate space.
    """
    vec = pt2 - pt1
    yrot, zrot = get_yz_rotation(vec)
    return rotate_coords(xRot, yrot, zrot)[2]


def assign_major_axes(mesh, xRot=0.0):
    """
    Takes a mesh an input and returns an array holding the major bending axis (in [x,y,z] form) for each element.

    :param mesh : an object which has the member variable nodes which holds the (x,y,z) coordinates of each node in the
    mesh and the member variable elems holding an array of describing the connectivity of the nodal coordinates.
    :param xRot : The amount the major axis is rotated clockwise relative to the z-axis of the element if it were
    aligned along the x-axis and you were looking down the element from the end point to the first point.
    """
    numElems = mesh.elems.shape[0]
    majorAxes = np.empty((numElems, 3))
    for i in range(numElems):
        pt1 = mesh.nodes[mesh.elems[i, 0]]
        pt2 = mesh.nodes[mesh.elems[i, 1]]
        majorAxes[i] = get_major_axis(pt1, pt2, xRot)
    # set small values in transformation matrix to 0
    majorAxes[np.abs(majorAxes) < 1e-15] = 0
    return majorAxes