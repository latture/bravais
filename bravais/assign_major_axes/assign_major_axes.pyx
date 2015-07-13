import numpy as np

cimport numpy as np
from libc.math cimport cos, sin, sqrt, acos
cimport cython

cdef inline double dbl_max(double a, double b): return a if a >= b else b
cdef inline double dbl_min(double a, double b): return a if a <= b else b

__all__ = ["assign_major_axes"]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef void get_yz_rotation(double *vec_in, double *vec_out):
    """
    Populates `vec_out` with the rotations around the z and y axes required such that the local x-axis
    is aligned along `vec_in`.

    :param vec_in  : `double*`. Pointer to the [dx, dy, dz] vector values along the global coordinate axes.
    :param vec_out : `double*`. Variable to store the y and z rotation values.
    """
    cdef:
        # get the length of the x-y components of the vector
        double xy_length = sqrt(vec_in[0] * vec_in[0] + vec_in[1] * vec_in[1])

        # get the length of the entire vector
        double vec_length = sqrt(vec_in[0] * vec_in[0] + vec_in[1] * vec_in[1] + vec_in[2] * vec_in[2])
        double zrot, yrot, val

    # calculate the rotations about the y and z axes:
    # if there is no xyLength then the element is already aligned in the z direction
    if xy_length < 1.e-7:
        zrot = 0.0
        yrot = 1.57079632679489661923 # = pi/2
        # check if vector along z points up or down
        if vec_in[0] < 0:
            yrot *= -1.0
    # if not aligned with z-axis calculate the rotations normally
    else:
        zrot = -acos(vec_in[0] / xy_length)
        val = (vec_in[0] * vec_in[0] + vec_in[1] * vec_in[1]) / (xy_length * vec_length)
        # check that the value is in the valid range of acos (mainly for rounding errors when value approaches +/-1)
        val = dbl_min(1, dbl_max(val, -1))
        yrot = acos(val)
    # adjust rotation to correct sign based on quadrant the vector is in
    if vec_in[1] < 0.0:
        zrot *= -1.0
    if vec_in[2] < 0.0:
        yrot *= -1.0
    # store results in return array
    vec_out[0] = yrot
    vec_out[1] = zrot


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void rotate_coords(double xrot, double yrot, double zrot, double *vec_out):
    """
    Returns the local x, y, and z coordinate vectors that result from rotating the global x, y, and z vectors by the
    amounts xRot, yRot, and zRot, respectively.

    :param xrot    : `double`. Clockwise rotation in radians to apply about the x-axis.
    :param yrot    : `double`. Clockwise rotation in radians to apply about the y-axis.
    :param zrot    : `double`. Clockwise rotation in radians to apply about the z-axis.
    :param vec_out : `double*`. Variable to store the result.
    """
    cdef:
        double cx = cos(xrot)
        double sx = sin(xrot)
        double cy = cos(yrot)
        double sy = sin(yrot)
        double cz = cos(zrot)
        double sz = sin(zrot)

    vec_out[0] = -cx * cz * sy + sx * sz
    vec_out[1] = cz * sx + cx * sy * sz
    vec_out[2] = cx * cy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void get_major_axis(double *delta, double xrot, double *vec_out):
    """
    `delta` is defined as the vector between 2 points. A local coordinate system is defined where the x-axis
    runs parallel to the vector and the z-axis is coplanar with the global z-axis. The variable xrot will the apply
    a rotation about the x-axis by the specified amount and return the resultant z-axis unit vector in
    the local rotated coordinate system.

    :param delta  : `double*`. Pointer to the vector between 2 points in 3D space, i.e. [dx, dy, dz].
    :param xrot : Clockwise rotation about the x-axis that should be applied in local coordinate space.
    :param vec_out : `double*`. Array to store the resultant vector, should have space for 3 doubles.
    """
    cdef:
        double rotvec[2]
        double yrot, zrot 

    get_yz_rotation(delta, &rotvec[0])
    yrot = rotvec[0]
    zrot = rotvec[1]
    rotate_coords(xrot, yrot, zrot, vec_out)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def assign_major_axes(double[:, ::1] nodes, long[:, ::1] elems, double xrot=0.0):
    """
    Takes a mesh an input and returns an array holding the major bending axis (in [x,y,z] form) for each element.

    :param nodes : `numpy.ndarray`, dtype=`double`. Holds the (x,y,z) coordinates of each node in the job.
    :param elems : `numpy.ndarray`, dtype=`long`. Holds indices of `nodes` that describe the connectivity of the nodal coordinates.
    :param xrot : The amount the major axis is rotated clockwise relative to the z-axis of the element if it were
    aligned along the x-axis and you were looking down the element from the end point to the first point.
    """
    cdef:
        long i, j
        long num_elems = elems.shape[0]
        long dofs_per_node = 3
        double[:, ::1] major_axes = np.empty((num_elems, dofs_per_node))
        # double *major_axes_ptr = &major_axes[0,]
        double[::1] axis = np.empty(dofs_per_node)
        double[::1] delta = np.empty(dofs_per_node)
        double *delta_ptr = &delta[0]
        double *axis_ptr = &axis[0]
        double *pt1
        double *pt2

    for i in xrange(num_elems):
        pt1 = &nodes[elems[i, 0], 0]
        pt2 = &nodes[elems[i, 1], 0]
        for j in xrange(dofs_per_node):
            delta[j] = pt2[j] - pt1[j]
        get_major_axis(delta_ptr, xrot, axis_ptr)
        for j in xrange(dofs_per_node):
            major_axes[i, j] = axis[j]

    return major_axes
