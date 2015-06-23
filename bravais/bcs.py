__author__ = 'Ryan'

__all__ = ["get_corners", "get_load_points", "get_center_planes", "get_pinned_node", 'find_element', 'get_face_nodes']

from unit_cells import SimpleCubic, FC_Pts, FCC
import numpy as np
from cpp_utils import *


def get_corner_pos(lattice, dimX, dimY, dimZ):
    """
    Returns the corners of each unit cell as a numpy array with `dimX x dimY x dimZ x lattice.nodes.shape[0]` rows
    and `lattice.nodes.shape[1]` columns.

    :param lattice : An instance of a single unit cell with member variables nodes and elems which define the nodal
                    coordinates and connectivity.
    :param dimX : `Int`. The number of unit cells to repeat along the x-axis.
    :param dimY : `Int`. The number of unit cells to repeat along the y-axis.
    :param dimZ : `Int`. The number of unit cells to repeat along the z-axis.
    :return : A numpy array containing the positions of the corners for each unit cell.
    """
    # form anchors
    anchors = np.empty((dimX * dimY * dimZ, 3), dtype=np.float64)
    idx = 0
    for i in xrange(dimX):
        for j in xrange(dimY):
            for k in xrange(dimZ):
                anchors[idx, 0] = i
                anchors[idx, 1] = j
                anchors[idx, 2] = k
                anchors[idx] = lattice.transformMatrix.dot(anchors[idx])
                idx += 1
    # form element lists, expand lattice points about basis
    elem_list = np.empty((dimX * dimY * dimZ, lattice.elems.shape[0], lattice.elems.shape[1], 3), dtype=np.float64)
    node_list = np.empty((dimX * dimY * dimZ, lattice.nodes.shape[0], lattice.nodes.shape[1]))
    for i in xrange(anchors.shape[0]):
        node_list[i] = lattice.nodes + anchors[i]
        for j in xrange(elem_list.shape[1]):
            for k in xrange(elem_list.shape[2]):
                elem_list[i, j, k] = node_list[i][lattice.elems[j, k]]

    # reshape elements
    # (dimX x dimY x dimZ, (number of nodes with duplicates), 3 (x,y,z position))
    elems_shape = elem_list.shape
    elem_list = elem_list.reshape((elems_shape[0], elems_shape[1] * elems_shape[2], elems_shape[3]))

    # create an array to hold output.
    # size = (dimX x dimY x dimZ, (number of corners), 3 (x,y,z position))
    elem_corners = np.empty((dimX * dimY * dimZ, lattice.nodes.shape[0], lattice.nodes.shape[1]))

    # delete the duplicates in each set of nodes for the current unit cell
    for i in range(elem_list.shape[0]):
        elem_corners[i] = np.asarray(delete_duplicates_dbl(elem_list[i]))

    return elem_corners


def get_corners(lattice_nodes, a, dimX, dimY, dimZ, full_output=False, tile_pts='SC'):
    """
    Find the indices where the corners of each unit cell occur within the tiled lattice specified by `lattice_nodes`.

    :param lattice_nodes: The nodal coordinates of the original mesh
    :param a: Lattice constant of the original unit cell
    :param dimX: `Int`. Number of repeated unit cells in the x-direction.
    :param dimY: `Int`. Number of repeated unit cells in the y-direction.
    :param dimZ: `Int`. Number of repeated unit cells in the z-direction.
    :param full_output: `Bool`.
            True = corner (x,y,z) coordinates and indices within the original nodal coordinates are output.
            False = only corner indices within the original mesh are output.
    :param tile_pts: `String`.
            'SC' = Tile simple cubic lattice.
            'FCC' = Tile face-centered cubic lattice.
            'FC_only' = Tile face-centered atoms of FCC lattice only.
    :return: Indices where the corners occur within the original nodal coordinates, and (optionally) the (x,y,z)
            positions of the corners.
    """

    # unit cell that will be tiled
    if tile_pts.lower() == 'sc': 
        base_lattice = SimpleCubic(a, num_elems=1)
    elif tile_pts.lower() == 'fc_only':
        base_lattice = FC_Pts(a, num_elems=1)
    elif tile_pts.lower() == 'fcc':
        base_lattice = FCC(a, num_elems=1)
    else:
        raise Exception("%s is not a valid set of points to tile. Please choose from FCC, SC, or FC_only." % tile_pts)
    # tile the box in space to match the number of repeat units that are in lattice_nodes
    corner_pos = get_corner_pos(base_lattice, dimX, dimY, dimZ)
    # replace the x,y,z coordinates of the indices of where the points occur in lattice_nodes
    if full_output:
        return corner_pos, np.asarray(replace_with_idx(lattice_nodes, corner_pos))
    else:
        return np.asarray(replace_with_idx(lattice_nodes, corner_pos))


def get_face_nodes(nodes, idx):
    """
    Returns the indices of all nodes which contain the maximum and minimum value along the specified coordinate
    direction.
    :param nodes: Numpy array. `[x, y, z]` coordinates of the nodal positions.
    :param idx: `Int`. Column to search `nodes` for max and min values, i.e. `idx=0` would search the first column and return
                 the max & min nodes on the faces parallel to the x-axis.
    :return: max_nodes, min_nodes. Nodes which either are located at the maximum or minimum value along the coordinate
             direction specified by idx.
    """
    # find max & min val from nodes
    max_val = nodes[:, idx].max()
    min_val = nodes[:, idx].min()

    # find max and min faces
    max_nodes = np.where(nodes[:, idx] == max_val)[0]
    min_nodes = np.where(nodes[:, idx] == min_val)[0]

    return max_nodes, min_nodes


def get_load_points(nodes, dimX, dimY, dimZ, tile_pts="SC", load_type='axial'):
    """
    Returns the nodal indices that should be involved in either a boundary or force constraint
    based on the type of loading. This function searches `nodes` for the nodes on the surface
    of the mesh (i.e. `min` and `max` values) then based on loading will return the nodes on the
    appropriate face(s) that should be constrained.

    :param nodes: Numpy array. `[x, y, z]` coordinates of the nodal positions.
    :param dimX: `Int`. Dimensions in the x-direction.
    :param dimY: `Int`. Dimensions in the y-direction.
    :param dimZ: `Int`. Dimensions in the z-direction.
    :param tile_pts: `String`. Specifies the load carrying nature of the lattice structure.
                    'SC' = Intersects simple cubic lattice points over the `min` and `max` values of nodes.
                    'FCC' = Intersects FCC lattice points over the `min` and `max` values of nodes.
                    'FC_only' = Intersects face centered lattice points over the `min` and `max` values of nodes.
    :param load_type: `String`. Specifies the type of loading the lattice will be subjected to. Can be either
                      `axial`, `shear`, or `bulk`. 
                      If `axial` indices are returned for x-max and x-min faces.
                      If `shear` indices are returned for x-max, x-min, y-max, and y-min faces.
                      If `bulk` indices are returned for x-max, x-min, y-max, y-min, z-max, and z-min faces.
    :return: Numpy arrays containing the nodal indices that correspond to nodes located on the specified face
             that are involved in a constraint for the specified load type.
    """
    # find max and min faces
    xmax_nodes, xmin_nodes = get_face_nodes(nodes, 0)
    ymax_nodes, ymin_nodes = get_face_nodes(nodes, 1)
    zmax_nodes, zmin_nodes = get_face_nodes(nodes, 2)

    # get the corners of all the unit cells
    corners = get_corners(nodes, 1.0, dimX, dimY, dimZ, tile_pts=tile_pts)

    # flatten corners into 1D array
    corners = corners.reshape((corners.shape[0] * corners.shape[1]))

    # find the load points on each face
    xmax_loadpoints = np.intersect1d(xmax_nodes, corners)
    xmin_loadpoints = np.intersect1d(xmin_nodes, corners)
    ymax_loadpoints = np.intersect1d(ymax_nodes, corners)
    ymin_loadpoints = np.intersect1d(ymin_nodes, corners)
    zmax_loadpoints = np.intersect1d(zmax_nodes, corners)
    zmin_loadpoints = np.intersect1d(zmin_nodes, corners)

    if load_type.lower() == 'axial':
        return xmax_loadpoints, xmin_loadpoints
    elif load_type.lower() == 'shear':
        return xmax_loadpoints, xmin_loadpoints, ymax_loadpoints, ymin_loadpoints
    elif load_type.lower() == 'bulk':
        return xmax_loadpoints, xmin_loadpoints, ymax_loadpoints, ymin_loadpoints, zmax_loadpoints, zmin_loadpoints
    else:
        raise Exception("%s is not a valid load type. Please choose from 'axial', 'bulk', or 'shear'." % load_type)


def get_center_planes(nodes):
    """
    returns the indices of all nodes located in the central x-plane and central z-plane.

    :param nodes: Numpy array. `[x, y, z]` coordinates of the nodal positions.
    :return: Numpy arrays of indices located at `(xmax - xmin)/2` and `(zmax - zmin)/2`, respectively.
    """
    # find max and min faces
    xmax = nodes[:, 0].max()
    xmin = nodes[:, 0].min()
    zmax = nodes[:, 2].max()
    zmin = nodes[:, 2].min()

    # find center symmetry planes
    yz_centerplane = np.where(nodes[:, 0] == (xmax - xmin)/2)[0]
    xy_centerplane = np.where(nodes[:, 2] == (zmax - zmin)/2)[0]

    return yz_centerplane, xy_centerplane


def get_pinned_node(nodes, position):
    """
    Based on the lattice type, this function will searches for a node that can be pinned without affecting
    the response of the lattice when loaded axially or in a state or shear (will not work for bulk loading).
    :param position: `String`. Specifies the position of the pinned node.
                      If 'global_center', the node at `[xmid, ymid, zmid]` is returned, where `xmid = (xmax + xmin)/2.0`.
                      If 'global_mid_top', the node at `[xmid, ymax, zmid]` is returned, where `xmid = (xmax + xmin)/2.0`.
    :return: Int. Index of the node satisfying the required positional constraints.
    """
    # find max and min faces
    xmax = nodes[:, 0].max()
    xmin = nodes[:, 0].min()
    ymax = nodes[:, 1].max()
    ymin = nodes[:, 1].min()
    zmax = nodes[:, 2].max()
    zmin = nodes[:, 2].min()

    xmid = (xmax + xmin) / 2.0
    ymid = (ymax + ymin) / 2.0
    zmid = (zmax + zmin) / 2.0

    p = position.lower()

    if p == 'global_mid_top':
        pinned_node = np.where((nodes[:, 1] == ymax) & (nodes[:, 0] == xmid) & (nodes[:, 2] == zmid))[0]

    elif p == 'global_center':
        pinned_node = np.where((nodes[:, 1] == ymid) & (nodes[:, 0] == xmid) & (nodes[:, 2] == zmid))[0]

    else:
        raise Exception("%s is not a position. Please choose from 'global_center' or 'shear'." % position)
    return pinned_node[0]


def find_element(job, base_lattice, elem_idx, base_dims):
    """
    Finds the element in `base_lattice` indexed by `elem_idx` in the `job`'s elements.
    :param job : `BravaisJob`. Object that contains the node list and element list for the mesh.
    :param base_lattice : The repeated unit cell that makes up the input job.
    :param elem_idx : `Int`. the index of the element within the base_lattice to find in the job's element list.
    :param base_dims : `array`. The `[x,y,z]` offset from unit cell `[0, 0, 0]` to search for the element.
    :return: `Int`. Index of the element.
    """

    elem = np.empty((1, base_lattice.elems.shape[1], base_lattice.nodes.shape[1]))
    for i in xrange(base_lattice.elems.shape[1]):
        elem[0, i, :] = base_lattice.nodes[base_lattice.elems[elem_idx][i]]
        elem[0, i, :] += base_lattice.transformMatrix.dot(base_dims)

    global_elem = np.asarray(replace_with_idx(job.nodes, elem))

    return replace_with_idx_int(job.elems, global_elem)[0]
