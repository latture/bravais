__author__ = 'Ryan'

__all__ = ["Job", "BravaisJob", "mesh_bravais", "mesh_rhombohedral"]

from cpp_utils import *
from python_utils import sort_rows
import numpy as np


class Job:
    """
    Holds the nodal coordinate positions in nodes, the connectivity of the nodes in elems,
    and the properties associated with each element in props.
    """
    def __init__(self, nodes, elems, props=None):
        self.nodes = nodes
        self.elems = elems
        self.props = props


class BravaisJob(Job):
    """
    Subclass of Job which store additional information about the dimensions of the final mesh in terms of the number
    of repeated unit cells and the volume of the total mesh.
    """
    def __init__(self, nodes, elems, dimX, dimY, dimZ, volume):
        Job.__init__(self, nodes, elems)
        self.dimX = dimX
        self.dimY = dimY
        self.dimZ = dimZ
        self.volume = volume
        self.totalLength = None

    def merge(self, job, return_elem_indices=False):
        """
        Merges the node and element lists from the input Job into the parent Job
        :param job: Job. Contains the node and element lists to merge into current Job's corresponding member variables.
        """
        # pull out coordinates of each element set
        elems_orig = self.nodes[self.elems]
        elems_new = job.nodes[job.elems]

        # concatenate the 2 element arrays
        elems = np.concatenate((elems_orig, elems_new))

        # concatenate the 2 nodal array and sort
        nodes = np.concatenate((self.nodes, job.nodes))
        nodes = sort_rows(nodes)

        # delete any duplicates
        nodes = np.asarray(delete_duplicates_dbl(nodes))

        # replace nodal coordinates with index it occurs in the nodes
        elems = np.asarray(replace_with_idx(nodes, elems))
        elems_orig = elems[:elems_orig.shape[0]]
        elems_new = elems[elems_orig.shape[0]:]

        # sort result and delete duplicates
        elems.sort()
        elems = np.asarray(delete_duplicates_int(elems))

        # store result as member variables
        self.nodes = nodes
        self.elems = elems

        if return_elem_indices:
            elems_orig_idx = replace_with_idx_int(elems, elems_orig)
            elems_new_idx = replace_with_idx_int(elems, elems_new)
            return elems_orig_idx, elems_new_idx

    def _calc_total_length(self):
        """
        Calculates the total length of all the struts.
        """
        assert self.totalLength is None, "Total length is already initialized."
        self.totalLength = 0.0
        for i in xrange(self.elems.shape[0]):
            idx1 = self.elems[i, 0]
            idx2 = self.elems[i, 1]
            n1 = self.nodes[idx1]
            n2 = self.nodes[idx2]
            dist = np.linalg.norm(n2 - n1)
            self.totalLength += dist

    def calc_cross_section(self, volume):
        """
        Calculates the cross-sectional area requires for the struts to occupy the specified volume.
        :param volume: the volume the struts should occupy.
        :return: The constant cross-sectional area of the struts.
        """
        if self.totalLength is None:
            self._calc_total_length()

        return volume/self.totalLength

    def calc_volume(self, crossSectArea):
        """
        Returns the total volume the struts occupy defined by the lattice and their cross-sectional area.
        :param crossSectArea: The cross-sectional area of the struts.
        :return: The occupied volume.
        """
        if self.totalLength is None:
            self._calc_total_length()

        return crossSectArea * self.totalLength

    def calc_rel_density(self, crossSectArea):
        """
        Calculates the relative density of the lattice based on the cross-sectional area of the struts.
        :param crossSectArea: The cross-sectional area of the struts
        """
        occupied_volume = self.calc_volume(crossSectArea)
        return occupied_volume/self.volume

'''
Tile unit cells
'''


def mesh_bravais(lattice, dimX, dimY, dimZ):
    """
    Tiles the unit cell defined by lattice by the specified number of repeat units along the respective axes.

    :param lattice: An instance of a single unit cell with member variables nodes and elems which define the nodal
                    coordinates and connectivity.
    :param dimX: int. The number of unit cells to repeat along the x-axis.
    :param dimY: int. The number of unit cells to repeat along the y-axis.
    :param dimZ: int. The number of unit cells to repeat along the z-axis.
    :return: A new instance of the Job class which holds the nodal coordinated and connectivity of the input lattice
            tiled by the specified dimensions.
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
    # delete duplicates
    node_list = node_list.reshape((node_list.shape[0] * node_list.shape[1], 3))
    node_list = np.asarray(delete_duplicates_dbl(node_list))
    # replace the elem_list (x,y,z) coordinates with the index the point occurs in the node list
    elems_shape = elem_list.shape
    elem_list = elem_list.reshape((elems_shape[0] * elems_shape[1], elems_shape[2], elems_shape[3]))
    elem_list = np.asarray(replace_with_idx(node_list, elem_list))
    elem_list = np.asarray(delete_duplicates_int(elem_list))

    # form output, store information about dimensions and volume in object
    return BravaisJob(node_list, elem_list, dimX, dimY, dimZ, lattice.volume * dimX * dimY * dimZ)


def mesh_rhombohedral(lattice, dimX, dimY, dimZ):
    """
    Tiles the unit cell defined by lattice by the specified number of repeat units along the respective axes.
    This version of mesh generation is intended for use with the `Rhombohedral` Bravais lattice when unit cells
    should be tiled corner-to-corner instead of face-to-face.

    :param lattice: An instance of a single unit cell with member variables nodes and elems which define the nodal
                    coordinates and connectivity.
    :param dimX: int. The number of unit cells to repeat along the x-axis.
    :param dimY: int. The number of unit cells to repeat along the y-axis.
    :param dimZ: int. The number of unit cells to repeat along the z-axis.
    :return: A new instance of the Job class which holds the nodal coordinated and connectivity of the input lattice
            tiled by the specified dimensions.
    """
    # form anchors
    anchors = np.empty((dimX * dimY * dimZ, 3), dtype=np.float64)
    anchor = np.empty(3, dtype=np.float64)
    next_level_vector = lattice.transformMatrix.dot(np.array([1.0, 1.0, 1.0]))
    idx = 0
    for i in xrange(dimX):
        for j in xrange(dimY):
            for k in xrange(-(dimX + dimY + 1), 1):
                if (i + j + k) == 0:
                    anchor[0] = i
                    anchor[1] = j
                    anchor[2] = k
                    anchor = lattice.transformMatrix.dot(anchor)

                    for l in xrange(dimZ):
                        anchors[idx + l * (dimX * dimY)] = anchor + next_level_vector * l

                    idx += 1
        
    # form element lists, expand lattice points about basis
    elem_list = np.empty((dimX * dimY * dimZ, lattice.elems.shape[0], lattice.elems.shape[1], 3), dtype=np.float64)
    node_list = np.empty((dimX * dimY * dimZ, lattice.nodes.shape[0], lattice.nodes.shape[1]))
    for i in xrange(anchors.shape[0]):
        node_list[i] = lattice.nodes + anchors[i]
        for j in xrange(elem_list.shape[1]):
            for k in xrange(elem_list.shape[2]):
                elem_list[i, j, k] = node_list[i][lattice.elems[j, k]]
    # delete duplicates
    node_list = node_list.reshape((node_list.shape[0] * node_list.shape[1], 3))
    node_list = np.asarray(delete_duplicates_dbl(node_list))
    # replace the elem_list (x,y,z) coordinates with the index the point occurs in the node list
    elems_shape = elem_list.shape
    elem_list = elem_list.reshape((elems_shape[0] * elems_shape[1], elems_shape[2], elems_shape[3]))
    elem_list = np.asarray(replace_with_idx(node_list, elem_list))
    elem_list = np.asarray(delete_duplicates_int(elem_list))

    # form output, store information about dimensions and volume in object
    return BravaisJob(node_list, elem_list, dimX, dimY, dimZ, lattice.volume * dimX * dimY * dimZ)
