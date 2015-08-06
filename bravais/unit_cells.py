__author__ = 'Ryan'

__all__ = ["SimpleCubic", "FCC", "BCC", "Hexagonal", "Rhombohedral", "PrimitiveTetragonal", "BodyCenteredTetragonal",
           "PrimitiveOrthorhombic", "BodyCenteredOrthorhombic", "BaseCenteredOrthorhombic", "FaceCenteredOrthorhombic",
           "PrimitiveMonoclinic", "BaseCenteredMonoclinic", "Triclinic", "Hexagonal_v2", "FC_Pts",
           "Square", "Cross", "FCC_Primitive", "DiamondCubic", "Kagome", "UnitData", "Triangulated"]


from cpp_utils import *
import numpy as np
from math import cos, sin, sqrt
from python_utils import sort_rows
from collections import namedtuple


class UnitData(namedtuple("UnitData", ["unit_cells", "label"])):
    def __new__(cls, unit_cells, label):
        assert type(label) is str, "label must be a string."
        return super(UnitData, cls).__new__(cls, unit_cells, label)


'''
Parent Bravais lattice class
'''


class BravaisLattice(object):
    """
    Parent class to all bravais lattice unit cells. Contains the parameters which describe a general unit cell.
    From the 6 general parameters the transformation matrix is calculated and stored as a member variable.
    """
    def __init__(self, a=1.0, b=1.0, c=1.0, alpha=np.pi / 2.0, beta=np.pi / 2.0, gamma=np.pi / 2.0, num_elems=1):
        """
        Initialzes a bravais unit cell. The 6 parameters which describe the orientation of the primitive vectors are
        required. If a parameter is not specified, then the default is 1.0 for the length (a, b, c) or 90 degrees
        for the angle between vectors (alpha, beta , gamma), i.e. if no parameters are passed a unit cube is
        constructed.

        :param a: Axial distance of the primitive vector along the 1st axis.
        :param b: Axial distance of the primitive vector along the 2nd axis.
        :param c: Axial distance of the primitive vector along the 3rd axis.
        :param alpha: Angle between the 2nd and 3rd axes.
        :param beta: Angle between the 1st and 3rd axes.
        :param gamma: Angle between the 1st and 2nd axes.
        :param num_elems: The number of elements along each axial length.
        :return: BravaisLattice object.
        """
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_elems = num_elems
        self.volume = sqrt(
            1.0 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2 + 2.0 * cos(alpha) * cos(beta) * cos(gamma))
        self.transformMatrix = np.array([[a, b * cos(gamma), c * cos(beta)],
                                         [0.0, b * sin(gamma), c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)],
                                         [0.0, 0.0, c * self.volume / sin(gamma)]])
        # set small values in transformation matrix to 0
        self.transformMatrix[np.abs(self.transformMatrix) < 1e-15] = 0

        # set the total length of the elements to None: initialized by calling calculate cross section
        self.total_length = None

    def create_basis_mesh(self):
        """
        Fills the nodes and elems member variables with the appropriate nodal coordinates and connectivity list.
        For each strut between 2 nodes, the span defined by the basis_elems and basis_nodes variables is divided into
        num_elems + 1 nodes. The node list is populated with the increased number of nodes and the element list is
        ordered based on the new nodes.
        """
        # create arrays to hold the elements/nodes along each span of basis strut
        span_nodes = np.empty((self.num_elems + 1, 3), dtype=np.float64)
        span_elems = np.empty((self.num_elems, 2, 3), dtype=np.float64)

        # loop through each span and create num_elems elements along the strut
        for i in xrange(len(self.basis_elems)):
            pt1 = self.basis_nodes[self.basis_elems[i][0]]
            pt2 = self.basis_nodes[self.basis_elems[i][1]]
            xpts = np.linspace(pt1[0], pt2[0], self.num_elems + 1)
            ypts = np.linspace(pt1[1], pt2[1], self.num_elems + 1)
            zpts = np.linspace(pt1[2], pt2[2], self.num_elems + 1)

            # form (x,y,z) coordinates
            for j in xrange(self.num_elems + 1):
                span_nodes[j, 0] = xpts[j]
                span_nodes[j, 1] = ypts[j]
                span_nodes[j, 2] = zpts[j]
            # add form element list for span (contains [x,y,z] entries for each node in the element)
            for j in xrange(self.num_elems):
                span_elems[j, 0, :] = span_nodes[j]
                span_elems[j, 1, :] = span_nodes[j + 1]
            # add the spans nodes/elements to the instance's nodes/elements
            self.nodes[i] = span_nodes
            self.elems[i] = span_elems

        self.nodes = np.round(self.nodes, decimals=6)
        self.elems = np.round(self.elems, decimals=6)
        # flatten the node list into an array of (x,y,z) points
        self.nodes = self.nodes.reshape((self.basis_elems.shape[0] * (self.num_elems + 1), 3))
        # cleanup any duplicate nodes
        self.nodes = np.asarray(delete_duplicates_dbl(self.nodes))
        # sort the nodes
        self.nodes = sort_rows(self.nodes)
        # flatten the element list which contains arrays of elements list for each
        # span to one array with all the elements (still contains (x,y,z) coordinates as entries)
        elemsShape = self.elems.shape
        self.elems = self.elems.reshape((elemsShape[0] * elemsShape[1], elemsShape[2], elemsShape[3]))
        # replace the (x,y,z) coordinate with the index it occurs in the node list
        self.elems = np.asarray(replace_with_idx(self.nodes, self.elems))
        # transform nodes from fraction coordinate system to Cartesion coordinates
        for i in xrange(self.nodes.shape[0]):
            self.nodes[i] = self.transformMatrix.dot(self.nodes[i])

    def calc_cross_section(self, volume):
        """
        Calculates the constant cross section required if the unit cell is constructed with the specified volume.
        :param volume: The amount of volume allocated to the struts of the unit cell.
        :return: The constant cross-section for all struts such that the volume of the unit cell is that of
        the specified amount.
        """
        if self.total_length is None:
            # set initial length to 0
            self.total_length = 0.0

            # transform basis nodal positions. The unit cell nodes are just subdividing the basis span, so calculating
            # the length from the transformed basis will result in the same length as doing each element in self.elems
            # one at a time.
            basis_tranformed = np.empty_like(self.basis_nodes)
            for i in xrange(self.basis_nodes.shape[0]):
                basis_tranformed[i] = self.transformMatrix.dot(self.basis_nodes[i])

            # loop through elements and add the length of the element to the total length
            for i in xrange(self.basis_elems.shape[0]):
                idx1 = self.basis_elems[i, 0]
                idx2 = self.basis_elems[i, 1]
                n1 = basis_tranformed[idx1]
                n2 = basis_tranformed[idx2]
                dist = np.linalg.norm(n2 - n1)
                self.total_length += dist

        return volume/self.total_length

############
# Subclasses
############


class Square(BravaisLattice):
    def __init__(self, a, b=1.0, c=1.0, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0]])
        self.basis_elems = np.array([[0, 1],
                                    [0, 2],
                                    [1, 3],
                                    [2, 3]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class Cross(BravaisLattice):
    def __init__(self, a, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.5, 0.5],
                                    [0.5, 0.0, 0.5],
                                    [0.5, 0.5, 0.0],
                                    [0.5, 0.5, 0.5],
                                    [1.0, 0.5, 0.5],
                                    [0.5, 1.0, 0.5],
                                    [0.5, 0.5, 1.0]])

        self.basis_elems = np.array([[0, 3],
                                    [1, 3],
                                    [2, 3],
                                    [4, 3],
                                    [5, 3],
                                    [6, 3]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class SimpleCubic(BravaisLattice):
    def __init__(self, a, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basis_elems = np.array([[0, 1],
                                    [0, 2],
                                    [0, 4],
                                    [1, 3],
                                    [1, 5],
                                    [2, 3],
                                    [2, 6],
                                    [3, 7],
                                    [4, 5],
                                    [4, 6],
                                    [5, 7],
                                    [6, 7]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class FC_Pts(BravaisLattice):
    def __init__(self, a, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.5, 0.5],
                                    [0.5, 0.0, 0.5],
                                    [0.5, 0.5, 0.0],
                                    [1.0, 0.5, 0.5],
                                    [0.5, 1.0, 0.5],
                                    [0.5, 0.5, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [1, 3], [1, 5], [2, 3], [2, 4],
            [3, 4], [3, 5], [4, 5]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class FCC(BravaisLattice):
    def __init__(self, a, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.0, 0.5, 0.5],
                                    [0.5, 0.0, 0.5],
                                    [0.5, 0.5, 0.0],
                                    [1.0, 0.5, 0.5],
                                    [0.5, 1.0, 0.5],
                                    [0.5, 0.5, 1.0]])

        self.basis_elems = np.array(
            [[0, 8], [0, 9], [0, 10],
             [1, 8], [1, 9], [1, 13],
             [2, 8], [2, 10], [2, 12],
             [3, 9], [3, 10], [3, 11],
             [4, 8], [4, 12], [4, 13],
             [5, 9], [5, 11], [5, 13],
             [6, 10], [6, 11], [6, 12],
             [7, 11], [7, 12], [7, 13],
             [8, 9], [8, 10], [8, 12], [8, 13],
             [9, 10], [9, 11], [9, 13],
             [10, 11], [10, 12],
             [11, 12], [11, 13],
             [12, 13]])

        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class FCC_Primitive(BravaisLattice):
    def __init__(self, a, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.5, 0.5],
                                    [0.5, 0.0, 0.5],
                                    [0.5, 0.5, 0.0],
                                    [0.5, 0.5, 1.0],
                                    [0.5, 1.0, 0.5],
                                    [1.0, 0.5, 0.5],
                                    [1.0, 1.0, 1.0]])

        self.basis_elems = np.array(
            [[0, 1],
             [0, 2],
             [0, 3],
             [1, 4],
             [1, 5],
             [2, 4],
             [2, 6],
             [3, 5],
             [3, 6],
             [4, 7],
             [5, 7],
             [6, 7]])

        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class BCC(BravaisLattice):
    def __init__(self, a, num_elems=1, inverted=False):
        BravaisLattice.__init__(self, a=a, b=a, c=a, num_elems=num_elems)

        if inverted:
            self.basis_nodes = np.array([[0.0, 0.5, 0.0],
                                        [0.0, 0.5, 1.0],
                                        [0.5, 0.0, 0.5],
                                        [0.5, 1.0, 0.5],
                                        [1.0, 0.5, 0.0],
                                        [1.0, 0.5, 1.0]])
            self.basis_elems = np.array(
                [[0, 2], [0, 3], [1, 2], [1, 3], [2, 4], [2, 5], [3, 4], [3, 5]])
        else:
            self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 1.0, 1.0],
                                        [0.5, 0.5, 0.5],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 0.0],
                                        [1.0, 1.0, 1.0]])
            self.basis_elems = np.array(
                [[0, 4], [1, 4], [2, 4], [3, 4],
                 [4, 5], [4, 6], [4, 7], [4, 8]])

        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class DiamondCubic(BravaisLattice):
    def __init__(self, a, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, num_elems=num_elems)
        self.basis_nodes = np.array([
                                    [-0.25,  -0.25,  0.25],
                                    [-0.25,  -0.25,  1.25],
                                    [-0.25,  0.25,  -0.25],
                                    [-0.25,  0.25,  0.75],
                                    [-0.25,  0.75,  0.25],
                                    [-0.25,  0.75,  1.25],
                                    [-0.25,  1.25,  -0.25],
                                    [-0.25,  1.25,  0.75],

                                    [0.0,  0.0,  0.0],
                                    [0.0,  0.0,  1.0],
                                    [0.0,  0.5,  0.5],
                                    [0.0,  1.0,  0.0],
                                    [0.0,  1.0,  1.0],

                                    [0.25,  -0.25,  -0.25],
                                    [0.25,  -0.25,  0.75],
                                    [0.25,  0.25,  0.25],
                                    [0.25,  0.25,  1.25],
                                    [0.25,  0.75,  -0.25],
                                    [0.25,  0.75,  0.75],
                                    [0.25,  1.25,  0.25],
                                    [0.25,  1.25,  1.25],

                                    [0.5,  0.0,  0.5],
                                    [0.5,  0.5,  0.0],
                                    [0.5,  0.5,  1.0],
                                    [0.5,  1.0,  0.5],

                                    [0.75,  -0.25,  0.25],
                                    [0.75,  -0.25,  1.25],
                                    [0.75,  0.25,  -0.25],
                                    [0.75,  0.25,  0.75],
                                    [0.75,  0.75,  0.25],
                                    [0.75,  0.75,  1.25],
                                    [0.75,  1.25,  -0.25],
                                    [0.75,  1.25,  0.75],

                                    [1.0,  0.0,  0.0],
                                    [1.0,  0.0,  1.0],
                                    [1.0,  0.5,  0.5],
                                    [1.0,  1.0,  0.0],
                                    [1.0,  1.0,  1.0],

                                    [1.25,  -0.25,  -0.25],
                                    [1.25,  -0.25,  0.75],
                                    [1.25,  0.25,  0.25],
                                    [1.25,  0.25,  1.25],
                                    [1.25,  0.75,  -0.25],
                                    [1.25,  0.75,  0.75],
                                    [1.25,  1.25,  0.25],
                                    [1.25,  1.25,  1.25]])

        self.basis_elems = np.array([[0, 8],
                                    [1, 9],
                                    [2, 8],
                                    [3, 9], [3, 10],
                                    [4, 10], [4, 11],
                                    [5, 12],
                                    [6, 11],
                                    [7, 12],
                                    [8, 13], [8, 15],
                                    [9, 14], [9, 16],
                                    [10, 15], [10, 18],
                                    [11, 17], [11, 19],
                                    [12, 18], [12, 20],
                                    [14, 21],
                                    [15, 21], [15, 22],
                                    [16, 23],
                                    [17, 22],
                                    [18, 23], [18, 24],
                                    [19, 24],
                                    [21, 25], [21, 28],
                                    [22, 27], [22, 29],
                                    [23, 28], [23, 30],
                                    [24, 29], [24, 32],
                                    [25, 33],
                                    [26, 34],
                                    [27, 33],
                                    [28, 34], [28, 35],
                                    [29, 35], [29, 36],
                                    [30, 37],
                                    [31, 36],
                                    [32, 37],
                                    [33, 38], [33, 40],
                                    [34, 39], [34, 41],
                                    [35, 40], [35, 43],
                                    [36, 42], [36, 44],
                                    [37, 43], [37, 45]])

        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class Hexagonal(BravaisLattice):
    def __init__(self, a, c, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=c, gamma=np.pi / 3, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 4], [2, 6],
             [3, 5], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3), dtype=np.float64)
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3), dtype=np.float64)
        self.create_basis_mesh()


class Hexagonal_v2(BravaisLattice):
    def __init__(self, a, c, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=c, gamma=np.pi / 3, num_elems=num_elems)
        self.basis_nodes = np.array([[1.0/3.0, 2.0/3.0, 0.25],
                                    [1.0/3.0, 5.0/3.0, 0.25],
                                    [4.0/3.0, 5.0/3.0, 0.25],
                                    [4.0/3.0, 2.0/3.0, 0.25],
                                    [2.0/3.0, 1.0/3.0, 0.75],
                                    [2.0/3.0, 4.0/3.0, 0.75],
                                    [5.0/3.0, 4.0/3.0, 0.75],
                                    [4.0/3.0, 1.0/3.0, 0.75],
                                    [1.0/3.0, 2.0/3.0, 1.25],
                                    [1.0/3.0, 5.0/3.0, 1.25],
                                    [4.0/3.0, 5.0/3.0, 1.25],
                                    [4.0/3.0, 2.0/3.0, 1.25],
                                    [2.0/3.0, 1.0/3.0, 1.75],
                                    [2.0/3.0, 4.0/3.0, 1.75],
                                    [5.0/3.0, 4.0/3.0, 1.75],
                                    [4.0/3.0, 1.0/3.0, 1.75]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 5], [2, 3], [2, 5], [2, 6], [3, 4], [3, 6], [3, 7],
             [4, 5], [4, 6], [4, 7], [4, 8], [4, 11], [5, 6], [5, 8], [5, 9], [5, 10], [6, 7], [6, 10], [6, 11],
             [7, 11], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [9, 10], [9, 13], [10, 11], [10, 13], [10, 14],
             [11, 12], [11, 14], [11, 15], [12, 13], [12, 14], [12, 15], [13, 14], [14, 15]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3), dtype=np.float64)
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3), dtype=np.float64)
        self.create_basis_mesh()


class Rhombohedral(BravaisLattice):
    def __init__(self, a, alpha, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, alpha=alpha, beta=alpha, gamma=alpha, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0],
                                     [0.0, 1.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [0.0, 1.0, 1.0],
                                     [1.0, 0.0, 1.0],
                                     [1.0, 1.0, 0.0],
                                     [1.0, 1.0, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class Kagome(BravaisLattice):
    def __init__(self, a, alpha, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=a, alpha=alpha, beta=alpha, gamma=alpha, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.5],
                                     [0.0, 0.0, 1.0],
                                     [0.0, 0.5, 0.0],
                                     [0.0, 0.5, 1.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 1.0, 0.5],
                                     [0.0, 1.0, 1.0],
                                     [0.5, 0.0, 0.0],
                                     [0.5, 0.0, 1.0],
                                     [0.5, 1.0, 0.0],
                                     [0.5, 1.0, 1.0],
                                     [1.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.5],
                                     [1.0, 0.0, 1.0],
                                     [1.0, 0.5, 0.0],
                                     [1.0, 0.5, 1.0],
                                     [1.0, 1.0, 0.0],
                                     [1.0, 1.0, 0.5],
                                     [1.0, 1.0, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 3], [0, 8], 
             [1, 2], [1, 3], [1, 8], 
             [2, 4], [2, 9], 
             [3, 5], [3, 8], 
             [4, 6], [4, 7], [4, 9],
             [5, 6], [5, 10],
             [6, 7], [6, 10],
             [7, 11],
             [8, 12],
             [9, 13], [9, 14],
             [10, 15], [10, 17],
             [11, 16], [11, 18], [11, 19],
             [12, 13], [12, 15],
             [13, 14], [13, 15], 
             [14, 16],
             [15, 17],
             [16, 18], [16, 19],
             [17, 18],
             [18, 19]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class PrimitiveTetragonal(BravaisLattice):
    def __init__(self, a, c, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=c, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class Triangulated(BravaisLattice):
    def __init__(self, a, b, c, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0],
                                     [0.0, 0.5, 0.0],
                                     [0.0, 1.0, 0.0],
                                     [0.0, 1.0, 1.0],
                                     [0.5, 0.0, 0.0],
                                     [0.5, 0.0, 1.0],
                                     [0.5, 0.5, 1.0],
                                     [0.5, 1.0, 0.0],
                                     [0.5, 1.0, 1.0],
                                     [1.0, 0.0, 0.0],
                                     [1.0, 0.0, 1.0],
                                     [1.0, 0.5, 0.0],
                                     [1.0, 1.0, 0.0],
                                     [1.0, 1.0, 1.0]])

        self.basis_elems = np.array(
            [[0, 2],
             [1, 2], [1, 4], [1, 5], [1, 7],
             [2, 3], [2, 4], [2, 5], [2, 7], [2, 8],
             [4, 7], [4, 8],
             [5, 7], [5, 8], [5, 11], [5, 12],
             [6, 7],
             [7, 8], [7, 9], [7, 11], [7, 12], [7, 14],
             [8, 12], [8, 14],
             [10, 12],
             [11, 12], [11, 14],
             [12, 13], [12, 14]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class BodyCenteredTetragonal(BravaisLattice):
    def __init__(self, a, c, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=a, c=c, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.5]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8], [6, 8], [7, 8]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class PrimitiveOrthorhombic(BravaisLattice):
    def __init__(self, a, b, c, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basis_elems = np.array([[0, 1],
                                    [0, 2],
                                    [0, 4],
                                    [1, 3],
                                    [1, 5],
                                    [2, 3],
                                    [2, 6],
                                    [3, 7],
                                    [4, 5],
                                    [4, 6],
                                    [5, 7],
                                    [6, 7]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class BodyCenteredOrthorhombic(BravaisLattice):
    def __init__(self, a, b, c, num_elems=1, inverted=False):
        BravaisLattice.__init__(self, a=a, b=b, c=c, num_elems=num_elems)
        if inverted:
            self.basis_nodes = np.array([[0.0, 0.0, 0.5],
                                        [0.0, 1.0, 0.5],
                                        [0.5, 0.5, 0.0],
                                        [0.5, 0.5, 1.0],
                                        [1.0, 0.0, 0.5],
                                        [1.0, 1.0, 0.5]])
            self.basis_elems = np.array(
                [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]])
        else:
            self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 1.0, 1.0],
                                        [0.5, 0.5, 0.5],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 0.0],
                                        [1.0, 1.0, 1.0]])
            self.basis_elems = np.array(
                [[0, 4], [1, 4], [2, 4], [3, 4],
                 [4, 5], [4, 6], [4, 7], [4, 8]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class BaseCenteredOrthorhombic(BravaisLattice):
    def __init__(self, a, b, c, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.0],
                                    [0.5, 0.5, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [2, 8], [3, 8], [6, 8], [1, 9], [4, 9], [5, 9], [7, 9]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class FaceCenteredOrthorhombic(BravaisLattice):
    def __init__(self, a, b, c, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.0, 0.5, 0.5],
                                    [0.5, 0.0, 0.5],
                                    [0.5, 0.5, 0.0],
                                    [1.0, 0.5, 0.5],
                                    [0.5, 1.0, 0.5],
                                    [0.5, 0.5, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [1, 8], [2, 8], [4, 8], [0, 9], [1, 9], [3, 9], [5, 9], [0, 10], [2, 10], [3, 10], [6, 10],
             [3, 11], [5, 11], [6, 11], [7, 11], [2, 12], [4, 12], [6, 12], [7, 12], [1, 13], [4, 13], [5, 13],
             [7, 13]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class PrimitiveMonoclinic(BravaisLattice):
    def __init__(self, a, b, c, beta, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, beta=beta, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class BaseCenteredMonoclinic(BravaisLattice):
    def __init__(self, a, b, c, beta, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, beta=beta, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0],
                                    [0.5, 0.5, 0.0],
                                    [0.5, 0.5, 1.0]])
        self.basis_elems = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7],
             [0, 8], [2, 8], [3, 8], [6, 8], [1, 9], [4, 9], [5, 9], [7, 9]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()


class Triclinic(BravaisLattice):
    def __init__(self, a, b, c, alpha, beta, gamma, num_elems=1):
        BravaisLattice.__init__(self, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma, num_elems=num_elems)
        self.basis_nodes = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, 1.0, 0.0],
                                    [1.0, 0.0, 0.0],
                                    [0.0, 1.0, 1.0],
                                    [1.0, 0.0, 1.0],
                                    [1.0, 1.0, 0.0],
                                    [1.0, 1.0, 1.0]])
        self.basis_elems = np.array([[0, 1],
                                    [0, 2],
                                    [0, 3],
                                    [1, 4],
                                    [1, 5],
                                    [2, 4],
                                    [2, 6],
                                    [3, 5],
                                    [3, 6],
                                    [4, 7],
                                    [5, 7],
                                    [6, 7]])
        self.nodes = np.empty((self.basis_elems.shape[0], num_elems + 1, 3))
        self.elems = np.empty((self.basis_elems.shape[0], num_elems, 2, 3))
        self.create_basis_mesh()
