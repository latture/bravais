import numpy as np
import os
from assign_major_axes import assign_major_axes
from bcs import get_corners, get_load_points, get_face_nodes, get_center_planes
from cpp_utils import *
from mesher import Job
from collections import namedtuple

__all__ = ['create_inp']


class EquationConstraint(namedtuple('EquationConstaint', ['nodeset', 'dof', 'coeff'])):
    def __str__(self):
        return "%s, %d, %f," % (self.nodeset, self.dof, self.coeff)


def get_instance(instName, partName):
    """
    Returns the data to be written to file to create an instance of the specified part.

    :param instName : The name of the instance to write to the input file.
    :param partName : The name of the part to instance.
    """
    return ("*INSTANCE", ", NAME=" + instName, ", PART=" + partName, "\n", "*END INSTANCE\n", "**\n")


def get_nset(nsetName, instName):
    """
    Returns the data (minus the nodal coordinates) to be written to file to create the specified node set.

    :param nsetName : The name of the nodal set to write to the input file.
    :param instName : The name of the instance associated with the nodes.
    """
    return ("*NSET", ", NSET=" + nsetName, ", INSTANCE=" + instName, ", unsorted\n")

def write_dict_to_nset(f, dictionary, instance_name, break_signifier='**\n'):
    """
    Writes each key value pair in the dictionary to a node set. The key is the node set's
    designation, and the value contains a list of node label to associate with the set.

    :param f: File to write to.
    :param dictionary: Dictionary of key, value pairs to write.
    :param instance_name: Name of the instance to associate the node set with.
    :param break_signifier: Any signifier to mark the end of the set. Default is a `'**\n'`.
    """
    for key, value in dictionary.iteritems():
            for entry in get_nset(key, instance_name):
                f.write(entry)
            for idx in value:
                f.write("%d\n" % idx)
            f.write(break_signifier)

def write_eq_constraint(f, eq_constraints, break_signifier='**\n'):
    """
    Writes the specified equation constrain to the file `f`.
    :param f: File to write to.
    :param eq_constraints: `Tuple`, dtype=`EquationConstraint`. Set of equation constraint data to apply.
    :param break_signifier: Any signifier to mark the end of the equation constraint.
                            Default is a `'**\n'`.
    """
    f.write("*EQUATION\n")
    f.write("%d\n" % len(eq_constraints))
    line = ""
    for ec in eq_constraints:
        line += "%s " % str(ec)
    line += "\n"
    f.write(line)
    f.write(break_signifier)

def create_inp(file_name, jobs, radii, load_type, strain, tile_pts='SC',
               youngs_modulus=200.0e9, shear_modulus=80.0e9, poisson_ratio=0.3,
               periodic_bcs=False):
    """
    Creates an Abaqus input file for the Bravais lattice specified in job with the name `file_name.inp`

    :param file_name         : `String`. The name of the input file.
    :param job               : `BravaisJob`. The mesh to analyze. Must have member variables nodes and elems which are
                                numpy arrays.
    :param radius            : `Float`. The radius of the cross-sectional area of the struts (assumed to be circular).
    :param load_type         : `String`. Specifies the type of loading the lattice will be subjected to. Can be either
                               `axial`, `shear`, or `bulk`.
    :param strain            : `Float`. Amount of strain to apply to the `job`.
    :param tile_pts          : `String`. lattice points to tile when selecting the min/max load points.
                                Options are 'SC', 'FCC', or 'FC_only'.
    :param youngs_modulus    : `Float`. Young's modulus of the parent material.
    :param shear_modulus     : `Float`. Shear modulus of the parent material.
    :param poisson_ratio     : `Float`. Poisson ratio of the parent material.
    :param periodic_bcs      : `Bool`, Default=`False`. If `True` periodic boundary conditions will be applied to the 
                                faces of the mesh.
    """
    assert len(jobs) == len(radii), "Length of input jobs and radii do not match."
    assert len(jobs) <= 2, "Too many jobs. Job merging currently only works with 2 elementary meshes."

    job = jobs[0]
    elsets = ()
    major_axes = []
    unique_major_axes = []
    if len(jobs) == 1:
        elsets = (range(job.elems.shape[0]),)
        major_axes.append(assign_major_axes(job))
        unique_major_axes.append(np.asarray(delete_duplicates_dbl(major_axes[0])))
    else:
        for j in jobs[1:]:
            elsets = job.merge(j, return_elem_indices=True)
        for set in elsets:
            _job_temp = Job(job.nodes, job.elems[np.array(set)])
            _major_axes_temp = assign_major_axes(_job_temp)
            _unique_major_axes_temp = np.asarray(delete_duplicates_dbl(_major_axes_temp))
            major_axes.append(_major_axes_temp)
            unique_major_axes.append(_unique_major_axes_temp)

    material_data = ["*MATERIAL, NAME=mat", "*ELASTIC", "%f, %f" % (youngs_modulus, poisson_ratio)]
    assembly_data_header = ["*ASSEMBLY, NAME=assembly"]
    assembly_data_footer = ["*END ASSEMBLY"]

    instance_name = "unit_cell-1"
    part_name = "part"
    nodes_name = "partNodes"
    elems_name = "partElems"
    fict_node_name = "fict"
    fict_instance_name = "fict-1"
    part_data_header = "*PART, NAME="
    part_data_footer = "*END PART"

    break_signifier = "********************************************************\n"

    f = open(os.getcwd() + "/" + file_name + ".inp", "w")
    f.write("*PREPRINT,MODEL=YES\n")

    # [write fictitious node part data
    if periodic_bcs:
        f.write(part_data_header + fict_node_name + "\n")
        f.write("*NODE, NSET=fictNode\n")
        for i in range(1, 4):
            f.write("%d, 1.0, 1.0, 1.0\n" % i)
        f.write(part_data_footer + "\n")
        f.write(break_signifier)
    # ]

    # [write mesh part data
    f.write(part_data_header + part_name + "\n")
    # nodes:
    f.write("*NODE, NSET=%s\n" % nodes_name)
    for i in xrange(job.nodes.shape[0]):
        f.write("%d, " % (i + 1))
        for j in xrange(job.nodes.shape[1]):
            f.write("%.12f" % job.nodes[i, j])
            if j != job.nodes.shape[1] - 1:
                f.write(", ")
        f.write("\n")

    # elems:
    for i, elset in enumerate(elsets):
        f.write("*ELEMENT, TYPE=B31, ELSET=%s\n" % (elems_name + "-" + str(i + 1)))
        for idx in elset:
            f.write("%d, " % (idx + 1))
            for j, node_num in enumerate(job.elems[idx]):
                f.write("%d" % (node_num + 1))
                if j != job.elems.shape[1] - 1:
                    f.write(", ")
            f.write("\n")

    # sections:
    for i in xrange(len(elsets)):
        uma_rows, uma_cols = unique_major_axes[i].shape
        for j in xrange(uma_rows):
            set_name = elems_name + "_Sect_Group-" + str(i + 1) + "-" + str(j + 1)
            f.write("*ELSET, ELSET=%s\n" % set_name)
            rows = np.asarray(test_rows(unique_major_axes[i][j], major_axes[i]))

            output_elems = ""
            for idx in rows:
                output_elems += "%d\n" % (elsets[i][idx] + 1)
            f.write(output_elems)

            norm_vec = ""
            for k in xrange(uma_cols):
                norm_vec += str(unique_major_axes[i][j, k])
                if k != uma_cols - 1:
                    norm_vec += ", "

            sect_data = ["*BEAM GENERAL SECTION, SECTION=CIRC, ELSET=" + set_name,
                          str(radii[i]), norm_vec, "%f, %f" % (youngs_modulus, shear_modulus)]

            # write the section data
            for line in sect_data:
                f.write("%s\n" % line)

    f.write(part_data_footer + "\n")
    f.write(break_signifier)
    # write part]

    # write material data
    for line in material_data:
        f.write("%s\n" % line)
    f.write(break_signifier)

    # create assembly
    for line in assembly_data_header:
        f.write("%s\n" % line)
    f.write(break_signifier)

    # instance part(s)
    for line in get_instance(instance_name, part_name):
        f.write(line)

    if periodic_bcs:
        for line in get_instance(fict_instance_name, fict_node_name):
            f.write(line)

    # get mesh faces:
    xmax_nodes, xmin_nodes = get_face_nodes(job.nodes, 0)
    ymax_nodes, ymin_nodes = get_face_nodes(job.nodes, 1)
    zmax_nodes, zmin_nodes = get_face_nodes(job.nodes, 2)

    # calculate length along x, y, and z directions
    xlen = np.linalg.norm(job.nodes[xmax_nodes[0]] - job.nodes[xmin_nodes[0]])
    ylen = np.linalg.norm(job.nodes[ymax_nodes[0]] - job.nodes[ymin_nodes[0]])
    zlen = np.linalg.norm(job.nodes[zmax_nodes[0]] - job.nodes[zmin_nodes[0]])

    minmax_nodes = {
        "xmin" : xmin_nodes + 1,
        "xmax" : xmax_nodes + 1,

        "ymin" : ymin_nodes + 1,
        "ymax" : ymax_nodes + 1,

        "zmin" : zmin_nodes + 1,
        "zmax" : zmax_nodes + 1,
    }

    if periodic_bcs:
        # only constrain nodes that are already part of pbc to avoid over constraint
        # xmin, xmax contain all nodes on the respective faces
        # ymin, ymax exclude nodes on x min or max faces
        # zmin, zmax exclude nodes on both x and y directions on min or max faces
        x_nodes_combined = np.concatenate([xmin_nodes, xmax_nodes])
        y_nodes_combined = np.concatenate([ymin_nodes, ymax_nodes])
        xy_nodes_combined = np.unique(np.concatenate([x_nodes_combined, y_nodes_combined]))

        ymax_nodes_excluded = np.setdiff1d(ymax_nodes, x_nodes_combined, assume_unique=True)
        ymin_nodes_excluded = np.setdiff1d(ymin_nodes, x_nodes_combined, assume_unique=True)

        zmax_nodes_excluded = np.setdiff1d(zmax_nodes, xy_nodes_combined, assume_unique=True)
        zmin_nodes_excluded = np.setdiff1d(zmin_nodes, xy_nodes_combined, assume_unique=True)

        # check that nodes on max and min faces are in corresponding order before applying bc
        x_expected = np.zeros((xmax_nodes.shape[0], 3))
        x_expected[:, 0] = xlen

        y_expected = np.zeros((ymax_nodes_excluded.shape[0], 3))
        y_expected[:, 1] = ylen

        z_expected = np.zeros((zmax_nodes_excluded.shape[0], 3))
        z_expected[:, 2] = zlen

        assert np.allclose(x_expected,
                           (job.nodes[xmax_nodes] - job.nodes[xmin_nodes]), rtol=1.e-4, atol=1.e-7), \
                           "xmax and xmin nodes are not sorted in corresponding order. Required for " \
                           "periodic boundary conditions in Abaqus."

        assert np.allclose(y_expected,
                           (job.nodes[ymax_nodes_excluded] - job.nodes[ymin_nodes_excluded]), rtol=1.e-4, atol=1.e-7), \
                           "ymax and ymin nodes are not sorted in corresponding order. Required for " \
                           "periodic boundary conditions in Abaqus."

        assert np.allclose(z_expected,
                           (job.nodes[zmax_nodes_excluded] - job.nodes[zmin_nodes_excluded]), rtol=1.e-4, atol=1.e-7), \
                           "zmax and zmin nodes are not sorted in corresponding order. Required for " \
                           "periodic boundary conditions in Abaqus."

        pbc_node_keys = ['xmax', 'xmin', 'ymax', 'ymin', 'zmax', 'zmin']
        pbc_node_values = [xmax_nodes, xmin_nodes,
                           ymax_nodes_excluded, ymin_nodes_excluded,
                           zmax_nodes_excluded, zmin_nodes_excluded]
        for i in range(len(pbc_node_keys)):
            # avoid writing empty node sets to dict
            if len(pbc_node_values[i]) > 0:
                minmax_nodes[pbc_node_keys[i]+"_pbc"] = pbc_node_values[i] + 1

    write_dict_to_nset(f, minmax_nodes, instance_name, break_signifier=break_signifier)

    # find the load points on each face
    loadpoints = {}

    if load_type.lower() == 'axial':
        loadpoint_keys = ['xmax', 'xmin']
    elif load_type.lower() == 'shear':
        loadpoint_keys = ['xmax', 'xmin', 'ymax', 'ymin']
    elif load_type.lower() == 'bulk':
        loadpoint_keys = ['xmax', 'xmin', 'ymax', 'ymin', 'zmax', 'zmin']
    else:
        raise Exception("Load type: %s is not a valid load type. Please choose from `axial`, `shear`, or `bulk`."
                        % load_type)

    for key in loadpoint_keys:
        loadpoints[key + "_loadpoints"] = minmax_nodes[key]

    write_dict_to_nset(f, loadpoints, instance_name, break_signifier=break_signifier)

    # find center symmetry planes
    centerplanes = {}
    centerplane_keys = ['yz_center_plane', 'xy_center_plane']
    centerplane_values = get_center_planes(job.nodes)

    for i in range(len(centerplane_keys)):
        centerplanes[centerplane_keys[i]] = centerplane_values[i] + 1

    write_dict_to_nset(f, centerplanes, instance_name, break_signifier=break_signifier)

    if periodic_bcs:
        fict_dict = {
            "f1" : [1],
            "f2" : [2],
            "f3" : [3]
        }
        write_dict_to_nset(f, fict_dict, fict_instance_name, break_signifier=break_signifier)

        # apply equation constraint to tie x, y, and z faces to fictitious nodes
        directions = ('x', 'y', 'z')
        counter = 1
        for direct in directions:
            key1 = direct + 'max_pbc'
            key2 = direct + 'min_pbc'
            if key1 in minmax_nodes.keys() and key2 in minmax_nodes.keys():
                for i in range(1, 4):
                    eq_constraints = (
                        EquationConstraint(nodeset=key1, dof=i, coeff=1.0),
                        EquationConstraint(nodeset=key2, dof=i, coeff=-1.0),
                        EquationConstraint(nodeset='f'+str(counter), dof=i, coeff=-1.0)
                    )
                    write_eq_constraint(f, eq_constraints)
                f.write(break_signifier)
            counter += 1

        # enforce epsilon_ij == epsilon_ji
        lengths = [xlen, ylen, zlen]
        for i in range(0, 3):
            idx1 = i + 1
            idx2 = (i + 1) % 3 + 1
            eq_constraints = (
                EquationConstraint(nodeset='f'+str(idx1), dof=idx2, coeff=lengths[idx2-1]),
                EquationConstraint(nodeset='f'+str(idx2), dof=idx1, coeff=-lengths[idx1-1])
            )
            write_eq_constraint(f, eq_constraints)
        f.write(break_signifier)

    # end assembly
    for line in assembly_data_footer:
        f.write("%s\n" % line)
    f.write(break_signifier)

    # create step
    f.write("*STEP, PERTURBATION, NLGEOM=YES\n")
    f.write("Apply load to faces\n")
    f.write("*STATIC, STABILIZE\n")
    f.write(break_signifier)

    f.write("*BOUNDARY\n")

    # Apply symmetric boundary conditions at center planes
    if load_type.lower() != 'shear':
        f.write("%s, 1\n" % centerplane_keys[0])
        f.write("%s, 3\n" % centerplane_keys[1])

    # apply displacement
    for key in loadpoints.keys():
        if key[0] == 'x':
            if load_type.lower() != 'shear':
                direction = 1
            else:
                direction = 2
            displacement = strain * xlen / 2.0
        elif key[0] == 'y':
            if load_type.lower() != 'shear':
                direction = 2
            else:
                direction = 1
            displacement = strain * ylen / 2.0
        elif key[0] == 'z':
            direction = 3
            displacement = strain * zlen / 2.0
        else:
            raise Exception("Direction %s is not valid. Key should specify either x, y, or z direction." % key[0])

        if key[1:4] == 'min':
            sign = 1
        elif key[1:4] == 'max':
            sign = -1
        else:
            raise Exception("Side %s is not valid. Key should specify min or max face after direction." % key[1:4])

        f.write("%s, %i, %i, %f\n" % (key, direction, direction, sign * displacement))

    f.write(break_signifier)

    # request outputs
    f.write("*OUTPUT, FIELD\n")
    f.write("*NODE OUTPUT\n")
    f.write("U, RF\n")
    f.write("*END STEP")

    # close file
    f.close()
