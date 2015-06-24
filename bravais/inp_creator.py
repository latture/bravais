import numpy as np
import os
from assign_major_axes import assign_major_axes
from bcs import get_corners, get_load_points, get_face_nodes, get_center_planes
from cpp_utils import *
from mesher import Job

__all__ = ['create_inp']


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

def write_dict_to_nset(f, dictionary, instance_name, break_signifier='\n'):
    for key, value in dictionary.iteritems():
            for entry in get_nset(key, instance_name):
                f.write(entry)
            for idx in value:
                f.write("%d\n" % idx)
            f.write(break_signifier)

def create_inp(file_name, jobs, radii, load_type, strain, tile_pts='SC',
                 youngs_modulus=200.0e9, shear_modulus=80.0e9, poisson_ratio=0.3):
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
        unique_major_axes.append(np.asarray(delete_duplicates_dbl(major_axes)))
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
    part_data_header = "*PART, NAME="
    part_data_footer = "*END PART"

    break_signifier = "********************************************************\n"

    f = open(os.getcwd() + "/" + file_name + ".inp", "w")
    f.write("*PREPRINT,MODEL=YES\n")

    # [write part
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

    # instance part
    for line in get_instance(instance_name, part_name):
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
