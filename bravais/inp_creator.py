import numpy as np
import os
from assign_major_axes import *
from bcs import get_face_nodes, get_center_planes
from cpp_utils import *
from mesher import Job
from collections import namedtuple

__all__ = ['create_inp', 'InpDescription']


class InpDescription(namedtuple('InpDescription', ['job_type', 'dimensions', 'num_elems', 'load_type', 'misc'])):
    """
    Holds a description of the inp file being created. Used to construct the file name as well as the
    naming convention for node and element sets.
    """

    def __new__(cls, job_type, dimensions, num_elems, load_type, misc=''):
        """
        :param job_type   : `String`. Label for current job, e.g. `'FCC-SC'`.
        :param dimensions : `array_like`, dtype=`Int`. Dimensions of the job along x, y, and z directions in terms of the
                             number of unit cells, e.g. `[1, 1, 1]`.
        :param num_elems  : `Int`. The number of elements along each strut of the job.
        :param load_type  : `String`. Type of loading the job will be subjected to, e.g. `'axial'`
        :param misc       : `String`. Miscellaneous identifiers to append to the end of the file name.
        """
        return super(InpDescription, cls).__new__(cls, job_type, dimensions, num_elems, load_type, misc)


def get_instance(instName, partName):
    """
    Returns the data to be written to file to create an instance of the specified part.

    :param instName : The name of the instance to write to the input file.
    :param partName : The name of the part to instance.
    """
    return ('*INSTANCE', ', NAME=' + instName, ', PART=' + partName, '\n', '*END INSTANCE\n', '**\n')


def get_nset(nsetName, instName):
    """
    Returns the data (minus the nodal coordinates) to be written to file to create the specified node set.

    :param nsetName : The name of the nodal set to write to the input file.
    :param instName : The name of the instance associated with the nodes.
    """
    return ('*NSET', ', NSET=' + nsetName, ', INSTANCE=' + instName, ', unsorted\n')

def write_keys_to_nset(f, save_path, keys, instance_name, break_signifier='\n'):
    """
    Writes node sets to the inp file `f` with include directives to `save_path/keys.csv` and node set names
    designated by keys. All node sets pertain to the instance defined by the variable `instance_name`.
    :param f: file to the node set data to.
    :param save_path: `String`. Path to the node csv data.
    :param keys: `array_like`, dtype=`String`. List of keys specifying node set names.
    :param instance_name: `String`. Name of the instance associated with the node sets.
    :param break_signifier: `String`, default=`'\n'`. A break signifier to write between node sets to improve readability.
    """
    for k in keys:
            for entry in get_nset(k, instance_name):
                f.write(entry)
            f.write('*INCLUDE, INPUT=%s\n' % (save_path+k+'.csv'))
            f.write(break_signifier)

def write_dict_to_csv(save_path, dictionary):
    """
    Writes the key/value pairs in dictionary to csv files at the location specified by `save_path`.
    :param save_path: `String`. Path to folder where csv files should be saved.
    :param dictionary: `Dict`. List of key value pairs where keys are strings specifying the name of the node set and
    values are arrays of values that are in the node set.
    """
    for key, value in dictionary.iteritems():
        np.savetxt(save_path+key+'.csv', value, fmt='%d')

def create_inp(inp_description, jobs, radii, load_type, strain, csv_dir='csv_data',
               youngs_modulus=200.0e9, shear_modulus=80.0e9, poisson_ratio=0.3):
    """
    Creates an Abaqus input file for the Bravais lattices specified in jobs.

    :param inp_description   : `InpDescription`. Description of the input file.
    :param jobs              : `Tuple`, dtype=`BravaisJob`. The mesh to analyze.
                                Must have member variables nodes and elems which are numpy arrays.
    :param radii             : `Tuple`, dtype=`Float`. The radius of the cross-sectional area of the struts
                               (assumed to be circular).
    :param load_type         : `String`. Specifies the type of loading the lattice will be subjected to. Can be either
                               `axial`, `shear`, or `bulk`.
    :param strain            : `Float`. Amount of strain to apply to the `job`.
    :param youngs_modulus    : `Float`. Young's modulus of the parent material.
    :param shear_modulus     : `Float`. Shear modulus of the parent material.
    :param poisson_ratio     : `Float`. Poisson ratio of the parent material.
    """
    assert len(jobs) == len(radii), 'Length of input jobs and radii do not match.'
    assert len(jobs) <= 2, 'Too many jobs. Job merging currently only works with 2 elementary meshes.'

    material_data = ['*MATERIAL, NAME=mat', '*ELASTIC', '%f, %f' % (youngs_modulus, poisson_ratio)]
    assembly_data_header = ['*ASSEMBLY, NAME=assembly']
    assembly_data_footer = ['*END ASSEMBLY']

    instance_name = 'unit_cell-1'
    part_name = 'part'
    nodes_name = 'partNodes'
    elems_name = 'partElems'
    part_data_header = '*PART, NAME='
    part_data_footer = '*END PART'

    break_signifier = '********************************************************\n'

    dims_str = ''
    for i in range(len(inp_description.dimensions)):
        dims_str += '%d' % inp_description.dimensions[i]
        if i < len(inp_description.dimensions) - 1:
            dims_str += 'x'

    if csv_dir[-1] != '/':
        csv_dir += '/'

    save_path = csv_dir + inp_description.job_type + '/' + dims_str + '/'

    job_data_available = False
    if os.path.exists(save_path):
        job_data_available = os.path.isfile(save_path+'.data_available')
    else:
        os.makedirs(save_path)

    file_name = '%s_%s_%s-elems_%s_%s.inp' % \
                (inp_description.job_type,
                 dims_str,
                 inp_description.num_elems,
                 inp_description.load_type,
                 inp_description.misc)

    unique_major_axes_filename = save_path + inp_description.job_type + '_' + dims_str + '_' + 'unique_major_axes'

    if not job_data_available:
        job = jobs[0]
        major_axes = []
        unique_major_axes = []

        if len(jobs) == 1:
            elsets = (range(job.elems.shape[0]),)
            major_axes.append(np.asarray(assign_major_axes(job.nodes, job.elems)))
            unique_major_axes.append(np.asarray(delete_duplicates_dbl(major_axes[0])))
        else:
            for j in jobs[1:]:
                elsets = job.merge(j, return_elem_indices=True)
            for eset in elsets:
                _job_temp = Job(job.nodes, job.elems[np.asarray(eset)])
                _major_axes_temp = np.asarray(assign_major_axes(_job_temp.nodes, _job_temp.elems))
                _unique_major_axes_temp = np.asarray(delete_duplicates_dbl(_major_axes_temp))
                major_axes.append(_major_axes_temp)
                unique_major_axes.append(_unique_major_axes_temp)
        for i in range(len(jobs)):
            np.savetxt(unique_major_axes_filename + '-' + str(i) + '.csv', unique_major_axes[i])
    else:
        unique_major_axes = []
        for i in range(len(jobs)):
            unique_major_axes.append(np.loadtxt(unique_major_axes_filename + '-' + str(i) + '.csv'))
        np.asarray(unique_major_axes)

    f = open(os.getcwd() + '/' + file_name, 'w')

    f.write('*PREPRINT,MODEL=YES\n')

    # [write part
    f.write(part_data_header + part_name + '\n')
    # nodes:
    nodes_filename = save_path + inp_description.job_type + '_' + dims_str + "-nodes.csv"
    f.write('*NODE, NSET=%s\n' % nodes_name)
    f.write('*INCLUDE, INPUT=%s\n' % nodes_filename)
    if not os.path.isfile(nodes_filename):
        node_str = ''
        for i in xrange(job.nodes.shape[0]):
            node_str += '%d, ' % (i + 1)
            for j in xrange(job.nodes.shape[1]):
                node_str += '%.12f' % job.nodes[i, j]
                if j != job.nodes.shape[1] - 1:
                    node_str += ', '
                else:
                    node_str += '\n'
        with open(nodes_filename, "w") as csv_file:
            csv_file.write(node_str)

    # elems:
    for i in range(len(jobs)):
        elset_filename = save_path + inp_description.job_type + '_' + dims_str + '-' + elems_name + '-' + str(i + 1) + '.csv'
        f.write('*ELEMENT, TYPE=B31, ELSET=%s\n' % (elems_name + '-' + str(i + 1)))
        f.write('*INCLUDE, INPUT=%s\n' % elset_filename)
        if not os.path.isfile(elset_filename):
            elem_str = ''
            for idx in elsets[i]:
                elem_str += '%d, ' % (idx + 1)
                for j, node_num in enumerate(job.elems[idx]):
                    elem_str += '%d' % (node_num + 1)
                    if j != job.elems.shape[1] - 1:
                        elem_str += ', '
                elem_str += '\n'
            with open(elset_filename, 'w') as csv_file:
                csv_file.write(elem_str)

    # sections:
    for i in xrange(len(jobs)):
        uma_rows, uma_cols = unique_major_axes[i].shape
        for j in xrange(uma_rows):
            set_name = elems_name + '_Sect_Group-' + str(i + 1) + '-' + str(j + 1)
            set_filename = save_path + set_name + '.csv'
            f.write('*ELSET, ELSET=%s\n' % set_name)
            f.write('*INCLUDE, INPUT=%s\n' % set_filename)
            if not os.path.isfile(set_filename):
                rows = np.asarray(test_rows(unique_major_axes[i][j], major_axes[i]))

                output_elems = ''
                for idx in rows:
                    output_elems += '%d\n' % (elsets[i][idx] + 1)
                with open(set_filename, 'w') as csv_file:
                    csv_file.write(output_elems)

            norm_vec = ''
            for k in xrange(uma_cols):
                norm_vec += str(unique_major_axes[i][j, k])
                if k != uma_cols - 1:
                    norm_vec += ', '

            sect_data = ['*BEAM GENERAL SECTION, SECTION=CIRC, ELSET=' + set_name,
                         str(radii[i]), norm_vec, '%f, %f' % (youngs_modulus, shear_modulus)]

            # write the section data
            for line in sect_data:
                f.write('%s\n' % line)

    f.write(part_data_footer + '\n')
    f.write(break_signifier)
    # write part]

    # write material data
    for line in material_data:
        f.write('%s\n' % line)
    f.write(break_signifier)

    # create assembly
    for line in assembly_data_header:
        f.write('%s\n' % line)
    f.write(break_signifier)

    # instance part
    for line in get_instance(instance_name, part_name):
        f.write(line)

    centerplane_keys = ['yz_center_plane', 'xy_center_plane']
    face_node_keys = ['xmax', 'xmin', 'ymax', 'ymin', 'zmax', 'zmin']

    if load_type.lower() == 'axial':
        loadpoint_keys = ['xmax', 'xmin']
    elif load_type.lower() == 'shear':
        loadpoint_keys = ['xmax', 'xmin', 'ymax', 'ymin']
    elif load_type.lower() == 'bulk':
        loadpoint_keys = ['xmax', 'xmin', 'ymax', 'ymin', 'zmax', 'zmin']
    else:
        raise Exception('Load type: %s is not a valid load type. Please choose from `axial`, `shear`, or `bulk`.'
                        % load_type)
    write_keys_to_nset(f, save_path, face_node_keys, instance_name, break_signifier=break_signifier)
    write_keys_to_nset(f, save_path, centerplane_keys, instance_name, break_signifier=break_signifier)

    if not job_data_available:
        # get mesh faces:
        xmax_nodes, xmin_nodes = get_face_nodes(job.nodes, 0)
        ymax_nodes, ymin_nodes = get_face_nodes(job.nodes, 1)
        zmax_nodes, zmin_nodes = get_face_nodes(job.nodes, 2)

        # calculate length along x, y, and z directions
        xlen = np.linalg.norm(job.nodes[xmax_nodes[0]] - job.nodes[xmin_nodes[0]])
        ylen = np.linalg.norm(job.nodes[ymax_nodes[0]] - job.nodes[ymin_nodes[0]])
        zlen = np.linalg.norm(job.nodes[zmax_nodes[0]] - job.nodes[zmin_nodes[0]])
        lengths = np.array([xlen, ylen, zlen])
        np.savetxt(save_path+'lengths.csv', lengths)

        minmax_nodes = {
            'xmin' : xmin_nodes + 1,
            'xmax' : xmax_nodes + 1,

            'ymin' : ymin_nodes + 1,
            'ymax' : ymax_nodes + 1,

            'zmin' : zmin_nodes + 1,
            'zmax' : zmax_nodes + 1,
        }

        write_dict_to_csv(save_path, minmax_nodes)

        # find center symmetry planes
        centerplanes = {}
        centerplane_values = get_center_planes(job.nodes)

        for i in range(len(centerplane_keys)):
            centerplanes[centerplane_keys[i]] = centerplane_values[i] + 1
        write_dict_to_csv(save_path, centerplanes)

    else:
        lengths = np.loadtxt(save_path+'lengths.csv')

    # end assembly
    for line in assembly_data_footer:
        f.write('%s\n' % line)
    f.write(break_signifier)

    # create step
    f.write('*STEP, PERTURBATION, NLGEOM=YES\n')
    f.write('Apply load to faces\n')
    f.write('*STATIC, STABILIZE\n')
    f.write(break_signifier)

    f.write('*BOUNDARY\n')

    # Apply symmetric boundary conditions at center planes
    if load_type.lower() != 'shear':
        f.write('%s, 1\n' % centerplane_keys[0])
        f.write('%s, 3\n' % centerplane_keys[1])

    # apply displacement
    for key in loadpoint_keys:
        if key[0] == 'x':
            if load_type.lower() != 'shear':
                direction = 1
            else:
                direction = 2
            displacement = strain * lengths[0] / 2.0
        elif key[0] == 'y':
            if load_type.lower() != 'shear':
                direction = 2
            else:
                direction = 1
            displacement = strain * lengths[1] / 2.0
        elif key[0] == 'z':
            direction = 3
            displacement = strain * lengths[2] / 2.0
        else:
            raise Exception('Direction %s is not valid. Key should specify either x, y, or z direction.' % key[0])

        if key[1:4] == 'min':
            sign = 1
        elif key[1:4] == 'max':
            sign = -1
        else:
            raise Exception('Side %s is not valid. Key should specify min or max face after direction.' % key[1:4])

        f.write('%s, %i, %i, %f\n' % (key, direction, direction, sign * displacement))

    f.write(break_signifier)

    # request outputs
    f.write('*OUTPUT, FIELD\n')
    f.write('*NODE OUTPUT\n')
    f.write('U, RF\n')
    f.write('*ELEMENT OUTPUT\n')
    f.write('SE, S, SF\n')
    f.write('*END STEP\n')

    # close file
    f.close()

    open(save_path+'.data_available', 'a').close()
