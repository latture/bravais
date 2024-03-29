__author__ = 'ryan'

__all__ = ["sort_rows", "calc_rel_density", "print_nodes", "calc_radius", "calc_radii",
           "add_tie_line", "calc_axial_strain", "calc_direction_vectors", "sort_struts"]

import numpy as np
from cpp_utils import replace_with_idx
from math import sqrt


def sort_rows(a):
    """
    Returns input_array with the rows sorted without reordering the entries.

    :param a : 2D array (assumed to have 3 columns).
    :return  : 2D array sorted by rows.
    """

    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


def calc_rel_density(total_volume, jobs, areas):
    """
    Calculates the relative density of one or many jobs.
    :param total_volume : `Float`. The total volume of all `BravaisJob`'s.
    :param jobs         : `Tuple` dtype=`BravaisJob`. Collection of 
                          `BravaisJob`'s contained within the `total_volume`.
    :param areas        : `Tuple` dtype=`Float`. Collection of cross-sectional areas. 
                           The `ith` area corresponds to the struts of the `ith` job.
    :return             :  Relative density. `Float`.
    """
    assert len(jobs) == len(areas), "jobs and areas are not the same length."
    vol = 0.0
    for i in xrange(len(jobs)):
        vol += jobs[i].calc_volume(areas[i])
    return vol / total_volume


def print_nodes(job1, job2, save_file=False, filename="nodes.csv"):
    """
    Prints the indices where the nodes from `job2` occur in `job1`.
    :param job1      : `Job`. The Job in which to search for the nodes.
    :param job2      : `Job`. The Job that contains the nodes to search for.
    :param save_file : `Bool`. If `True` a file designated by `filename` is saved which contains the indices where
                        the nodes of `job2` occur in `job1.nodes`.
    :param filename  : `String`. Name of file to save if `save_file` is `True`.
    :return          :  Numpy array, dtype=`int`. Indices.
    """
    nodes = np.sort(np.asarray(replace_with_idx(job1.nodes, np.array([job2.nodes]))).ravel())
    if save_file:
        np.savetxt(filename, nodes, fmt='%d', delimiter=',', newline=',')
    n_str = "["
    for i, n in enumerate(nodes):
        n_str += "%s" % str(n)
        if i != nodes.shape[0] - 1:
            n_str += ", "
    n_str += "]"
    print n_str
    return nodes


def calc_radius(job, num_elems, aspect_ratio):
    """
    Calculates the strut radius for the elements in `job` for a given aspect ratio
    and number of elements along each strut. Assumes all struts in the job are the
    same length.
    :param job          : Contains `nodes` and `elems` member variables 
                          corresponding to the node and element lists for the current mesh.
    :param num_elems    : `Int`. The number of elements along each strut.
    :param aspect_ratio : `Float`. Aspect ratio given to each strut.
    """
    nn1, nn2 = job.elems[0]
    n1 = job.nodes[nn1]
    n2 = job.nodes[nn2]

    # this assumes that all elements have the same length
    strut_length = num_elems * np.linalg.norm(n2 - n1)

    return strut_length / (2.0 * aspect_ratio)


def calc_radii(jobs, fractions, relative_density, total_volume):
    """
    Calculates the radii for the struts for in `jobs` such that the fraction 
    of total material occupied in each job is equal to the specified fraction.
    :param jobs             : `array_like`, dtype=`BravaisJob`. List of jobs.
    :param fractions        : `array_like`, dtype=`Float`. Volume fraction of material to 
                               allocate to the corresponding entry in job, i.e. the ith job
                               in `jobs` with have the fraction of the total material specified
                               by the ith entry in fractions. Length of `fractions` must equal the
                               length of `jobs`.
    :param relative_density : `Float`, `0.0<=relative_density<=1.0`. Relative density
                               for the combined material of `jobs`.
    :param total_volume     : `Float`. Total volume, including free space occupied by the jobs.
                               This is not the total material volume.
    """
    assert len(jobs) == len(fractions), "Length of jobs and fractions are not equal. Each job must have 1 volume fraction."
    assert abs(np.sum(fractions) - 1.0) <= 1.e-6, "Sum of fractions must add up to 1.0."

    material_volume = relative_density * total_volume
    radii = []

    for j, f in zip(jobs, fractions):
        vol = f * material_volume
        radii.append(sqrt(vol / (j.total_length * np.pi)))

    return tuple(radii)


def add_tie_line(x, y):
    """
    Appends a row to `y` which is the linear tie line between the first and last data points for `x` and `y`
    evaluated at each x point.
    :param x : `array_like`. 1D array of x points.
    :param y : `array_like`. 1D array of y points.
    :return  : 2D numpy array where the first row is the original y data and the second row is the data 
               from evaluating the linear fit at each x value.
    """
    lin_fit = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
    fit_fcn = np.poly1d(lin_fit)
    y_tie_line = fit_fcn(x)
    return np.vstack([y, y_tie_line])


def calc_axial_strain(job, displacements):
    """
    Returns the axial strain in each strut in the `job`. The `ith` entry in
    the returned array corresponds to the axial strain in the `ith` element
    of the `job`.
    :param job           : `Job`. Object that contains the element and node list.
    :param displacements : `array_like`. Nodal displacements to apply to the job.
    :return              :  Numpy array, dtype=`float`. Elemental axial strains.
    """
    num_elems = job.elems.shape[0]
    axial_strain = np.empty(num_elems)

    for i in xrange(num_elems):
        # pull out node numbers from element
        nn1 = job.elems[i, 0]
        nn2 = job.elems[i, 1]

        # get the nodes referenced by the element
        n1 = job.nodes[nn1]
        n2 = job.nodes[nn2]

        # calculate the vector between the nodes
        dn = n2 - n1

        # calculate the original length of the element
        length_orig = np.sqrt(dn.dot(dn))

        # move nodes by appropriate displacement
        n1_def = n1 + displacements[nn1, 0:3]
        n2_def = n2 + displacements[nn2, 0:3]

        # recalculate distance between nodes
        dn_def = n2_def - n1_def

        # calculate deformed length
        length_def = np.sqrt(dn_def.dot(dn_def))

        axial_strain[i] = (length_def - length_orig) / length_orig

    return axial_strain


def calc_direction_vectors(job):
    """
    Calculates and returns the vectors between the beginning and ending nodes of each element in `job`.
    :param job: `Job`. Contains the node and element list stored in `nodes` and `elems` member variables, respectively.
    :return: Numpy array, dtype=`Float`. Direction vectors.
    """
    num_elems = job.elems.shape[0]
    dir_vecs = np.empty((num_elems, job.nodes.shape[1]))

    for i in xrange(num_elems):
        # pull out node numbers from element
        nn1 = job.elems[i, 0]
        nn2 = job.elems[i, 1]

        # get the nodes referenced by the element
        n1 = job.nodes[nn1]
        n2 = job.nodes[nn2]

        # calculate the vector between the nodes
        dir_vecs[i] = n2 - n1

    return dir_vecs


def sort_struts(jobs):
    """
    Sorts strut populations for the job(s) specified by the angle between the element and the global x-directin.
    :param jobs          : `array_like`. List of constituent job(s) for which `axial_strains` is defined.
                            If Compound mesh was not formed from merging 2 `BravaisJob` instances,
                            `jobs` will have a length equal to 1.
    :return: Sorted struts in the form `[[pop1_j1, pop2_j1,...], [pop1_j2, pop2_j2,...], ... [pop1_jN, pop2_jN,...]]`
             where `pop1_j1` is the first population of struts for the first job and each population is a
             Numpy array with dtype=`int`.
    """
    strut_populations = []
    for j in jobs:
        dir_vecs = calc_direction_vectors(j)

        # calculate the absolute value of the dot product of
        # the direction vectors with the global x-direction
        dot_prods = np.round(np.abs(dir_vecs[:, 0]), decimals=5)
        unique_dot_prods = np.unique(dot_prods)

        # define strut populations based on groups of struts that
        # have the same value from dot_prods
        job_population = []
        for unique_dot_prod in unique_dot_prods:
            curr_population = np.where(dot_prods == unique_dot_prod)[0]
            job_population.append(curr_population)
        strut_populations.append(job_population)

    return strut_populations
