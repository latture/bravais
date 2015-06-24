__author__ = 'ryan'

__all__ = ["sort_rows", "calc_rel_density", "print_nodes"]

import numpy as np
from cpp_utils import replace_with_idx

def sort_rows(a):
    """
    Returns input_array with the rows sorted without reordering the entries.

    :param input_array : 2D array (assumed to have 3 columns).
    :return: 2D array sorted by rows.
    """

    return a[np.lexsort((a[:, 2], a[:, 1], a[:, 0]))]


def calc_rel_density(total_volume, jobs, areas):
    """
    Calculates the relative density of one or many jobs.
    :param total_volume: `Float`. The total volume of all `BravaisJob`'s.
    :param jobs: `Tuple` dtype=`BravaisJob`. Collection of `BravaisJob`'s contained within the `total_volume`.
    :param areas: `Tuple` dtype=`Float`. Collection of cross-sectional areas. The `ith` area corresponds to the struts
                   of the `ith` job.
    :return: Relative density. `Float`.
    """
    assert len(jobs) == len(areas), "jobs and areas are not the same length."
    vol = 0.0
    for i in xrange(len(jobs)):
        vol += jobs[i].calc_volume(areas[i])

    return vol / total_volume


def print_nodes(job1, job2, save_file=False, filename="nodes.csv"):
    """
    Prints the indices where the nodes from `job2` occur in `job1`.
    :param job1 : `Job`. The Job in which to search for the nodes.
    :param job2 : `Job`. The Job that contains the nodes to search for.
    :param save_file : `Bool`. If `True` a file designated by `filename` is saved which contains the indices where
                       the nodes of `job2` occur in `job1.nodes`.
    :param filename : `String`. Name of file to save if `save_file` is `True`.
    :return: Numpy array, dtype=`int`. Indices.
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
