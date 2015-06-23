__author__ = 'ryan'

__all__ = ["sort_rows", "calc_rel_density"]

import numpy as np


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
