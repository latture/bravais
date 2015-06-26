__author__ = 'ryan'

__all__ = ["categorize_files", "NodalData", "process_data"]

import numpy as np
from bcs import get_face_nodes
import matplotlib.pyplot as plt
import itertools


COLORS = {
        "red"    : (0.7176, 0.1098, 0.1098),
        "green"  : (0.65 * 0.298, 0.65 * 0.6863, 0.65 * 0.3137),
        "blue"   : (0.9 * 0.0824, 0.9 * 0.3961, 0.9 * 0.7529),
        "orange" : (0.85 * 1.0, 0.85 * 0.5961, 0.0),
        "purple" : (0.49412, 0.3412, 0.7608),
        "grey"   : (0.45, 0.45, 0.45),
        "cyan"   : (0.0, 0.7373, 0.8314),
        "teal"   : (0.0, 0.5882, 0.5333),
        "lime"   : (0.8039, 0.8627, 0.2235),
        "brown"  : (0.4745, 0.3333, 0.2824)
        }


class NodalData(object):
    """
    Holds data for nodal degrees of freedom for axial, shear, and bulk loading.
    """
    def __init__(self, files, path_to_files):
        self.axial = None
        self.shear = None
        self.bulk = None

        for f in files:
            if 'axial' in f:
                self.axial = np.loadtxt(path_to_files + f)
            elif 'shear' in f:
                self.shear = np.loadtxt(path_to_files + f)
            elif 'bulk' in f:
                self.bulk = np.loadtxt(path_to_files + f)


def plot_data(x, y, labels, xlabel=None, ylabel=None, xlim=None, ylim=None, xticks=None, yticks=None,
              add_legend=False, savefig=False, showfig=True, filename="test.png"):
    """
    Plots the rows of `y` with respect to the vector `x`.
    :param x: Array. x-axis values.
    :param y: Matrix. Each row of `y` will constitute a line plotted with respect to `x`. The number of entries in x
                      must match the number of columns of `y`.
    :param labels: Array, dtype=`String`. Labels for each line. every row in `y` must have a corresponding label.
    :param xlabel: `String`. Label for x-axis.
    :param ylabel: `String`. Label for y-axis.
    :param xlim: Array, `len(xlim)==2`. Upper and lower limits for the x-axis.
    :param ylim: Array, `len(ylim)==2`. Upper and lower limits for the y-axis.
    :param xticks: Array. List of ticks to use on the x-axis. Should be within the upper and lower bounds of the x-axis.
    :param yticks: Array. List of ticks to use on the y-axis. Should be within the upper and lower bounds of the y-axis.
    :param add_legend: `Bool`, default=`False`. If `True` a legend will be added to the plot.
    :param savefig: `Bool`, default=`False`. Whether to save the figure.
    :param showfig: `Bool`, default=`True`. Whether to show the figure.
    :param filename: `String`, default=`test.png`. Name of file to save the figure to if `savefig=True`.
    """
    assert y.shape[0] == len(labels), "Length of labels does not match the number of rows in y."
    # define the colors and markers for the plot
    color = itertools.cycle((COLORS["blue"],
                             COLORS["green"],
                             COLORS["red"],
                             COLORS["orange"],
                             COLORS["purple"],
                             COLORS["grey"],
                             COLORS["cyan"],
                             COLORS["teal"],
                             COLORS["lime"],
                             COLORS["brown"]))
    marker = itertools.cycle((u's', u'>', u'o', u'D', u'p',u'H', u'^', u'v', u'd'))

    fig = plt.figure(figsize=(8, 6), dpi=150)
    axis = fig.add_subplot(111)

    for i in xrange(y.shape[0]):
        axis.plot(x, y[i, :], label=labels[i], linewidth=2, marker=marker.next(), color=color.next())

    plt.rcParams.update({'font.size': 22})

    # update plot labels and format based on user input 
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    if xlim is not None:
        axis.set_xlim(xlim)
    if ylim is not None:
        axis.set_ylim(ylim)
    if xticks is not None:
        axis.set_xticks(xticks)
    if yticks is not None:
        axis.set_yticks(yticks)
    if add_legend:
        plt.legend(prop={'size': 12}, loc="upper right")
    if savefig:
        plt.savefig(filename, dpi=150, transparent=True, bbox_inches='tight')
    if showfig:
        plt.show()


def categorize_files(filenames, keys):
    """
    Splits the file names up based on whether the file name contains a keyword.
    :param filenames: `List`. List of file names to categorize.
    :param keys: `Tuple`. Keywords to search the file names for.
    :return: `Tuple`. categorized files. The `ith` tuple in the return will contain the matches to the `ith` key.
    """
    data = []
    for k in keys:
        matches = []
        for f in filenames:
            if k in f:
                matches.append(f)
        data.append(matches)
    return tuple(data)


def calc_total_value(indices, values):
    """
    Totals the rows in in `values` referenced by the variable `indices`.
    :param indices : `Array`, dtype=`int`. Array containing the indices of the rows to sum.
    :param values : `Array`, dtype=`float`. Nodal values. If data is from 3D analysis,
                           Each row contains the x, y, and z components, respectively.
    :return: Total values summed by column over the specified rows. `Array`, dtype=`float`.
    """
    # initialize variables to hold displacement in x, y, and z directions
    cols = len(values[0])
    total_vals = np.zeros(cols)

    # loop through indices and sum displacement in each direction
    for i in indices:
        total_vals += values[i]

    # return average values
    return total_vals


def calc_total_force(nodes, index, forces, data_direction):
    """
    Calculates the total force on the external face specified by `index` in the direction specified by `data_direction`.
    Forces are calculated for the minimum and maximum faces to check the values are the same, i.e. the job is in
    equilibrium, but only the force data for one face is returned.
    :param nodes: 2D Numpy array, dtype=`float`. List of nodal coordinates.
    :param index: `Int`. Index of the direction to search for the min and max nodes.
    :param forces: 1D Numpy array, dtype=`float`. Force data to total.
    :param data_direction: `Int`. Index of the direction for the total force to return.
    :return: 'Float'. Total force.
    """
    max_nodes, min_nodes = get_face_nodes(nodes, index)
    max_force = calc_total_value(max_nodes, forces)
    min_force = calc_total_value(min_nodes, forces)

    assert np.allclose(np.abs(max_force), np.abs(min_force), atol=1e-7), "Forces on max and min faces are not equal."

    return max_force[data_direction]


def calc_average_displacements(nodes, index, displacements, data_direction):
    """
    Calculates the average displacements of the displacements on the min and max faces of `nodes`.
    :param nodes: 2D Numpy array, dtype=`float`. List of nodal coordinates.
    :param index: `Int`. Index of the direction to search for the min and max nodes.
    :param displacements: 1D Numpy array, dtype=`float`. Dispalcement data to average.
    :param data_direction: `Int`. Index of the direction for averaged displacements over the faces to return.
    :return: `Float`. Average displacement.
    """
    max_nodes, min_nodes = get_face_nodes(nodes, index)
    max_disp = calc_total_value(max_nodes, displacements) / max_nodes.shape[0]
    min_disp = calc_total_value(min_nodes, displacements) / min_nodes.shape[0]
    delta_disp = max_disp - min_disp
    return delta_disp[data_direction]


def process_data(job, files, path_to_files):
    """
    Process the FEA output and returns the non-normalized Young's modulus, Poisson ratio,
    shear modulus, and bulk modulus. The shear modulus and bulk modulus are only calculated
    if displacement and force data are included for those types of loadings.
    :param job: `BravaisJob`. The job analyzed via FEA that the `files` correspond to.
    :param files: `List`, dtype=`String`. List of files that contain the nodal force and displacement data
                   (Ex. `['axial_displacements.txt', 'axial_forces.txt']`). The displacement data must have
                   "displacements" in the file name, and the force data must have the word "forces" in its
                   file name. The load type ('axial', 'shear', or 'bulk') must also be in the file name
                   corresponding to the loading the data belongs to.
    :param path_to_files: `String`. Path to the specified `files`.
    :return: Non-normalized elastic constants in the form
    `(youngs_modulus, poisson_ratio, shear_modulus, bulk_modulus)`. Note that the shear and bulk moduli will only
    contain useful data if bulk and shear data are available in the files that are input.
    """
    # split files into displacement or force categories
    displacement_files, force_files = categorize_files(files, ('displacements', 'forces'))

    # load force and displacement data using correct file names
    displacement_data = NodalData(displacement_files, path_to_files)
    force_data = NodalData(force_files, path_to_files)

    # find the lengths along each direction of the job's nodes
    dimensionality = len(displacement_data.axial[0])
    lengths = np.empty(dimensionality)
    for i in range(dimensionality):
        max_val = job.nodes[:, i].max()
        min_val = job.nodes[:, i].min()
        lengths[i] = max_val - min_val

    # initialize arrays to hold strains
    # format of strains:
    # axial -> [e11, e22, e33]
    # bulk  -> [e11, e22, e33]
    # shear -> [e12, e21]
    strains_axial = np.empty(dimensionality)
    strains_bulk = np.empty_like(strains_axial)
    strains_shear = np.empty(2)

    for i in range(strains_axial.shape[0]):
        strains_axial[i] = calc_average_displacements(job.nodes, i, displacement_data.axial, i) / lengths[i]

    if displacement_data.bulk is not None:
        for i in range(strains_bulk.shape[0]):
            strains_bulk[i] = calc_average_displacements(job.nodes, i, displacement_data.bulk, i) / lengths[i]

    if displacement_data.shear is not None:
        strains_shear[0] = calc_average_displacements(job.nodes, 0, displacement_data.shear, 1) / lengths[0]
        strains_shear[1] = calc_average_displacements(job.nodes, 1, displacement_data.shear, 0) / lengths[1]

    # initialize arrays to hold stresses
    stresses_axial = np.empty_like(strains_axial)
    stresses_bulk = np.empty_like(strains_bulk)
    stresses_shear = np.empty_like(strains_shear)

    for i in range(dimensionality):
        stresses_axial[i] = calc_total_force(job.nodes, i, force_data.axial, i) / \
            (lengths[(i + 1) % dimensionality] * lengths[(i + 2) % dimensionality])

    if force_data.bulk is not None:
        for i in range(dimensionality):
            stresses_bulk[i] = calc_total_force(job.nodes, i, force_data.bulk, i) / \
                (lengths[(i + 1) % dimensionality] * lengths[(i + 2) % dimensionality])

    if force_data.shear is not None:
        stresses_shear[0] = calc_total_force(job.nodes, 0, force_data.shear, 1) / (lengths[1] * lengths[2])
        stresses_shear[1] = calc_total_force(job.nodes, 1, force_data.shear, 0) / (lengths[0] * lengths[2])

    # based on stresses and strains calculate the elastic properties
    youngs_modulus = stresses_axial[0] / strains_axial[0]
    poisson_ratio = -(strains_axial[1] + strains_axial[2]) / 2.0 * (1 / strains_axial[0])

    shear_modulus = -1.0e18
    if force_data.shear is not None:
        avg_shear_stress = np.sum(stresses_shear) / len(stresses_shear)
        shear_modulus = avg_shear_stress / np.sum(strains_shear)

    bulk_modulus = -1.0e18
    if force_data.bulk is not None:
        dilation = np.sum(strains_bulk)
        avg_bulk_stress = np.sum(stresses_bulk) / 3.0
        bulk_modulus = avg_bulk_stress / dilation

    return youngs_modulus, poisson_ratio, shear_modulus, bulk_modulus
