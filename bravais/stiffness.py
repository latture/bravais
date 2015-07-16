
__all__ = ["plot_stiffness_slice", "plot_stiffness_slice", "Props"]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors
from collections import namedtuple

Props = namedtuple("Props", ["E", "G", "nu", "label"])


def calc_stiffness(E, G, nu, direction_vec):
    """
    Calculates the stiffness along a given direction for a material 
    with cubic symmetry and the specified elastic constants.
    :param E             : `Float`. Young's modulus.
    :param G             : `Float`. Shear modulus.
    :param nu            : `Float`. Poisson ratio.
    :param direction_vec : `array_like`, `len(direction_vec)==3`. Direction for which the stiffness is calculated.
    :return              : `Float`. The stiffness along the specified direction.
    """
    nu12 = nu13 = nu23 = nu
    G12 = G13 = G23 = G
    E1 = E2 = E3 = E

    S = np.array([[1.0 / E1,    -nu12 / E2,  -nu13 / E3,  0.0,        0.0,        0.0],
                  [-nu12 / E1,  1.0 / E2,    -nu23 / E3,  0.0,        0.0,        0.0],
                  [-nu13 / E1,  -nu23 / E2,  1.0 / E3,    0.0,        0.0,        0.0],
                  [0.0,         0.0,         0.0,         1.0 / G23,  0.0,        0.0],
                  [0.0,         0.0,         0.0,         0.0,        1.0 / G13,  0.0],
                  [0.0,         0.0,         0.0,         0.0,        0.0,        1.0 / G12]])
    C = np.linalg.inv(S)

    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    exx = direction_vec / np.linalg.norm(direction_vec)

    m1 = np.dot(e1, exx)
    m2 = np.dot(e2, exx)
    m3 = np.dot(e3, exx)

    return 1.0 / (S[0, 0] - 2.0 * (S[0, 0] - S[0, 1] - S[3, 3] / 2.0) * (m1**2 * m2**2 + m2**2 * m3**2 + m3**2.0 * m1**2))


def plot_stiffness_surface(props, num_rings=50, num_sectors=50, vmin=0.6, vmax=1.4,
                           print_minmax=False, showfig=True, savefig=False):
    """
    Plots the stiffness surface for the supplied properties.
    :param props        : `Props`. Defines the elastic properties for the material, e.g. E, G, and nu.
    :param num_rings    : `Int`, default=50. Number of latitudinal rings to evaluate the stiffness at.
    :param num_sectors  : `Int`, default=50. Number of longitudinal rings to evaluate the stiffness at.
    :param vmin         : `Float`, default=0.6. Lower bound on normalization for colormap.
    :param vmax         : `Float`, default=1.4. Upper bound on normalization for colormap.
    :param print_minmax : `Bool`, default=`False`. Whether to print the bounds on stiffness found while computing the surface.
    :param showfig      : `Bool`, default=`True`. Whether to display the resulting figure.
    :param savefig      : `Bool`, default=`False`. Whether to save the current figure to a file.
    """
    upper_bound = 1.0 / 6.0

    u = np.linspace(0.0, 2.0 * np.pi, num_sectors)
    v = np.linspace(0.0, np.pi, num_rings)

    x_data = np.outer(np.cos(u), np.sin(v))
    y_data = np.outer(np.sin(u), np.sin(v))
    z_data = np.outer(np.ones(np.size(u)), np.cos(v))
    rows, cols = x_data.shape

    cmap = plt.get_cmap('coolwarm')
    normalization = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(normalization, cmap=cmap)

    colors = np.zeros((rows, cols, 4))
    mag_min = 1e20
    mag_max = -1e20

    for i in range(rows):
        for j in range(cols):

            direction = np.array([x_data[i, j], y_data[i, j], z_data[i, j]])
            mag = calc_stiffness(props.E, props.G, props.nu, direction)

            x_data[i, j] *= mag
            y_data[i, j] *= mag
            z_data[i, j] *= mag

            mag /= upper_bound

            colors[i, j] = m.to_rgba(mag)
            colors[i, -1] = 0.0

            if mag > mag_max:
                mag_max = mag
            elif mag < mag_min:
                mag_min = mag
    if print_minmax:
        print "%s: min=%.3f\tmax=%.3f" % (props.label, mag_min*upper_bound, mag_max*upper_bound)

    fig = plt.figure(figsize=(16, 20))
    axis = fig.add_subplot(111, projection='3d', aspect='equal')
    plt.rcParams.update({'font.size': 24})

    axis.set_zlim3d([-0.16, 0.16])
    axis.set_ylim3d([-0.16, 0.16])
    axis.set_xlim3d([-0.16, 0.16])
    p = axis.plot_surface(x_data, y_data, z_data, rstride=1, cstride=1, facecolors=colors, linewidth=0)

    if savefig:
        plt.savefig(props.label + "_polar_surface.png", dpi=250, transparent=True, bbox_inches='tight')

    if showfig:
        plt.show()
    plt.close('all')


def plot_stiffness_slice(props, v, ulim=(0.0,2*np.pi), num_sectors=50,
                         print_minmax=False, savefig=False, showfig=True):
    """
    Plots a slice of the stiffness surface for the specified material.
    :param props        : `Props`. Defines the elastic properties for the material, e.g. E, G, and nu.
    :param v            : `Float`. Azimuthal spherical coordinate of slice. Should be on the range `[0.0, pi]`.
    :param ulim         : `array_like`, default=`[0.0, 2*pi]`.`len(ulim)==2`.
                           Limits for the longitudinal sphereical coordinates of the slice.
                           Should be on the range `[0.0, 2*pi]`.
    :param num_sectors  : `Int`, default=50. Number of longitudinal rings to evaluate the stiffness at.
    :param print_minmax : `Bool`, default=`False`. Whether to print the bounds on stiffness found while computing the surface.
    :param showfig      : `Bool`, default=`True`. Whether to display the resulting figure.
    :param savefig      : `Bool`, default=`False`. Whether to save the current figure to a file.
    """
    upper_bound = 1.0 / 6.0

    assert len(ulim) == 2, "ulim must have length of 2 for minimum and maximum values."
    u = np.linspace(ulim[0], ulim[1], num_sectors)

    x_data = np.cos(u) * np.sin(v)
    y_data = np.sin(u) * np.sin(v)
    z_data = np.cos(v)

    r_data = np.empty_like(x_data)

    mag_min = 1e20
    mag_max = -1e20

    for i in range(num_sectors):
        direction = np.array([x_data[i], y_data[i], z_data])
        mag = calc_stiffness(props.E, props.G, props.nu, direction)

        r_data[i] = mag

        if mag > mag_max:
            mag_max = mag
        elif mag < mag_min:
            mag_min = mag

    if print_minmax:
        print "%s: min=%.3f\tmax=%.3f" % (props.label, mag_min, mag_max)

    fig = plt.figure(figsize=(16/2, 20/2))
    axis = fig.add_subplot(111, polar=True, aspect='equal')
    plt.rcParams.update({'font.size': 22})
    
    axis.plot(u, r_data, linewidth=2, color='black')
    axis.plot(u, upper_bound * np.ones_like(u), linewidth=3, linestyle='-.', color='orange')
    axis.set_ylim([0.0, 0.20])

    if savefig:
        plt.savefig(props.label + "_polar_slice.png", dpi=250, transparent=True, bbox_inches='tight')

    if showfig:
        plt.show()
    plt.close('all')
