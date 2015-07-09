import sys
import numpy as np
import os
import time
import copy

sys.path.append('../')
from bravais.unit_cells import SimpleCubic, FCC, BCC, UnitData
from bravais.mesher import mesh_bravais
from bravais.python_utils import calc_radius, calc_rel_density
from bravais.post_process import process_data, plot_data


def main():
    num_elems = 12
    ar1 = ar2 = 25.0
    a = 1.0
    E_o = 200.0e9
    dimx = dimy = dimz = 20
    sc_unit = SimpleCubic(a=a, num_elems=num_elems)
    bcc_unit = BCC(a=a, num_elems=num_elems)
    fcc_unit = FCC(a=a, num_elems=num_elems)
    dimensions = sc_unit.nodes[0].shape[-1]

    unit_data = (
        UnitData(unit_cells=(sc_unit,), label='SimpleCubic', aspect_ratios=(ar1,)),
        UnitData(unit_cells=(bcc_unit,), label='BCC', aspect_ratios=(ar1,)),
        UnitData(unit_cells=(fcc_unit,), label='FCC', aspect_ratios=(ar1,)),
        UnitData(unit_cells=(sc_unit, bcc_unit), label='SC-BCC', aspect_ratios=(ar1, ar2)),
        UnitData(unit_cells=(sc_unit, fcc_unit), label='SC-FCC', aspect_ratios=(ar1, ar2))
    )

    E = np.empty((len(unit_data), dimensions))
    G = np.empty_like(E)
    v = np.empty_like(E)
    K = np.empty_like(E)

    data_dir = os.getcwd() + '/data/'

    for data in unit_data:
        jobs = []
        radii = []
        areas = []
        for i, unit_cell in enumerate(data.unit_cells):
            j = mesh_bravais(unit_cell, dimx, dimy, dimz)
            jobs.append(j)
            radius = calc_radius(unit_cell, num_elems, data.aspect_ratios[i])
            radii.append(radius)
            areas.append(np.pi * radius**2)

        if len(data.unit_cells) > 1:
            j = copy.deepcopy(jobs[0])
            for k in range(1, len(jobs)):
                j.merge(jobs[k])

        current_dir = data_dir + data.label + "/"
        current_files = os.listdir(current_dir)
        txt_files = []
        for f in current_files:
            try:
                fbase, ext = f.split('.')
                if ext == 'txt':
                    txt_files.append(f)
            except ValueError:
                pass

        print "Processing %s..." % data.label
        E_i, v_i, G_i, K_i = process_data(j, current_files, path_to_files=current_dir)
        total_volume = a**3
        relative_density = calc_rel_density(total_volume, jobs, areas)
        E[i, j] = E_i / (relative_density * E_o)
        v[i, j] = v_i
        G[i, j] = G_i / (relative_density * E_o)
        K[i, j] = K_i / (relative_density * E_o)

    unit_cell_names = []
    for data in unit_data:
        unit_cell_names.append(data.label)

    plot_data(dimensions, E, unit_cell_names, xlabel="Number, n", ylabel="Young's Modulus, $E_{1}/\\rho E_{o}$",
                           xlim=[0, 27], ylim=[-0.01, 0.35])
    plot_data(dimensions, v, unit_cell_names, xlabel="Number, n", ylabel="Poission ratio, $\\nu_{12}$",
                           xlim=[0, 27], ylim=[-0.01, 0.51])
    plot_data(dimensions, G, unit_cell_names, xlabel="Number, n", ylabel="Shear Modulus, $G_{12}/\\rho E_{o}$",
                           xlim=[0, 27], ylim=[-0.01, 0.35])
    plot_data(dimensions, K, unit_cell_names, xlabel="Number, n", ylabel="Bulk modulus, $K/\\rho E_{o}$",
                           xlim=[0, 27], ylim=[-0.01, 0.35])

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print "Program completed in %.3f seconds" % (t2 - t1)