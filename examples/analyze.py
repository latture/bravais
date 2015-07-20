import sys
import numpy as np
import os
import time
import copy

sys.path.append('../')
from bravais.unit_cells import SimpleCubic, FCC, BCC, UnitData
from bravais.mesher import mesh_bravais
from bravais.post_process import process_data, plot_data, COLORS
from bravais.python_utils import add_tie_line
from bravais.stiffness import plot_stiffness_surface, Props


def analyze():
    num_elems = 6
    a = 1.0
    E_o = 200.0e9
    dimx = dimy = dimz = 10
    percents = np.linspace(0.1, 0.9, 9)
    percents = np.insert(percents, 0, 0.001)
    percents = np.append(percents, 0.999)
    relative_density = 0.01
    sc_unit = SimpleCubic(a=a, num_elems=num_elems)
    bcc_unit = BCC(a=a, num_elems=num_elems)
    fcc_unit = FCC(a=a, num_elems=num_elems)

    unit_data = (
        UnitData(unit_cells=(sc_unit, bcc_unit), label='BCC-SC'),
        UnitData(unit_cells=(sc_unit, fcc_unit), label='FCC-SC'),
    )

    unit_colors = {
        'BCC-SC' : (COLORS["orange"], COLORS["black"]),
        'FCC-SC' : (COLORS["purple"], COLORS["black"])
    }

    data_dir = '/home/ryan/Desktop/bravais-cnsi/examples/fea_data/'

    for data in unit_data:
        E = np.empty(len(percents))
        G = np.empty_like(E)
        nu = np.empty_like(E)
        K = np.empty_like(E)

        job = mesh_bravais(data.unit_cells[0], dimx, dimy, dimz)
        if len(data.unit_cells) > 1:
            for i in range(1, len(data.unit_cells)):
                job.merge(mesh_bravais(data.unit_cells[i], dimx, dimy, dimz))

        for i, p in enumerate(percents):
            percent_str = ('%0.f' % (p * 100)).zfill(3) + '-percent'
            current_dir = data_dir + data.label + "/"
            current_files = os.listdir(current_dir)
            txt_files = []
            for f in current_files:
                try:
                    fbase, ext = f.split('.')
                    if ext == 'txt' and percent_str in fbase:
                        txt_files.append(f)
                except ValueError:
                    pass

            print "Processing %s %s..." % (data.label, percent_str)
            E_i, v_i, G_i, K_i = process_data(job, txt_files, path_to_files=current_dir)
            E[i] = E_i / (relative_density * E_o)
            nu[i] = v_i
            G[i] = G_i / (relative_density * E_o)
            K[i] = K_i / (relative_density * E_o)
            plot_stiffness_surface(Props(E=E[i], G=G[i], nu=nu[i], label=data.label+'-'+percent_str+'_SC'),
                                   showfig=False, savefig=True, num_rings=100, num_sectors=100,
                                   plot_title=('%.0f%% SC' % (p * 100)), transparent=False)

        E = add_tie_line(percents, E)
        G = add_tie_line(percents, G)
        nu = add_tie_line(percents, nu)
        K = add_tie_line(percents, K)

        unit_cell_names = [data.label, data.label + "_tie_line"]

        plot_data(percents, E, unit_cell_names, xlabel="Percent SC, %", 
                  markers=[u'o', u''], linestyles=['-', '-.'], colors=unit_colors[data.label],
                  ylabel="Young's Modulus, $E_{1}/\\rho E_{o}$", xlim=[0, 1], ylim=[-0.01, 0.35],
                  showfig=False, savefig=True, filename=data.label+'_youngs.svg')

        plot_data(percents, nu, unit_cell_names, xlabel="Percent SC, %",
                  markers=[u'o', u''], linestyles=['-', '-.'], colors=unit_colors[data.label],
                  ylabel="Poission ratio, $\\nu_{12}$", xlim=[0, 1], ylim=[-0.01, 0.51],
                  showfig=False, savefig=True, filename=data.label+'_poisson.svg')

        plot_data(percents, G, unit_cell_names, xlabel="Percent SC, %",
                  markers=[u'o', u''], linestyles=['-', '-.'], colors=unit_colors[data.label],
                  ylabel="Shear Modulus, $G_{12}/\\rho E_{o}$", xlim=[0, 1], ylim=[-0.01, 0.35],
                  showfig=False, savefig=True, filename=data.label+'_shear.svg')

        plot_data(percents, K, unit_cell_names, xlabel="Percent SC, %", 
                  markers=[u'o', u''], linestyles=['-', '-.'], colors=unit_colors[data.label],
                  ylabel="Bulk modulus, $K/\\rho E_{o}$", xlim=[0, 1], ylim=[-0.01, 0.35],
                  showfig=False, savefig=True, filename=data.label+'_bulk.svg')
                               

if __name__ == '__main__':
    t1 = time.time()
    analyze()
    t2 = time.time()
    print "Program completed in %.3f seconds" % (t2 - t1)

