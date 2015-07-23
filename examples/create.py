__author__ = 'ryan'

import sys
import os
import platform
import time
import copy
import numpy as np
sys.path.append('../')

from bravais.unit_cells import SimpleCubic, FCC, BCC, UnitData
from bravais.mesher import mesh_bravais
from bravais.inp_creator import create_inp, InpDescription
from bravais.python_utils import calc_radii


def create():
    num_elems = 6
    a = 1.0
    dimx = dimy = dimz = 10
    relative_density = 0.01
    total_volume = a * dimx * dimy * dimz
    sc_unit = SimpleCubic(a=a, num_elems=num_elems)
    bcc_unit = BCC(a=a, num_elems=num_elems)
    fcc_unit = FCC(a=a, num_elems=num_elems)

    unit_data = (
        UnitData(unit_cells=(sc_unit, fcc_unit), label='FCC-SC'),
        UnitData(unit_cells=(sc_unit, bcc_unit), label='BCC-SC'),
    )

    load_types = ('axial', 'bulk', 'shear',)

    percents = np.linspace(0.01, 0.99, 99)
    percents = np.insert(percents, 0, 0.001)
    percents = np.append(percents, 0.999)

    for data in unit_data:
        jobs = []
        for unit_cell in data.unit_cells:
            jobs.append(mesh_bravais(unit_cell, dimx, dimy, dimz))

        for p in percents:
            percent_str = ('%.0f' % (p * 100)).zfill(3) + '-percent_SC'
            radii = calc_radii(jobs, (p, 1.0 - p), relative_density, total_volume)

            for lt in load_types:
                inp_desc = InpDescription(job_type=data.label,
                                          dimensions=[dimx, dimy, dimz],
                                          num_elems=num_elems,
                                          load_type=lt,
                                          misc=percent_str)
                create_inp(inp_desc, copy.deepcopy(jobs), radii=radii, load_type=lt, strain=0.1)


if __name__ == '__main__':
    t1 = time.time()
    create()
    t2 = time.time()
    print "Program completed in %.3f seconds" % (t2 - t1)
