__author__ = 'ryan'

import sys
import os
import platform
import time
sys.path.append('../')

if platform.system() == 'Windows':
    from bravais import abaqus
else:
    from bravais.unit_cells import SimpleCubic, FCC, BCC, UnitData
    from bravais.mesher import mesh_bravais
    from bravais.inp_creator import create_inp
    from bravais.python_utils import calc_radius


def main():
    if platform.system() != 'Windows':
        num_elems = 12
        ar1 = ar2 = 25.0
        dimx = dimy = dimz = 20
        dims_str = '_%dx%dx%d_' % (dimx, dimy, dimz)
        sc_unit = SimpleCubic(a=1.0, num_elems=num_elems)
        bcc_unit = BCC(a=1.0, num_elems=num_elems)
        fcc_unit = FCC(a=1.0, num_elems=num_elems)

        unit_data = (
            UnitData(unit_cells=(sc_unit,), label='SimpleCubic', aspect_ratios=(ar1,)),
            UnitData(unit_cells=(bcc_unit,), label='BCC', aspect_ratios=(ar1,)),
            UnitData(unit_cells=(fcc_unit,), label='FCC', aspect_ratios=(ar1,)),
            UnitData(unit_cells=(sc_unit, fcc_unit), label='FCC-SC', aspect_ratios=(ar1, ar2)),
            UnitData(unit_cells=(sc_unit, bcc_unit), label='BCC-SC', aspect_ratios=(ar1, ar2)),
        )
        load_types = ('axial', 'bulk', 'shear')

        for data in unit_data:
            jobs = []
            radii = []
            for i, unit_cell in enumerate(data.unit_cells):
                j = mesh_bravais(unit_cell, dimx, dimy, dimz)
                jobs.append(j)
                radii.append(calc_radius(unit_cell, num_elems, data.aspect_ratios[i]))

            for lt in load_types:
                filename = data.label + dims_str + lt
                create_inp(filename, jobs, radii=radii, load_type=lt, strain=0.1)
    else:
        all_files = os.listdir(os.getcwd())

        # run all .inp files in cwd through Abaqus
        for f in all_files:
            fbase, ext = f.split('.')
            if ext == 'inp':
                abaqus.run_job(fbase)

                abaqus.run_script('../bravais/extract_nodal_component.py',
                                  options=(('-o', fbase+'.odb'), ('-c', 'U'), ('-f', fbase+'_displacements.txt')))

                abaqus.run_script('../bravais/extract_nodal_component.py',
                                  options=(('-o', fbase+'.odb'), ('-c', 'RF'), ('-f', fbase+'_forces.txt')))

        all_files = os.listdir(os.getcwd())

        # remove all unneeded files
        ext_to_keep = ['odb', 'txt', 'inp', 'py']
        unit_labels = ['SimpleCubic', 'BCC', 'FCC', 'FCC-SC', 'BCC-SC']
        for f in all_files:
            try:
                fbase, ext = f.split('.')
                if ext not in ext_to_keep:
                    os.remove(f)
            except ValueError:
                pass

        # create needed directories
        data_dir = os.getcwd()+'/data/'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for label in unit_labels:
            if not os.path.exists(data_dir+label):
                os.mkdir(data_dir+label)

        # move files into correct folders
        all_files = os.listdir(os.getcwd())
        for label in unit_labels:
            for f in all_files:
                if label+'_' in f:
                    os.rename(os.getcwd()+'/'+f, data_dir+label+'/'+f)


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print "Program completed in %.3f seconds" % (t2 - t1)
