__author__ = 'ryan'
import sys
sys.path.append('../')

import platform
if platform.system() == 'Windows':
    from bravais import abaqus
else:
    from bravais import unit_cells, inp_creator, mesher
import time

def main():
    if platform.system() != 'Windows':
        unit1 = unit_cells.FCC(a=1.0, num_elems=5)
        unit2 = unit_cells.BCC(a=1.0, num_elems=5)
        jobs = (mesher.mesh_bravais(unit1, 3, 3, 3), mesher.mesh_bravais(unit2, 3, 3, 3))
        inp_creator.create_inp("test", jobs, radii=[0.01, 0.02], load_type='axial', strain=1.0/3.0)
    else:
        abaqus.run_job("test")
        abaqus.run_script('extract_nodal_component.py',
                          options=(('-o', 'test.odb'), ('-c', 'U'), ('-f', 'disp_axial.txt')))

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print "Program completed in %.3f seconds" % (t2 - t1)
