__author__ = 'ryan'

import os
import sys
import time
sys.path.append('../')

from bravais import abaqus
from bravais.cli_progress import cli_progress


def run():
    inp_files = []
    for f in os.listdir(os.getcwd()):
        try:
            fbase, ext = f.split('.')
            if ext == 'inp':
                inp_files.append(fbase)
        except ValueError:
            pass

    counter = 0
    num_inp_files = len(inp_files)

    # run all .inp files in cwd through Abaqus
    for fbase in inp_files:
        abaqus.run_job(fbase, abaqus_executable='C:/SIMULIA/Abaqus/Commands/abaqus')

        abaqus.run_script('../bravais/extract_nodal_component.py', abaqus_executable='C:/SIMULIA/Abaqus/Commands/abaqus',
                  options=(('-o', fbase+'.odb'), ('-c', 'SE'), ('-f', fbase+'_elemental_strains.txt')))
        abaqus.run_script('../bravais/extract_nodal_component.py', abaqus_executable='C:/SIMULIA/Abaqus/Commands/abaqus',
                          options=(('-o', fbase+'.odb'), ('-c', 'U'), ('-f', fbase+'_nodal_displacements.txt')))
        abaqus.run_script('../bravais/extract_nodal_component.py', abaqus_executable='C:/SIMULIA/Abaqus/Commands/abaqus',
                          options=(('-o', fbase+'.odb'), ('-c', 'RF'), ('-f', fbase+'_nodal_forces.txt')))
        abaqus.run_script('../bravais/extract_nodal_component.py', abaqus_executable='C:/SIMULIA/Abaqus/Commands/abaqus',
                          options=(('-o', fbase+'.odb'), ('-c', 'RM'), ('-f', fbase+'_nodal_moments.txt')))
        abaqus.run_script('../bravais/extract_nodal_component.py', abaqus_executable='C:/SIMULIA/Abaqus/Commands/abaqus',
                          options=(('-o', fbase+'.odb'), ('-c', 'SF'), ('-f', fbase+'_elem_forces.txt')))
        counter += 1
        cli_progress(counter, num_inp_files)

    # remove all unneeded files
    ext_to_keep = ['odb', 'txt', 'inp', 'py']
    unit_labels = ['FCC']
    for f in os.listdir(os.getcwd()):
        try:
            fbase, ext = f.split('.')
            if ext not in ext_to_keep:
                os.remove(f)
        except ValueError:
            pass

    # create needed directories
    data_dir = os.getcwd()+'/fea_data/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for label in unit_labels:
        if not os.path.exists(data_dir+label):
            os.mkdir(data_dir+label)

    # move files into correct folders
    for label in unit_labels:
        for f in os.listdir(os.getcwd()):
            if label+'_' in f and f[-3:] != 'inp':
                os.rename(os.getcwd()+'/'+f, data_dir+label+'/'+f)

if __name__ == '__main__':
    t1 = time.time()
    run()
    t2 = time.time()
    print "Program completed in %.3f seconds" % (t2 - t1)
