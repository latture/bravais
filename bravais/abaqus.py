__all__ = ["run_job", "run_script"]

import subprocess

def run_job(inp_file, abaqus_executable='C:/SIMULIA/Abaqus/Commands/abq6135', num_cpus=1, double_precision=True):
    """
    Call abaqus via the command line and runs the FE analysis contained in the .inp file.
    :param inp_file: `String`. INP file to analyze, i.e. `path/to/file/filename.inp`.
    :param abaqus_executable: `String`. Path to abaqus executable, i.e. `path/to/abaqus`.
                                        Default = `C:/SIMULIA/Abaqus/Commands/abq6135`
    :param num_cpus: `Int`. Number of cpus to use for analysis. Default = 1.
    :param double_precision: `Bool`, Default=`True`. Whether to run the analysis using double precision.
    """
    shell_command = '%s interactive job=%s cpus=%d' % (abaqus_executable, inp_file, num_cpus)
    if double_precision:
        shell_command += ' output_precision=full double'
    subprocess.call(shell_command, shell=True)


def run_script(script, options=None, abaqus_executable='C:/SIMULIA/Abaqus/Commands/abq6135'):
    """
    Runs the specified script using the Python interpreter included with the Abaqus installation.
    :param script: `String`. Script to run, i.e. `path/to/script.py`.
    :param options: `Tuple`. Any command line options to pass to the script.
                             Must be in the form `((flag1, value1), (flag2, value2))`.
    :param abaqus_executable: String`. Path to abaqus executable, i.e. `path/to/abaqus`.
                                       Default = `C:/SIMULIA/Abaqus/Commands/abq6135`
    """
    print "Running script: %s" % script
    shell_command = '%s python %s' % (abaqus_executable, script)
    if options is not None:
        for opt in options:
            assert len(opt) == 2, "Specified options must have a flag and associated value."
            shell_command += ' %s %s' % (opt[0], opt[1])
    subprocess.call(shell_command, shell=True)
    print "Script completed."
