__author__ = 'ryan'

from odbAccess import *
import optparse
import sys
import numpy as np


def main(argv):
    parser = optparse.OptionParser()

    parser.add_option('-o', '--odb', action='store', type='string',
                      dest='odb_filename', help='odb file to extract data.')
    parser.add_option('-c', '--component', action='store', type='string',
                      dest='odb_component', help='component of the odb file to extract.')
    parser.add_option('-f', '--filename', action='store', type='string',
                      dest='output_filename', help='Name of output file to save data to.')
    options, args = parser.parse_args(argv)
    if not options.odb_filename:
        parser.error("ODB file not specified.")
    if not options.odb_component:
        parser.error("ODB component not specified.")
    if not options.output_filename:
        parser.error("Output file not specified.")

    odb = openOdb(path=options.odb_filename)

    field = odb.steps.values()[-1].frames[-1].fieldOutputs[options.odb_component]
    labels = field.componentLabels
    try:
        data = np.empty((len(field.values), len(field.values[0].dataDouble)))
        for i in xrange(len(field.values)):
            data[i] = field.values[i].dataDouble
    except OdbError:
        print "WARNING: double precision data not available."
        data = np.empty((len(field.values), len(field.values[0].data)))
        for i in xrange(len(field.values)):
            data[i] = field.values[i].data

    print "Saving component %s to %s." % (options.odb_component, options.output_filename)
    np.savetxt(options.output_filename, data, fmt='%.18f')


if __name__ == '__main__':
    main(sys.argv[1:])
