__all__ = ["cli_progress"]

import sys

def cli_progress(i, end_val, bar_length=20):
    """
    Displays a progress bar in the command line.
    :param i: `Int`. Current value.
    :param end_val: `Int`. End value when progress = 100%.
    :bar_length: `Int`. Length of progress bar to display in command line. 
    """
    percent = float(i) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rProgress: [{0}] {1}%\n".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()