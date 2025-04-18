"""
Plot scalars.

Usage:
    plot_scalars.py <file> [--output=<dir>]


Options:
    --output=<dir>  Output directory [default: ./scalars]

"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib
from docopt import docopt

args = docopt(__doc__)

# Parameters
tasks = ['mean_Re', 'mean_T1_bot']
figsize = (6, 4)
log_scale = False

# Plot
for task in tasks:

    fig = plt.figure(figsize=figsize)
    with h5py.File(args['<file>'], mode='r') as file:
        t = np.array(file['scales/sim_time'])
        dset = file['tasks'][task]
        plt.plot(t, dset[:].ravel(), label=task)
    plt.xlabel('t')
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    if args['--output'] is not None:
        savename = '{:s}'.format(args['--output']) + '_' + '{:s}'.format(task) + '.png'
    else:
        savename = 'scalars_' + '{:s}'.format(task) + '.png'
    print("Saving figure", savename)
    plt.savefig(savename)
