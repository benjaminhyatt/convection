"""
Make frames of profiles

Usage:
    plot_profiles.py <processed>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames] 
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib import transforms
from dedalus.extras import plot_tools

import pathlib
from docopt import docopt
from dedalus.tools import logging
from dedalus.tools import post
from dedalus.tools.parallel import Sync

args = docopt(__doc__)
if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    output_path = './frames'

# Create output directory if needed
with Sync() as sync:
    if sync.comm.rank == 0:
        if not output_path.exists():
            output_path.mkdir()

dpi = 200
title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
savename_func = lambda write: 'write_{:06}.png'.format(write)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.size'] = 8
plt.rcParams['figure.dpi'] = dpi


t_mar, b_mar, l_mar, r_mar = (0.3, 0.3, 0.3, 0.1)
golden_mean = (np.sqrt(5) - 1.) / 2.
h_plot, w_plot = (1., 1. / golden_mean)
w_pad = l_mar

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + w_pad + w_plot + r_mar

fig_width = 5.5
scale = fig_width/w_total

fig = plt.figure(figsize = (scale * w_total, scale * h_total))

##### construct axs #####
left1 = (l_mar) / w_total
bottom1 = 1 - (t_mar + h_plot) / h_total
width1 = w_plot / w_total
height1 = h_plot / h_total

left2 = (l_mar + w_plot + w_pad) / w_total
bottom2 = 1 - (t_mar + h_plot) / h_total
width2 = w_plot / w_total
height2 = h_plot / h_total

# Plot writes
f = np.load(args['<processed>'][0], allow_pickle = True)[()]
print(f.keys())
print(f[0].keys())
print(f['tasks'], f['subtasks'])
nframes = f['nout']
tasks = f['tasks']
subtasks = f['subtasks']
labels = f['labels']

progress_cad = np.ceil(nframes/50)
for index in range(nframes):
    print(index)
    ax1 = fig.add_axes([left1, bottom1, width1, height1])
    ax2 = fig.add_axes([left2, bottom2, width2, height2])
    
    task = tasks[0]
    print(task)
    for m, subtask in enumerate(subtasks[task]):
        print(subtask)
        xdata = f[index][task]['z']
        ydata = f[index][task]['data_' + subtask].ravel()
        ax1.plot(xdata, ydata, linewidth = 1, label = labels[task][m])
    ax1.set_xlabel(r'$z$')
    ax1.legend(fontsize = 4, loc = "lower right")

    task = tasks[1]
    print(task)
    for m, subtask in enumerate(subtasks[task]):
        print(subtask)
        xdata = f[index][task]['z']
        ydata = f[index][task]['data_' + subtask].ravel()
        ax2.plot(xdata, ydata, linewidth = 1, label = labels[task][m])
    ax2.set_xlabel(r'$z$')
    ax2.set_ylim(-0.5, 0.6)
    ax2.legend(fontsize = 4, loc = "lower right")

    # Add time title
    title = title_func(f[index][task]['t'])
    title_height = 1 - 0.25 * t_mar
    fig.suptitle(title, x=0.44, y=title_height, ha='left')
    # Save figure
    savename = savename_func(index)
    savepath = output_path.joinpath(savename)
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()

    if index % progress_cad == 0:
        print("(%d / %d) writes processed" %(index + 1, nframes))

plt.close(fig)

