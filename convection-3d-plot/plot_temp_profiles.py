"""
Make frames of profiles

Usage:
    plot_temp_profiles.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames] 
"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import transforms
from dedalus.extras import plot_tools

from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main(filename, start, count, output):
    
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
    h_pad = 0.3 * h_plot
    w_pad = 0.3 * w_plot

    h_total = t_mar + h_plot + h_pad + h_plot + b_mar
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

    left3 = (l_mar) / w_total
    bottom3 = 1 - (t_mar + h_plot + h_pad + h_plot) / h_total
    width3 = w_plot / w_total
    height3 = h_plot / h_total

    left4 = (l_mar + w_plot + w_pad) / w_total
    bottom4 = 1 - (t_mar + h_plot + h_pad + h_plot) / h_total
    width4 = w_plot / w_total
    height4 = h_plot / h_total

    # Plot writes
    f = np.load(filename, allow_pickle = True)[()]
    nframes = f['nout']
    tasks = f['tasks']
    subtasks = f['subtasks']
    labels = f['labels']

    progress_cad = np.ceil(count/20)

    for index in range(start, start+count):
 
        if index % progress_cad == 0 and rank == 0:
            frac = (index)/count
            percent = '{:.3}'.format(100*frac)
            print("Rank {:}:".format(rank), percent, "% complete")

        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        ax2 = fig.add_axes([left2, bottom2, width2, height2])
        ax3 = fig.add_axes([left3, bottom3, width3, height3])
        ax4 = fig.add_axes([left4, bottom4, width4, height4])

        task = tasks[0]
        for m, subtask in enumerate(subtasks[task]):
            xdata = f[index][task]['z']
            ydata = f[index][task]['data_' + subtask].ravel()
            ax1.plot(xdata, ydata, linewidth = 1, label = labels[task][m])
        ax1.set_xlabel(r'$z$')
        ax1.legend(fontsize = 4, loc = "upper right")

        task = tasks[1]
        for m, subtask in enumerate(subtasks[task]):
            xdata = f[index][task]['z']
            ydata = f[index][task]['data_' + subtask].ravel()
            ax2.plot(xdata, ydata, linewidth = 1, label = labels[task][m])
        ax2.vlines(xdata[f[index][task]['zoom_idxs'][1]], np.min(ydata), np.max(ydata), linewidth = 0.4, linestyle = "dashed", color = "black")
        ax2.set_xlabel(r'$z$')
        ax2.legend(fontsize = 4, loc = "upper right")

        task = tasks[2]
        for m, subtask in enumerate(subtasks[task]):
            xdata = f[index][task]['z']
            ydata = f[index][task]['data_' + subtask].ravel()
            ax3.plot(xdata, ydata, linewidth = 1, label = labels[task][m])
        ax3.vlines(xdata[f[index][task]['zoom_idxs'][1]], np.min(ydata), np.max(ydata), linewidth = 0.4, linestyle = "dashed", color = "black")
        ax3.set_xlabel(r'$z$')
        ax3.legend(fontsize = 4, loc = "upper right")

        task = tasks[3]
        for m, subtask in enumerate(subtasks[task]):
            xdata = f[index][task]['z']
            ydata = f[index][task]['data_' + subtask].ravel()
            ax4.plot(xdata, ydata, linewidth = 1, label = labels[task][m])
        ax4.vlines(xdata[f[index][task]['zoom_idxs'][1]], np.min(ydata), np.max(ydata), linewidth = 0.4, linestyle = "dashed", color = "black")
        ax4.set_xlabel(r'$z$')
        ax4.legend(fontsize = 4, loc = "upper right")

        # Add time title
        title = title_func(f[index][task]['t'])
        title_height = 1 - 0.125 * t_mar
        fig.suptitle(title, x=0.44, y=title_height, ha='left')
        # Save figure
        savename = savename_func(index)
        savepath = output.joinpath(savename)
        fig.savefig(str(savepath), dpi=dpi)
        fig.clear()

    plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    import post_npy
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
    post_npy.visit_writes(args['<files>'], main, output=output_path)
