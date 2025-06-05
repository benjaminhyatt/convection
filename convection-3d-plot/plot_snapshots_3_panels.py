"""
Make frames of snapshots

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

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

def get_bounds(C):
    Cmin = np.min(C)
    Cmax = np.max(C)
    Cabs = np.max((np.abs(Cmin), np.abs(Cmax)))
    if Cabs != 0:
        Clow = -1. * Cabs
        Chigh = Cabs
        return Clow, Chigh
    else:
        Clow = -1.
        Chigh = 1.
        return Clow, Chigh
    return Cmin, Cmax

def main(filename, start, count, output):

    dpi = 300
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = dpi

    t_mar, b_mar, l_mar, r_mar = (0.2, 0.2, 0.3, 0.1)
    Lz = 2e0
    Lx = 1e0
    h_plot, w_plot = (1., Lx/Lz)
    w_pad = l_mar
    h_pad = 0.25 * h_plot
    h_cbar = 0.05 * h_plot
    h_pad2 = 0.125 * h_plot

    h_total = t_mar + h_pad + h_cbar + h_pad2 + h_plot + h_pad + h_cbar + h_pad2 + h_plot + h_pad + h_cbar + h_pad2 + h_plot + b_mar
    w_total = l_mar + w_plot + r_mar

    fig_width = 5.5
    scale = fig_width/w_total

    fig = plt.figure(figsize = (scale * w_total, scale * h_total))

    ##### construct axs #####
    left1 = (l_mar) / w_total
    bottom1 = 1 - (t_mar + h_pad + h_cbar + h_pad2 + h_plot) / h_total
    width1 = w_plot / w_total
    height1 = h_plot / h_total

    left2 = (l_mar) / w_total
    bottom2 = 1 - (t_mar + h_pad + h_cbar + h_pad2 + h_plot + h_pad + h_cbar + h_pad2 + h_plot) / h_total
    width2 = w_plot / w_total
    height2 = h_plot / h_total

    left3 = (l_mar) / w_total
    bottom3 = 1 - (t_mar + h_pad + h_cbar + h_pad2 + h_plot + h_pad + h_cbar + h_pad2 + h_plot + h_pad + h_cbar + h_pad2 + h_plot) / h_total
    width3 = w_plot / w_total
    height3 = h_plot / h_total



    ##### construct caxs #####
    left1c = (l_mar) / w_total
    bottom1c = 1 - (t_mar + h_pad + h_cbar) / h_total
    width1c = w_plot / w_total
    height1c = h_cbar / h_total

    left2c = (l_mar) / w_total
    bottom2c = 1 - (t_mar + h_pad + h_cbar + h_pad2 + h_plot + h_pad + h_cbar) / h_total
    width2c = w_plot / w_total
    height2c = h_cbar / h_total

    left3c = (l_mar) / w_total
    bottom3c = 1 - (t_mar + h_pad + h_cbar + h_pad2 + h_plot + h_pad + h_cbar + h_pad2 + h_plot + h_pad) / h_total
    width3c = w_plot / w_total
    height3c = h_cbar / h_total

    # Plot writes
    f = np.load(filename, allow_pickle = True)[()]
    nframes = f['nout']
    tasks = f['tasks']
    labels = f['labels']
    
    progress_cad = np.ceil(count/20)

    for index in range(start, start+count):
 
        if index % progress_cad == 0 and rank == 0:
            frac = (index)/count
            percent = '{:.3}'.format(100*frac)
            print("Rank {:}:".format(rank), percent, "% complete")

        cax1 = fig.add_axes([left1c, bottom1c, width1c, height1c])
        ax1 = fig.add_axes([left1, bottom1, width1, height1])
        cax2 = fig.add_axes([left2c, bottom2c, width2c, height2c])
        ax2 = fig.add_axes([left2, bottom2, width2, height2])
        cax3 = fig.add_axes([left3c, bottom3c, width3c, height3c])
        ax3 = fig.add_axes([left3, bottom3, width3, height3])

        task = tasks[0]
        xdata = f[index][task]['x']
        zdata = f[index][task]['z']
        X, Z = np.meshgrid(xdata, zdata)
        C = f[index][task]['data_' + task]
        Clow, Chigh = get_bounds(C)
        center = matplotlib.colors.TwoSlopeNorm(vmin = Clow, vcenter = 0., vmax = Chigh)
        plt1 = ax1.pcolormesh(X.T, Z.T, C, cmap='RdBu_r', norm=center)
        ax1.set_ylabel(r'$z$', fontsize = 14)

        cbar1 = fig.colorbar(plt1, cax = cax1, label = labels[task], orientation='horizontal')
        cax1.xaxis.set_ticks_position('top')
        
        task = tasks[1]
        xdata = f[index][task]['x']
        zdata = f[index][task]['z']
        X, Z = np.meshgrid(xdata, zdata)
        C = f[index][task]['data_' + task]
        Clow, Chigh = get_bounds(C)
        center = matplotlib.colors.TwoSlopeNorm(vmin = Clow, vcenter = 0., vmax = Chigh)
        plt2 = ax2.pcolormesh(X.T, Z.T, C, cmap='PuOr_r', norm=center)
        ax2.set_ylabel(r'$z$', fontsize = 14)

        cbar2 = fig.colorbar(plt2, cax = cax2, label = labels[task], orientation='horizontal')
        cax2.xaxis.set_ticks_position('top')

        task = tasks[2]
        xdata = f[index][task]['x']
        zdata = f[index][task]['z']
        X, Z = np.meshgrid(xdata, zdata)
        C = f[index][task]['data_' + task]
        Clow, Chigh = get_bounds(C)
        center = matplotlib.colors.TwoSlopeNorm(vmin = Clow, vcenter = 0., vmax = Chigh)
        plt3 = ax3.pcolormesh(X.T, Z.T, C, cmap='PiYG_r', norm=center)
        ax3.set_ylabel(r'$z$', fontsize = 14)

        cbar3 = fig.colorbar(plt3, cax = cax3, label = labels[task], orientation='horizontal')
        cax3.xaxis.set_ticks_position('top')


        # Add time title
        title = title_func(f[index][task]['t'])
        title_height = 1 - 0.2 * t_mar
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
