"""
Plot profiles from joint analysis files.

Usage:
    plot_profiles.py <files>... [--output=<dir>]

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


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    #tasks = ['F_conv_bar', 'F_rad_bar', 'F_tot_bar_LHS', 'F_Q', 'F_tot_bar_0_RHS']
    tasks = ['F_conv_bar', 'F_rad_bar', 'F_tot_bar_LHS']
    labels = {}
    labels['F_conv_bar'] = r'$\langle u_z T\rangle$'
    labels['F_rad_bar'] = r'$-k \frac{d\langle T\rangle}{dz}$'
    labels['F_tot_bar_LHS'] = r'$\langle u_z T \rangle -k \frac{d\langle T\rangle}{dz}$'
    labels['F_tot_bar_0_RHS'] = r'$\int (Q) + F_{bot}$'
    labels['F_Q'] = r'$\int (Q)$'
    linestyles = {}
    linestyles['F_conv_bar'] = 'solid'
    linestyles['F_rad_bar'] = 'solid'
    linestyles['F_Q'] = 'solid'
    linestyles['F_tot_bar_LHS'] = 'dashed'
    linestyles['F_tot_bar_0_RHS'] = 'dashed'
    colors = {}
    colors['F_conv_bar'] = 'red'
    colors['F_rad_bar'] = 'blue'
    colors['F_Q'] = 'green'
    colors['F_tot_bar_LHS'] = 'black'
    colors['F_tot_bar_0_RHS'] = 'purple'

    scale = 1.5
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    # Layout
    nrows, ncols = 1, 1
    image = plot_tools.Box(2, 1)
    pad = plot_tools.Frame(0.3, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            print(index)
            axes = mfig.add_axes(0, 0, [0, 0, 1, 1])
            #pbbox = transforms.Bbox.from_bounds(0.2, 0.2, 0.8, 0.8)
            pbbox = transforms.Bbox.from_bounds(0.1, 0.2, 0.9, 0.95)
            to_axes_bbox = transforms.BboxTransformTo(axes.get_position())
            pbbox = pbbox.transformed(to_axes_bbox)
            paxes = axes.figure.add_axes(pbbox)
            axes.axis("off")

            for n, task in enumerate(tasks):
                dset = file['tasks'][task]
                xdata = np.array(dset.dims[2][0])
                ydata = dset[index, 0, :]
                paxes.plot(xdata, ydata, linewidth = 1, color = colors[task], linestyle = linestyles[task], label = labels[task])
            
            paxes.set_xlabel(dset.dims[2].label)
            paxes.legend(loc = 'upper right')
            paxes.set_ylim(-1e0, 1e0)


            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.44, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)






if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

