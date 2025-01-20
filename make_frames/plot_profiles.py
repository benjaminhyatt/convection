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
    #tasks = ['heating', 'k_zT0_z', 'kT0_zz', 'kprof']
    
    #tasks = ['T_rad_z0prof', 'T0', 'T0_z', 'T0_zz']
    #tasks = ['heating', 'Qflux', 'kprof', 'T_rad_z0prof']
    
    tasks = ['heating', 'cooling', 'fHprof', 'fH2prof']

    #tasks = ['T0_zz_init', 'T0_zz_H', 'T0_zz_del', 'T_rad_z0prof'] # will only find inside of profiles_dz2T0_256 (or 512)

    #tasks = ['Qfluxk2k_z', 'k_zprof', 'k1', 'Qzprof']
    
    titles = {}
    titles['heating'] = r'$Q(z)$'
    titles['cooling'] = r'$\partial_z k \partial_z T_0 + k \partial_z^2 T_0$'
    titles['fHprof'] = r'$Q + \partial_z k \partial_z T_0 + k \partial_z^2 T_0$'
    titles['fH2prof'] = r'$\int ( Q + \partial_z k \partial_z T_0 + k \partial_z^2 T_0 ) dz$'
    titles['Qflux'] = r'$\int Q dz$'
    titles['kprof'] = r'$k(z)$'
    titles['T_rad_z0prof'] = r'$(\int Q dz)/k(z)$'
    titles['T0'] = r'$T_0$'
    titles['T0_z'] = r'$\partial_z T_0$'
    titles['T0_zz'] = r'$\partial_z^2 T_0$'
    titles['k_zT0_z'] = r'$\partial_z k \partial_z T_0$'
    titles['kT0_zz'] = r'$k \partial_z^2 T_0$'
    titles['T0_z-T_ad'] = r'$\partial_z T_0 - T_{ad}$'

    titles['T0_zz_init'] = r'$\partial_z^2 T_0$ init'
    titles['T0_zz_H'] = r'$\partial_z^2 T_0 \times H$'
    titles['T0_zz_del'] = r'$\partial_z^2 T_0 \times \frac{\delta_{rad}}{\delta_{sim}}$'


    titles['Qfluxk2k_z'] = r'$-\frac{\int Q dz}{k^2}\frac{dk}{dz}$'
    titles['Qfluxk2'] = r'$-\frac{\int Q dz}{k^2}$'
    titles['k_zprof'] = r'$\frac{dk}{dz}$'
    titles['Qzprof'] = r'$\frac{Q}{k}$'
    titles['k1'] = r'$k^{-1}$'

    scale = 1.5
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    # Layout
    nrows, ncols = 2, 2
    image = plot_tools.Box(2, 1)
    pad = plot_tools.Frame(0.3, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]

                print(index, n, task, dset.shape)
                pbbox = transforms.Bbox.from_bounds(0.2, 0.2, 0.8, 0.8)
                to_axes_bbox = transforms.BboxTransformTo(axes.get_position())
                pbbox = pbbox.transformed(to_axes_bbox)
                paxes = axes.figure.add_axes(pbbox)
                axes.axis("off")

                xdata = np.array(dset.dims[2][0])
                ydata = dset[index, 0, :]

                paxes.plot(xdata, ydata, color = "black", linewidth = 1)
                #paxes.plot(xdata, np.abs(ydata), color = "black", linewidth = 1)
                #paxes.set_yscale("log")
                #paxes.set_ylim(2.0009995e3, 2.00100005e3)
                #paxes.set_ylim(2.001999999e-1, 2.002000001e-1)
                #paxes.set_xlim(1.25, 1.3)
                #paxes.set_ylim(-2.00200051e10, -2.00200049e10)
                #paxes.set_ylim(-5.008e6, -5.004e6)
                #paxes.set_ylim(4e6, 6e6)
                
                #paxes.set_ylim(-3e-7, 3e-7)
                #paxes.set_ylim(-1e-14, 1e-14)
                #paxes.set_ylim(1.0004999999e7, 1.0005000001e7)
                #paxes.set_ylim(4.999999e3, 5.000001e3)
                paxes.set_title(titles[task])
                paxes.set_xlabel(dset.dims[2].label)

                #plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=titles[task], even_scale=True, visible_axes=False)

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

