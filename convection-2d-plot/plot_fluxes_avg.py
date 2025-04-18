"""
Time-average (over specified range) and plot from joint analysis files.

Usage:
    plot_fluxes_avg.py


"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from matplotlib import transforms
from dedalus.extras import plot_tools

# Load in time-varying profiles
profiles = h5py.File('profiles_mu_1e-03_R_4e02/profiles_mu_1e-03_R_4e02_s4.h5', 'r')
# Load in time-stationary profiles
profiles_0 = h5py.File('profiles_0_mu_1e-03_R_4e02/profiles_0_mu_1e-03_R_4e02_s1.h5', 'r')

# Tasks to plot
tasks = ['F_conv_bar', 'F_rad_bar', 'F_tot_bar_LHS']
tasks_0 = ['F_Q', 'F_tot_bar_RHS']

# Times to average over
out_cadence = 1e0 # in freefall times
#start_t = 2.99e4 # time at which this write began
#t_range = (3.935e4, 4.035e4)
#t_range = (3e4, 3.5e4)

start_t = 0
t_range = (1e2, 5e3)

start_idx = int((t_range[0]-start_t)/out_cadence)
end_idx = int((t_range[1]-start_t)/out_cadence)

# Plot options
labels = {}
labels_0 = {}
labels['F_conv_bar'] = r'$\langle u_z T\rangle$'
labels['F_rad_bar'] = r'$-k \frac{d\langle T\rangle}{dz}$'
labels['F_tot_bar_LHS'] = r'$\langle u_z T \rangle -k \frac{d\langle T\rangle}{dz}$'
labels_0['F_tot_bar_RHS'] = r'$\int (Q) + F_{bot}$'
labels_0['F_Q'] = r'$\int (Q)$'
linestyles = {}
linestyles_0 = {}
linestyles['F_conv_bar'] = 'dashed'
linestyles['F_rad_bar'] = 'dashed'
linestyles['F_tot_bar_LHS'] = 'dashed'
linestyles_0['F_tot_bar_RHS'] = 'solid'
linestyles_0['F_Q'] = 'solid'
colors = {}
colors_0 = {}
colors['F_conv_bar'] = 'red'
colors['F_rad_bar'] = 'blue'
colors['F_tot_bar_LHS'] = 'black'
colors_0['F_tot_bar_RHS'] = 'purple'
colors_0['F_Q'] = 'green'

scale = 1.5
dpi = 200

# Layout
nrows, ncols = 1, 1
image = plot_tools.Box(2, 1)
pad = plot_tools.Frame(0.3, 0, 0, 0)
margin = plot_tools.Frame(0.2, 0.1, 0, 0)

# Create multifigure
mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
fig = mfig.figure

axes = mfig.add_axes(0, 0, [0, 0, 1, 1]) 
pbbox = transforms.Bbox.from_bounds(0.1, 0.2, 0.65, 0.85)
to_axes_bbox = transforms.BboxTransformTo(axes.get_position())
pbbox = pbbox.transformed(to_axes_bbox)
paxes = axes.figure.add_axes(pbbox)
axes.axis("off")

# Plot writes
for n, task in enumerate(tasks):
    dset = profiles['tasks'][task]   
    xdata = np.array(dset.dims[2][0])
    ydata = np.mean(dset[start_idx:end_idx, 0, :], axis = 0)
    paxes.plot(xdata, ydata, linewidth = 1, color = colors[task], linestyle = linestyles[task], label = labels[task])
for n, task in enumerate(tasks_0):
    dset = profiles_0['tasks'][task]   
    xdata = np.array(dset.dims[2][0])
    ydata = dset[0, 0, :]
    paxes.plot(xdata, ydata, linewidth = 1, color = colors_0[task], linestyle = linestyles_0[task], label = labels_0[task])

paxes.set_xlabel(dset.dims[2].label)
paxes.legend(loc = (1.05, 0.05), fontsize = 6)
paxes.set_ylim(-5e-1, 5e-1)
title = 'Fluxes time averaged from \n' + 't = %e to %e' %(t_range[0], t_range[1])
title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
fig.suptitle(title, x=0.04, y=title_height, ha='left')
# Save figure
savepath = 'fluxes_t_avg_very_early.png'
fig.savefig(savepath, dpi=dpi)
print("Saved", savepath)
fig.clear()
plt.close(fig)
