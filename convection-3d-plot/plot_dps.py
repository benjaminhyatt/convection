import numpy as np
import matplotlib.pyplot as plt 
import h5py

mu = 1e-3
R = 6.4e3 #4e2#1.6e3#6.4e3
Ro = 2e-2 #np.inf #2e-2
Nx, Ny, Nz = 128, 128, 512 
Lx, Ly, Lz = 1, 1, 2 #1, 1, 2#4, 4, 2#1, 1, 2
dealias = 3/2 
dtype = np.float64
P = 4e0 #1e0#4e0 

#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.0e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz) + '_SF'
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

f = np.load('../convection-3d-process/processed_dps_' + output_suffix + '.npy', allow_pickle=True)[()]
nout = f['nout']
tasks = f['tasks']
labels = f['labels']

colors = ['blue', 'orange', 'green']

plt.figure()
for m, task in enumerate(tasks):
    xdata = f[task]['top']['t']
    ydata = f[task]['top']['data']
    plt.plot(xdata, ydata, linewidth = 1, color = colors[m], linestyle = "solid", label = labels[m] + ', ' + r'$\nabla_{ad}(L_z)$')
    xdata = f[task]['vs_z']['t']
    ydata = f[task]['vs_z']['data']
    plt.plot(xdata, ydata, linewidth = 1, color = colors[m], linestyle = "dotted", label = labels[m] + ', ' + r'$\nabla_{ad}(z)$')

plt.xlabel('Time')
plt.ylabel(r'$L_S + \delta_p$')
plt.legend()
plt.title("Departure points time series" + "\n" + r"(if no $\delta_p$ was found, a value of zero is plotted)" )
plt.tight_layout()
plt.savefig("dps.pdf")
