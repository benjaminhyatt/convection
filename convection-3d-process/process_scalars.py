import numpy as np
import h5py

mu = 1e-3
R = 3.2e3#6.4e3
Ro = 2e-2
Nx, Ny, Nz = 128, 128, 512 
Lx, Ly, Lz = 2, 2, 2#1, 1, 2
dealias = 3/2 
dtype = np.float64
P = 4e0 

#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.0e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = output_suffix.replace('-','m').replace('+','p')
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')
output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz) + '_SF'
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')


# load in analysis
print("loading primary run")
f1 = h5py.File('../convection-3d-run/scalars_' + output_suffix + '/scalars_' + output_suffix + '_s1.h5')

t = f1['tasks/mean_Re'].dims[0]['sim_time'][:]
mean_Re = f1['tasks/mean_Re'][:, 0, 0, 0] # change: this is currently Average(|u|^2)
mean_Re = R * np.sqrt(mean_Re)
#mean_Re = f1['tasks/mean_Re'][:, 0, 0, 0]
#mean_Re /= R
#mean_Re = R * np.sqrt(mean_Re)
mean_T1_bot = f1['tasks/mean_T1_bot'][:, 0, 0, 0]

processed = {}
processed['t'] = t
processed['mean_Re'] = mean_Re
processed['mean_T1_bot'] = mean_T1_bot

print('saving output')
np.save('processed_scalars_' + output_suffix + '.npy', processed)
