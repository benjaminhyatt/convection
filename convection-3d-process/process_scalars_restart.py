import numpy as np
import h5py

mu = 1e-3
R = 6.4e3
Ro = 2e-2
Nx, Ny, Nz = 128, 128, 512 
Lx, Ly, Lz = 1, 1, 2
dealias = 3/2 
dtype = np.float64
P = 4e0 

#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.0e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = output_suffix.replace('-','m').replace('+','p')
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')
output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

# load in analysis
print("loading first run")
f1 = h5py.File('../convection-3d-run/scalars_' + output_suffix + '/scalars_' + output_suffix + '_s1.h5')

t1 = f1['tasks/mean_Re'].dims[0]['sim_time'][:]
mean_Re1 = f1['tasks/mean_Re'][:, 0, 0, 0]
mean_T1_bot1 = f1['tasks/mean_T1_bot'][:, 0, 0, 0]

print("loading run after restart")
#f2 = h5py.File('../convection-3d-run/scalars_' + output_suffix + '/scalars_' + output_suffix + '_s2.h5')
f2 = h5py.File('../convection-3d-run/scalars_' + output_suffix + '/scalars_' + output_suffix + '_s4.h5')
t2 = f2['tasks/mean_Re'].dims[0]['sim_time'][:]
mean_Re2 = f2['tasks/mean_Re'][:, 0, 0, 0]
mean_T1_bot2 = f2['tasks/mean_T1_bot'][:, 0, 0, 0]

prior = np.where(t1 < t2[0])[0][-1]
tprior = t1[prior]

t = np.concatenate((t1[:prior+1], t2))
mean_Re = np.concatenate((mean_Re1[:prior+1], mean_Re2))
mean_T1_bot = np.concatenate((mean_T1_bot1[:prior+1], mean_T1_bot2))

processed = {}
processed['t'] = t
processed['mean_Re'] = mean_Re
processed['mean_T1_bot'] = mean_T1_bot

print('saving output')
np.save('processed_scalars_' + output_suffix + '.npy', processed)
