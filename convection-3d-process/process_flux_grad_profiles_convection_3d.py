import numpy as np
import h5py
import dedalus.public as d3
import glob

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

profiles_dir = 'profiles_' + output_suffix
profiles_0_dir = 'profiles_0_' + output_suffix

profiles_file = glob.glob('../convection-3d-run/' + profiles_dir + '/*.h5')
profiles_0_file = glob.glob('../convection-3d-run/' + profiles_0_dir + '/*.h5')

processed = {}

tasks_in = ['grad', 'F_rad_bar', 'F_conv_bar', 'F_tot_bar_LHS', 'F_tot_bar_RHS']
tasks_out = ['grad', 'fluxes']

subtasks_out = {}
subtasks_out['grad'] = ['grad']
subtasks_out['fluxes'] = ['Frad', 'Fconv', 'Fsum', 'Ftot']
labels_out = {}
labels_out['grad'] = [r'$d\langle T\rangle /dz / \nabla_{ad}$']
labels_out['fluxes'] = [r'$-k d\langle T\rangle /dz $, ($F_{rad}$)', r'$\langle u_z T \rangle $, ($F_{conv}$)', r'$F_{rad} + F_{conv}$', r'$F_{bot} + \int Q dz$, ($F_{tot}$)']

#Qmag = 1e0
#S = 1e3
#P = 1e0
#invP = 1/P
#mu = 1e-3
#grad_ad = (Qmag * S * P) * (1 + mu + invP)

idx = 0

print("reading profiles_0 file")
f0 = h5py.File(profiles_0_file[0], mode = 'r')
dset_Ftot = f0['tasks'][tasks_in[4]]

print("reading profiles file")
f = h5py.File(profiles_file[0], mode = 'r')
tasksf = list(f['tasks'].keys())
nwritesf = f['tasks'][tasksf[0]].shape[0]
   
dset_grad = f['tasks'][tasks_in[0]]
dset_Frad = f['tasks'][tasks_in[1]]
dset_Fconv = f['tasks'][tasks_in[2]]
dset_Fsum = f['tasks'][tasks_in[3]]

progress_cad = np.ceil(nwritesf / 50)
for w in range(nwritesf):
    processed[idx + w] = {}

    for taskout in tasks_out:
        processed[idx + w][taskout] = {}
        # grad task
        if taskout == 'grad':
            processed[idx + w][taskout]['t'] = np.array(dset_grad.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['z'] = np.array(dset_grad.dims[3][0])
            processed[idx + w][taskout]['data_grad'] = np.copy(np.array(dset_grad[w])) # has already been normalized
                
        # fluxes task
        elif taskout == 'fluxes':
            processed[idx + w][taskout]['t'] = np.array(dset_Frad.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['z'] = np.array(dset_Frad.dims[3][0])
            processed[idx + w][taskout]['data_Frad'] = np.copy(np.array(dset_Frad[w]))
            processed[idx + w][taskout]['data_Fconv'] = np.copy(np.array(dset_Fconv[w]))
            processed[idx + w][taskout]['data_Fsum'] = np.copy(np.array(dset_Fsum[w]))
            processed[idx + w][taskout]['data_Ftot'] = np.copy(np.array(dset_Ftot[0]))
    if w % progress_cad == 0:
        print("(%d / %d) writes processed" %(w + 1, nwritesf))

f0.close()
f.close()

processed['nout'] = nwritesf
processed['tasks'] = tasks_out
processed['subtasks'] = subtasks_out
processed['labels'] = labels_out

np.save('processed_flux_grad_profiles_' + output_suffix + '.npy', processed)



