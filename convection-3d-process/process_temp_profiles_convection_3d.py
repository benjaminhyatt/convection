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

tasks_in = ['T0', 'T_bar', 'T1_bar', 'grad']
tasks_out = ['T_zoom', 'T', 'T1', 'grad']

subtasks_out = {}
subtasks_out['T_zoom'] = ['T0', 'T_bar']
subtasks_out['T'] = ['T0', 'T_bar']
subtasks_out['T1'] = ['T1_bar']
subtasks_out['grad'] = ['grad']
labels_out = {}
labels_out['T_zoom'] = [r'$T_0$', r'$\overline{T}$']
labels_out['T'] = [r'$T_0$', r'$\overline{T}$']
labels_out['T1'] = [r'$\overline{T_1}$']
labels_out['grad'] = [r'$d\langle T\rangle /dz / \nabla_{ad}$']

idx = 0 # don't remember why I had this

print("reading profiles_0 file")
f0 = h5py.File(profiles_0_file[0], mode = 'r')
dset_T0 = f0['tasks'][tasks_in[0]]

print("reading profiles file")
f = h5py.File(profiles_file[0], mode = 'r')
tasksf = list(f['tasks'].keys())
nwritesf = f['tasks'][tasksf[0]].shape[0]
dset_T_bar = f['tasks'][tasks_in[1]]
dset_T1_bar = f['tasks'][tasks_in[2]]
dset_grad = f['tasks'][tasks_in[3]]

zl, zr = (0., 0.15) # for (indicating the) zoomed in region
zin = np.array(dset_T_bar.dims[3][0])
idxl = np.where(zin > zl)[0][0]
idxr = np.where(zin < zr)[0][-1]

progress_cad = np.ceil(nwritesf / 50)
for w in range(nwritesf):
    processed[idx + w] = {}

    for taskout in tasks_out:
        processed[idx + w][taskout] = {}
        
        if taskout == 'grad':
            processed[idx + w][taskout]['t'] = np.array(dset_grad.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['z'] = np.array(dset_grad.dims[3][0])
            processed[idx + w][taskout]['data_grad'] = np.copy(np.array(dset_grad[w])) # has already been normalized
            processed[idx + w][taskout]['zoom_idxs'] = (idxl, idxr)
        elif taskout == 'T':
            processed[idx + w][taskout]['t'] = np.array(dset_T_bar.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['z'] = np.array(dset_T_bar.dims[3][0])
            processed[idx + w][taskout]['data_T0'] = np.copy(np.array(dset_T0[0]))
            processed[idx + w][taskout]['data_T_bar'] = np.copy(np.array(dset_T_bar[w]))
            processed[idx + w][taskout]['zoom_idxs'] = (idxl, idxr)
        elif taskout == 'T_zoom':
            processed[idx + w][taskout]['t'] = np.array(dset_T_bar.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['z'] = zin[idxl:idxr+1]
            processed[idx + w][taskout]['data_T0'] = np.copy(np.array(dset_T0[0]))[0, 0, idxl:idxr+1]
            processed[idx + w][taskout]['data_T_bar'] = np.copy(np.array(dset_T_bar[w]))[0, 0, idxl:idxr+1]
        elif taskout == 'T1':
            processed[idx + w][taskout]['t'] = np.array(dset_T1_bar.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['z'] = np.array(dset_T1_bar.dims[3][0])
            processed[idx + w][taskout]['data_T1_bar'] = np.copy(np.array(dset_T1_bar[w]))
            processed[idx + w][taskout]['zoom_idxs'] = (idxl, idxr)
    if w % progress_cad == 0:
        print("(%d / %d) writes processed" %(w + 1, nwritesf))

f0.close()
f.close()

processed['nout'] = nwritesf
processed['tasks'] = tasks_out
processed['subtasks'] = subtasks_out
processed['labels'] = labels_out

np.save('processed_temp_profiles_' + output_suffix + '.npy', processed)



