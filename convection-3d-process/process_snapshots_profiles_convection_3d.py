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

snapshots_dir = 'snapshots_' + output_suffix
profiles_dir = 'profiles_' + output_suffix
snapshots_file = glob.glob('../convection-3d-run/' + snapshots_dir + '/*.h5')
profiles_file = glob.glob('../convection-3d-run/' + profiles_dir + '/*.h5')
processed = {}

tasks_in = ['T_fluc_norm', 'T_fluc', 'u_z', 'w_y']
tasks_out = ['T', 'u_z', 'w_y'] 
#tasks_in = ['T_fluc_norm', 'T_fluc', 'u_z', 'w_z']
#tasks_out = ['T', 'u_z', 'w_z']

subtasks_out = {}
subtasks_out['T'] = ['T_fluc_norm', 'T_fluc']
subtasks_out['u_z'] = ['u_z']
subtasks_out['w_y'] = ['w_y']
#subtasks_out['w_z'] = ['w_z']
labels_out = {}
labels_out['T'] = r'$\left(T^{\prime} / \overline{|T^{\prime}|}\right)(y = L_y/2)$'
labels_out['u_z'] = r"$u_z(y = L_y/2)$"
labels_out['w_y'] = r"$\omega_y(y = L_y/2)$"
#labels_out['w_z'] = r"$\omega_z(z = L_z/4)$"

idx = 0

print("reading profiles file")
f = h5py.File(profiles_file[0], mode = 'r')
tasksf = list(f['tasks'].keys())
nwritesf = f['tasks'][tasksf[0]].shape[0]
dset_T_fluc_norm = f['tasks'][tasks_in[0]]

print("reading snapshots file")
fs = h5py.File(snapshots_file[0], mode = 'r')
dset_T_fluc = fs['tasks'][tasks_in[1]]
dset_u_z = fs['tasks'][tasks_in[2]]
dset_w_y = fs['tasks'][tasks_in[3]]
#dset_w_z = fs['tasks'][tasks_in[3]]

progress_cad = np.ceil(nwritesf / 50)
for w in range(nwritesf):
    processed[idx + w] = {}

    for taskout in tasks_out:
        processed[idx + w][taskout] = {}
     
        if taskout == 'T':
            processed[idx + w][taskout]['t'] = np.array(dset_T_fluc.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['x'] = np.array(dset_T_fluc.dims[1][0])
            processed[idx + w][taskout]['z'] = np.array(dset_T_fluc.dims[3][0])
            data_num = np.copy(np.array(dset_T_fluc[w]))
            #data_den = np.copy(np.array(dset_T_fluc_norm[w]))
            # previously was taking sqrt but getting some nans due to floating point errs (?) in last
            # grid point near z=Lz
            # for now, going to take abs val 
            data_den = np.sqrt(np.abs(np.copy(np.array(dset_T_fluc[w]))))
            processed[idx + w][taskout]['data_T'] = (data_num / data_den).reshape((data_num.shape[0], data_num.shape[2]))
        elif taskout == 'u_z': 
            processed[idx + w][taskout]['t'] = np.array(dset_u_z.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['x'] = np.array(dset_u_z.dims[1][0])
            processed[idx + w][taskout]['z'] = np.array(dset_u_z.dims[3][0])
            data = np.copy(np.array(dset_u_z[w]))
            #processed[idx + w][taskout]['data_u_z'] = (data / np.max(data)).reshape((data.shape[0], data.shape[2]))
            processed[idx + w][taskout]['data_u_z'] = data.reshape((data.shape[0], data.shape[2]))
        elif taskout == 'w_y': 
            processed[idx + w][taskout]['t'] = np.array(dset_w_y.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['x'] = np.array(dset_w_y.dims[1][0])
            processed[idx + w][taskout]['z'] = np.array(dset_w_y.dims[3][0])
            data = np.copy(np.array(dset_w_y[w]))
            #processed[idx + w][taskout]['data_w_y'] = (data / np.max(data)).reshape((data.shape[0], data.shape[2]))
            processed[idx + w][taskout]['data_w_y'] = data.reshape((data.shape[0], data.shape[2]))
        
        elif taskout == 'w_z':
            processed[idx + w][taskout]['t'] = np.array(dset_w_z.dims[0]['sim_time'])[w]
            processed[idx + w][taskout]['x'] = np.array(dset_w_z.dims[1][0])
            processed[idx + w][taskout]['y'] = np.array(dset_w_z.dims[2][0])
            data = np.copy(np.array(dset_w_z[w]))
            #processed[idx + w][taskout]['data_w_z'] = (data / np.max(data)).reshape((data.shape[0], data.shape[1]))
            processed[idx + w][taskout]['data_w_z'] = data.reshape((data.shape[0], data.shape[1]))

    if w % progress_cad == 0:
        print("(%d / %d) writes processed" %(w + 1, nwritesf))

f.close()
fs.close()

processed['nout'] = nwritesf
processed['tasks'] = tasks_out
processed['subtasks'] = subtasks_out
processed['labels'] = labels_out

np.save('processed_snapshots_' + output_suffix + '.npy', processed)




