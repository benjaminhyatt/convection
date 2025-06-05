import numpy as np
import h5py
import dedalus.public as d3
import glob

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

snapshots_dir = 'snapshots_' + output_suffix
profiles_dir = 'profiles_' + output_suffix

snapshots_file = sorted(glob.glob('../convection-3d-run/' + snapshots_dir + '/*.h5'))
profiles_file = sorted(glob.glob('../convection-3d-run/' + profiles_dir + '/*.h5'))

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

print(profiles_file)
print(snapshots_file)

idx1 = 0
idx2 = 3

print("reading first profiles file")
f1 = h5py.File(profiles_file[idx1], mode = 'r')
tasksf = list(f1['tasks'].keys())
nwritesf1 = f1['tasks'][tasksf[0]].shape[0]
dset_T_fluc_norm1 = f1['tasks'][tasks_in[0]]

print("reading first snapshots file")
fs1 = h5py.File(snapshots_file[idx1], mode = 'r')
dset_T_fluc1 = fs1['tasks'][tasks_in[1]]
dset_u_z1 = fs1['tasks'][tasks_in[2]]
dset_w_y1 = fs1['tasks'][tasks_in[3]]

print("reading second profiles file")
f2 = h5py.File(profiles_file[idx2], mode = 'r')
tasksf = list(f2['tasks'].keys())
nwritesf2 = f2['tasks'][tasksf[0]].shape[0]
dset_T_fluc_norm2 = f2['tasks'][tasks_in[0]]

print("reading second snapshots file")
fs2 = h5py.File(snapshots_file[idx2], mode = 'r')
dset_T_fluc2 = fs2['tasks'][tasks_in[1]]
dset_u_z2 = fs2['tasks'][tasks_in[2]]
dset_w_y2 = fs2['tasks'][tasks_in[3]]

print(nwritesf1, nwritesf2)
t1 = np.array(dset_T_fluc_norm1.dims[0]['sim_time'])
t2 = np.array(dset_T_fluc_norm2.dims[0]['sim_time'])
nwritesf1 = np.where(t1 < t2[0])[0][-1]
print(nwritesf1)

progress_cad = np.ceil((nwritesf1 + nwritesf2) / 50) 
for w in range(nwritesf1 + nwritesf2):
    if w < nwritesf1:
        use1 = True
    else:
        use1 = False
        w2 = w - nwritesf1

    processed[idx + w] = {}

    for taskout in tasks_out:
        processed[idx + w][taskout] = {}
     
        if taskout == 'T':
            if use1:
                processed[idx + w][taskout]['t'] = np.array(dset_T_fluc1.dims[0]['sim_time'])[w]
                processed[idx + w][taskout]['x'] = np.array(dset_T_fluc1.dims[1][0])
                processed[idx + w][taskout]['z'] = np.array(dset_T_fluc1.dims[3][0])
                data_num = np.copy(np.array(dset_T_fluc1[w]))
                data_den = np.sqrt(np.abs(np.copy(np.array(dset_T_fluc1[w]))))
                processed[idx + w][taskout]['data_T'] = (data_num / data_den).reshape((data_num.shape[0], data_num.shape[2]))
            else:
                processed[idx + w][taskout]['t'] = np.array(dset_T_fluc2.dims[0]['sim_time'])[w2]
                processed[idx + w][taskout]['x'] = np.array(dset_T_fluc2.dims[1][0])
                processed[idx + w][taskout]['z'] = np.array(dset_T_fluc2.dims[3][0])
                data_num = np.copy(np.array(dset_T_fluc2[w2]))
                data_den = np.sqrt(np.abs(np.copy(np.array(dset_T_fluc2[w2]))))
                processed[idx + w][taskout]['data_T'] = (data_num / data_den).reshape((data_num.shape[0], data_num.shape[2]))
                
        elif taskout == 'u_z': 
            if use1:
                processed[idx + w][taskout]['t'] = np.array(dset_u_z1.dims[0]['sim_time'])[w]
                processed[idx + w][taskout]['x'] = np.array(dset_u_z1.dims[1][0])
                processed[idx + w][taskout]['z'] = np.array(dset_u_z1.dims[3][0])
                data = np.copy(np.array(dset_u_z1[w]))
                processed[idx + w][taskout]['data_u_z'] = data.reshape((data.shape[0], data.shape[2]))
            else:
                processed[idx + w][taskout]['t'] = np.array(dset_u_z2.dims[0]['sim_time'])[w2]
                processed[idx + w][taskout]['x'] = np.array(dset_u_z2.dims[1][0])
                processed[idx + w][taskout]['z'] = np.array(dset_u_z2.dims[3][0])
                data = np.copy(np.array(dset_u_z2[w2]))
                processed[idx + w][taskout]['data_u_z'] = data.reshape((data.shape[0], data.shape[2]))
        elif taskout == 'w_y': 
            if use1:
                processed[idx + w][taskout]['t'] = np.array(dset_w_y1.dims[0]['sim_time'])[w]
                processed[idx + w][taskout]['x'] = np.array(dset_w_y1.dims[1][0])
                processed[idx + w][taskout]['z'] = np.array(dset_w_y1.dims[3][0])
                data = np.copy(np.array(dset_w_y1[w]))
                processed[idx + w][taskout]['data_w_y'] = data.reshape((data.shape[0], data.shape[2]))
            else:
                processed[idx + w][taskout]['t'] = np.array(dset_w_y2.dims[0]['sim_time'])[w2]
                processed[idx + w][taskout]['x'] = np.array(dset_w_y2.dims[1][0])
                processed[idx + w][taskout]['z'] = np.array(dset_w_y2.dims[3][0])
                data = np.copy(np.array(dset_w_y2[w2]))
                processed[idx + w][taskout]['data_w_y'] = data.reshape((data.shape[0], data.shape[2]))
        elif taskout == 'w_z':
            if use1:
                processed[idx + w][taskout]['t'] = np.array(dset_w_z1.dims[0]['sim_time'])[w]
                processed[idx + w][taskout]['x'] = np.array(dset_w_z1.dims[1][0])
                processed[idx + w][taskout]['y'] = np.array(dset_w_z1.dims[2][0])
                data = np.copy(np.array(dset_w_z1[w]))
                processed[idx + w][taskout]['data_w_z'] = data.reshape((data.shape[0], data.shape[1]))
            else:
                processed[idx + w][taskout]['t'] = np.array(dset_w_z2.dims[0]['sim_time'])[w2]
                processed[idx + w][taskout]['x'] = np.array(dset_w_z2.dims[1][0])
                processed[idx + w][taskout]['y'] = np.array(dset_w_z2.dims[2][0])
                data = np.copy(np.array(dset_w_z2[w2]))
                processed[idx + w][taskout]['data_w_z'] = data.reshape((data.shape[0], data.shape[1]))
    if w % progress_cad == 0:
        print("(%d / %d) writes processed" %(w + 1, nwritesf1 + nwritesf2))

f1.close()
fs1.close()
f2.close()
fs2.close()
processed['nout'] = nwritesf1 + nwritesf2
processed['tasks'] = tasks_out
processed['subtasks'] = subtasks_out
processed['labels'] = labels_out

np.save('processed_snapshots_' + output_suffix + '.npy', processed)




