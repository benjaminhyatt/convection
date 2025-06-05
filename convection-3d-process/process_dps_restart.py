import numpy as np
import h5py
import dedalus.public as d3
import glob

mu = 1e-3
R = 1.6e3 #6.4e3
Ro = 2e-1 #2e-2
Nx, Ny, Nz = 128, 128, 512 
Lx, Ly, Lz = 4, 4, 2#1, 1, 2
dealias = 3/2 
dtype = np.float64
P = 1e0 #4e0 
S = 1e3
Pinv = 1/P
Qmag = 1e0

grad_ad = (Qmag * S * P) * (1 + mu + Pinv)
#grad_rad_top = (Qmag * S * P) * (1 + mu)
#delta_grad = grad_ad - grad_rad_top

dfs = [0.1, 0.5, 0.9]

output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) +     '_Nz_{:}'.format(Nz)
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz) + '_SF'
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

profiles_0_dir = 'profiles_0_' + output_suffix
profiles_0_file = glob.glob('../convection-3d-run/' + profiles_0_dir + '/*.h5')

print("reading profiles_0 file")
f0 = h5py.File(profiles_0_file[0], mode = 'r')
dset_Ftot = f0['tasks']['F_tot_bar_RHS']
dset_k = f0['tasks']['k']
# calculate delta_grad(z)
Ftot = np.copy(np.array(dset_Ftot[0]))[0, 0, :]
k = np.copy(np.array(dset_k[0]))[0, 0, :]
grad_rad = Ftot / k
delta_grad = grad_ad - grad_rad

grads = {}
tasks_out = []
labels_out = []
for df in dfs:
    task_out = 'df_{:.0e}'.format(df)
    grads[task_out] = 1 - df*delta_grad/grad_ad
    tasks_out.append('df_{:.0e}'.format(df))
    labels_out.append(r'$p = $' + str(df))

profiles_dir = 'profiles_' + output_suffix
profiles_file = glob.glob('../convection-3d-run/' + profiles_dir + '/*.h5')

idx1 = 0 
idx2 = 3 

print("reading first profiles file")
f1 = h5py.File(profiles_file[idx1], mode = 'r')
tasksf = list(f1['tasks'].keys())
nwritesf1 = f1['tasks'][tasksf[0]].shape[0]
dset_grad1 = f1['tasks']['grad']

print("reading second profiles file")
f2 = h5py.File(profiles_file[idx2], mode = 'r')
tasksf = list(f2['tasks'].keys())
nwritesf2 = f2['tasks'][tasksf[0]].shape[0]
dset_grad2 = f2['tasks']['grad']

t1 = np.array(dset_grad1.dims[0]['sim_time'])
t2 = np.array(dset_grad2.dims[0]['sim_time'])
nwritesf1 = np.where(t1 < t2[0])[0][-1]

processed = {}
times = {}
data = {}

for m, taskout in enumerate(tasks_out):
    processed[taskout] = {}    
    times[taskout] = []
    data[taskout] = []

    grad_df_vs_z = grads[taskout]

    for w in range(nwritesf1 + nwritesf2):
        if w < nwritesf1:
            use1 = True
        else:
            use1 = False
            w2 = w - nwritesf1
        
        if use1:
            times[taskout].append(np.array(dset_grad1.dims[0]['sim_time'])[w])
            grad_vs_z = np.copy(np.array(dset_grad1[w]))[0, 0, :]
            z = np.copy(np.array(dset_grad1.dims[3][0]))
        else:
            times[taskout].append(np.array(dset_grad2.dims[0]['sim_time'])[w2])
            grad_vs_z = np.copy(np.array(dset_grad2[w2]))[0, 0, :]
            z = np.copy(np.array(dset_grad2.dims[3][0]))
            
        cz_bools = grad_vs_z > grad_df_vs_z
        if np.sum(cz_bools) > 0:
            dp = np.max(z[cz_bools])
        else:
            dp = 0.
        data[taskout].append(dp)
        
    processed[taskout]['t'] = times[taskout]
    processed[taskout]['data'] = data[taskout]

f1.close()
f2.close()

processed['nout'] = nwritesf1 + nwritesf2
processed['tasks'] = tasks_out
processed['labels'] = labels_out
print('saving output')
np.save('processed_dps_' + output_suffix + '.npy', processed)

