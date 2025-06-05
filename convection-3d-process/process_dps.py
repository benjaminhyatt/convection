import numpy as np
import h5py
import dedalus.public as d3
import glob

mu = 1e-3
R = 6.4e3 #4e2 #1.6e3 #6.4e3
Ro = 2e-2 #np.inf #2e-2
Nx, Ny, Nz = 128, 128, 512 
Lx, Ly, Lz = 1, 1, 2#4, 4, 2#1, 1, 2
dealias = 3/2 
dtype = np.float64
P = 4e0 #1e0 #4e0 
S = 1e3
Pinv = 1/P
Qmag = 1e0

grad_ad = (Qmag * S * P) * (1 + mu + Pinv) # positive
grad_rad_top = (Qmag * S * P) * (1 + mu)
delta_grad_top = 1 - grad_rad_top/grad_ad

#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.0e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
#output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.1e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_P_{:.0e}'.format(P) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz) + '_SF'
output_suffix = output_suffix.replace('-','m').replace('+','p').replace('.','d')

profiles_0_dir = 'profiles_0_' + output_suffix
profiles_dir = 'profiles_' + output_suffix
profiles_0_file = glob.glob('../convection-3d-run/' + profiles_0_dir + '/*.h5')
profiles_file = glob.glob('../convection-3d-run/' + profiles_dir + '/*.h5')

print("reading profiles_0 file")
f0 = h5py.File(profiles_0_file[0], mode = 'r')
dset_Ftot = f0['tasks']['F_tot_bar_RHS']
dset_k = f0['tasks']['k']
# calculate delta_grad(z)
Ftot = np.copy(np.array(dset_Ftot[0]))[0, 0, :]
k = np.copy(np.array(dset_k[0]))[0, 0, :]
grad_rad_vs_z = Ftot / k # positive 
delta_grad_vs_z = 1 - grad_rad_vs_z/grad_ad # positive for z sufficiently greater than L_S, but sign does switch (see z_match in IVP script)

dfs = [0.1, 0.5, 0.9]

grads_top = {}
grads_vs_z = {}
tasks_out = []
labels_out = []
for df in dfs:
    task_out = 'df_{:.0e}'.format(df)
    grads_top[task_out] = 1 - df*delta_grad_top
    grads_vs_z[task_out] = 1 - df*delta_grad_vs_z
    tasks_out.append('df_{:.0e}'.format(df))
    labels_out.append(r'$p = $' + str(df))

print("reading profiles file")
f = h5py.File(profiles_file[0], mode = 'r')
tasksf = list(f['tasks'].keys())
nwritesf = f['tasks'][tasksf[0]].shape[0]
dset_grad = f['tasks']['grad']   

processed = {}
times = {}
data = {}

for m, taskout in enumerate(tasks_out):
    processed[taskout] = {}
    processed[taskout]['top'] = {}
    processed[taskout]['vs_z'] = {}

    times[taskout] = {}
    times[taskout]['top'] = []
    times[taskout]['vs_z'] = []
    
    data[taskout] = {}
    data[taskout]['top'] = []
    data[taskout]['vs_z'] = []

    grad_df_top = grads_top[taskout]
    grad_df_vs_z = grads_vs_z[taskout]

    for w in range(nwritesf):
        times[taskout]['top'].append(np.array(dset_grad.dims[0]['sim_time'])[w])
        times[taskout]['vs_z'].append(np.array(dset_grad.dims[0]['sim_time'])[w])
        
        grad_vs_z = np.copy(np.array(dset_grad[w]))[0, 0, :]
        z = np.copy(np.array(dset_grad.dims[3][0])) 

        # grad_rad_top
        cz_bools_top = grad_vs_z > grad_df_top
        if np.sum(cz_bools_top) > 0:
            dp_top = np.max(z[cz_bools_top])
        else:
            dp_top = 0.

        # grad_rad(z)
        cz_bools = grad_vs_z > grad_df_vs_z
        if np.sum(cz_bools) > 0:
            dp = np.max(z[cz_bools])
        else:
            dp = 0.
        
        data[taskout]['top'].append(dp_top)
        data[taskout]['vs_z'].append(dp)

    processed[taskout]['top']['t'] = times[taskout]['top']
    processed[taskout]['top']['data'] = data[taskout]['top']
    processed[taskout]['vs_z']['t'] = times[taskout]['vs_z']
    processed[taskout]['vs_z']['data'] = data[taskout]['vs_z']


for i, z_i in enumerate(z):
    # normalized by grad_ad
    print(i, "z", z_i, "grad(z)", grad_vs_z[i], "grad_rad(z)", grad_rad_vs_z[i]/grad_ad)

f.close()
f0.close()

processed['nout'] = nwritesf
processed['tasks'] = tasks_out
processed['labels'] = labels_out
print('saving output')
np.save('processed_dps_' + output_suffix + '.npy', processed)

