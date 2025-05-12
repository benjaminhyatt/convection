import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import h5py as h5

## load in profiles_0 file
f = h5.File('../convection-3d-run/profiles_0_mu_1em03_R_4ep02_Ro_1ep00_Nx_128_Ny_128_Nz_512/profiles_0_mu_1em03_R_4ep02_Ro_1ep00_Nx_128_Ny_128_Nz_512_s1.h5')

## extract profiles
k = f['tasks/k'][0, 0, 0, :]
T0 = f['tasks/T0'][0, 0, 0, :]
Q = f['tasks/heating'][0, 0, 0, :]
cooling = f['tasks/cooling'][0, 0, 0, :]
F_Q = f['tasks/F_Q'][0, 0, 0, :]
F_tot_bar_RHS = f['tasks/F_tot_bar_RHS'][0, 0, 0, :]

z = np.array(f['tasks/k'].dims[3][0])

## inspect with plots (as needed - can also use print statements, etc.) 
index = 0

plt.figure()
plt.plot(z, k)
plt.savefig('k.pdf')

plt.figure()
plt.plot(z, T0)
plt.savefig('T0.pdf')

plt.figure()
plt.plot(z, Q)
plt.savefig('Q.pdf')

plt.figure()
plt.plot(z, cooling)
plt.savefig('cooling.pdf')

plt.figure()
plt.plot(z, F_Q)
plt.savefig('F_Q.pdf')

plt.figure()
plt.plot(z, F_tot_bar_RHS)
plt.savefig('F_tot_bar_RHS.pdf')
