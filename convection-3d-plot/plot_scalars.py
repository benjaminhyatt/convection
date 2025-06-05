import numpy as np
import matplotlib.pyplot as plt 
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

processed = np.load('../convection-3d-process/processed_scalars_' + output_suffix + '.npy', allow_pickle=True)[()]

# time to highlight with hline
#idx_hl = np.where(processed['t'] >= 1e3)[0][0]
idx_hl = -1

plt.figure()
plt.plot(processed['t'], processed['mean_Re'], color = 'blue')
plt.hlines(processed['mean_Re'][idx_hl], processed['t'][0], processed['t'][idx_hl], linestyle = "dotted", color = "purple", label = 'Re = {:.4e}'.format(processed['mean_Re'][idx_hl]))
plt.vlines(processed['t'][idx_hl], processed['mean_Re'][0], processed['mean_Re'][idx_hl], linestyle = "dotted", color = "purple")
plt.xlabel('Time')
plt.ylabel(r'$R|u|_{rms}$')
plt.legend()
plt.title("Re time series (max value = {:})".format(np.max(processed['mean_Re'])))
plt.tight_layout()
plt.savefig("mean_Re.pdf")

plt.figure()
plt.plot(processed['t'], processed['mean_T1_bot'], color = 'blue')
plt.hlines(processed['mean_T1_bot'][idx_hl], processed['t'][0], processed['t'][idx_hl], linestyle = "dotted", color = "purple", label = 'T1 = {:.4e}'.format(processed['mean_T1_bot'][idx_hl]))
plt.vlines(processed['t'][idx_hl], processed['mean_T1_bot'][0], processed['mean_T1_bot'][idx_hl], linestyle = "dotted", color = "purple")
plt.xlabel('Time')
plt.ylabel(r'$\overline{T_1(z=0)}$')
plt.legend()
plt.title(r"Temperature perturbation at $z=0$ time series (max value = {:})".format(np.max(processed['mean_T1_bot'])))
plt.tight_layout()
plt.savefig("mean_T1_bot.pdf")


plt.figure()
plt.plot(processed['t'], processed['mean_Re'], color = 'blue')
plt.hlines(processed['mean_Re'][idx_hl], processed['t'][0], processed['t'][idx_hl], linestyle = "dotted", color = "purple", label = 'Re = {:.4e}'.format(processed['mean_Re'][idx_hl]))
plt.vlines(processed['t'][idx_hl], processed['mean_Re'][0], processed['mean_Re'][idx_hl], linestyle = "dotted", color = "purple")
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel(r'$R|u|_{rms}$')
plt.legend()
plt.title("Re time series (max value = {:})".format(np.max(processed['mean_Re'])))
plt.tight_layout()
plt.savefig("mean_Re_log.pdf")

plt.figure()
plt.plot(processed['t'], processed['mean_T1_bot'], color = 'blue')
plt.hlines(processed['mean_T1_bot'], processed['t'][0], processed['t'][idx_hl], linestyle = "dotted", color = "purple", label = 'T1 = {:.4e}'.format(processed['mean_T1_bot'][idx_hl]))
plt.vlines(processed['t'][idx_hl], processed['mean_T1_bot'][0], processed['mean_T1_bot'][idx_hl], linestyle = "dotted", color = "purple")
plt.yscale('log')
plt.xlabel('Time')
plt.ylabel(r'$\overline{T_1(z=0)}$')
plt.legend()
plt.title(r"Temperature perturbation at $z=0$ time series (max value = {:})".format(np.max(processed['mean_T1_bot'])))
plt.tight_layout()
plt.savefig("mean_T1_bot_log.pdf")
