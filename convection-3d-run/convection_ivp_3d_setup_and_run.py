"""
Dedalus (d3) script to setup and run a 3d horizontally-periodic simulation 
with convection and stable stratification in the Boussinesq limit,
following the non-dimensionalization given in Anders et al. (2022). 
This script implements a discontinuous (erf) convective flux profile. 

First run convection_ivp_3d_setup.py, which this script will pull its
initial conditions from. (This is required when starting from a checkpoint
as well, since the problem has nccs/terms which are not stored in solver.state)  
"""
import numpy as np
import h5py
import dedalus.public as d3
from mpi4py import MPI 
import logging
logger = logging.getLogger(__name__)
from scipy.special import erf 
from dedalus.extras import flow_tools

logger.info("Script began")

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#mesh_x, mesh_y = (16, 8)
mesh_x, mesh_y = (32, 16)
#mesh_x, mesh_y = (16, 16)
if int(mesh_x * mesh_y) != size:
    logger.info("Warning: choice of mesh shape may be bad for the number of processes this job has. This may result in an error, or poor performance.")

logger.info("rank %d began, total of %d processes" %(rank, size))
logger.info("plan is to use a mesh of shape (%d, %d)" %(mesh_x, mesh_y))

##### Parameters #####

# Prognostic
P = 1e0 # Penetration parameter
S = 1e3 # Stiffness
Pr = 5e-1 # Prandtl 
R = 4e2 # Reynolds (freefall)
mu = 1e-3 # F_H/F_bot
Ro = np.inf #1e-1 #np.inf # Rossby (Ro = np.inf will result in non-rotating case)

Pinv = 1/P 
Rinv = 1/R 
PrRinv = 1/(Pr*R)
if Ro != np.inf:
    rot = True
    Roinv = 1/Ro
else:
    rot = False
    Roinv = 0


# Simulation
dtype = np.float64
Lx, Ly, Lz = 4, 4, 2
Nx, Ny, Nz = 128, 128, 512
#Nx, Ny, Nz = 64, 64, 256
dealias = 3/2 
#timestepper = d3.RK443
#safety = 0.35 # CFL prefactor
timestepper = d3.SBDF2
safety = 0.3
stop_sim_time = 1e2 # freefall times

output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.0e}'.format(R) + '_Ro_{:.0e}'.format(Ro) + '_Nx_{:}'.format(Nx) + '_Ny_{:}'.format(Ny) + '_Nz_{:}'.format(Nz)
output_suffix = output_suffix.replace('-','m').replace('+','p')

an_cad = 0.25 # analysis cadence in freefall times
cp_cad = 100 # checkpoint cadence in freefall times

# Initialization
restart = False
if restart:
    cp_idx = 300
    start_dir = 'checkpoints_' + output_suffix + '/checkpoints_' + output_suffix + '_s' + str(cp_idx) + '.h5'

# Model
L_Q = 1e-1 
dH = 2e-1 # width of internal heating layer: z in [L_Q, L_Q + dH]
Qmag = 1e0 # magnitude of Q inside heating layer
dwQ = 2e-2 # scale of Q jump
L_CZ = 1e0 # where to center jump in flux
dwk = 7.5e-2 # scale of k jump
dwT = 5e-2 # scale of T0_zz jump
F_bot = mu * Qmag * dH # radiative flux through z = 0

logger.info("Defining bases, coordinates, and fields")

### Bases ###
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype = dtype)
xbasis = d3.RealFourier(coords['x'], size = Nx, bounds = (0, Lx), dealias = dealias)
ybasis = d3.RealFourier(coords['y'], size = Ny, bounds = (0, Ly), dealias = dealias)
zbasis = d3.ChebyshevT(coords['z'], size = Nz, bounds = (0, Lz), dealias = dealias)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
x_de, y_de, z_de = dist.local_grids(xbasis, ybasis, zbasis, scales=(dealias, dealias, dealias))
ex, ey, ez = coords.unit_vector_fields(dist)
reducer = flow_tools.GlobalArrayReducer(dist.comm_cart)

### Fields and substitutions ###

# Time-varying
ux = dist.Field(name = 'ux', bases = (xbasis, ybasis, zbasis))
uy = dist.Field(name = 'uy', bases = (xbasis, ybasis, zbasis))
uz = dist.Field(name = 'uz', bases = (xbasis, ybasis, zbasis))
wx = dist.Field(name = 'wx', bases = (xbasis, ybasis, zbasis))
wy = dist.Field(name = 'wy', bases = (xbasis, ybasis, zbasis))
wz = dist.Field(name = 'wz', bases = (xbasis, ybasis, zbasis))
T1 = dist.Field(name = 'T1', bases = (xbasis, ybasis, zbasis))
T1_z = dist.Field(name = 'T1_z', bases = (xbasis, ybasis, zbasis))
p = dist.Field(name = 'p', bases = (xbasis, ybasis, zbasis))

# Tau
tau_ux1 = dist.Field(name = 'tau_ux1', bases = (xbasis, ybasis))
tau_ux2 = dist.Field(name = 'tau_ux2', bases = (xbasis, ybasis))
tau_uy1 = dist.Field(name = 'tau_uy1', bases = (xbasis, ybasis))
tau_uy2 = dist.Field(name = 'tau_uy2', bases = (xbasis, ybasis))
tau_uz1 = dist.Field(name = 'tau_uz1', bases = (xbasis, ybasis))
tau_uz2 = dist.Field(name = 'tau_uz2', bases = (xbasis, ybasis))
tau_T11 = dist.Field(name = 'tau_T11', bases = (xbasis, ybasis))
tau_T12 = dist.Field(name = 'tau_T12', bases = (xbasis, ybasis))
tau_p = dist.Field(name = 'tau_p')

# Stationary

# Appear explicitly in problem
H = dist.Field(name = 'H', bases = zbasis)
k = dist.Field(name = 'k', bases = zbasis)
Ups = dist.Field(name = 'Ups', bases = zbasis)

# Otherwise in setup and/or analysis
flux_of_z = dist.Field(name = 'flux_of_z', bases = zbasis)
k_z = dist.Field(name = 'k_z', bases = zbasis)
Q = dist.Field(name = 'Q', bases = zbasis)
T0 = dist.Field(name = 'T0', bases = zbasis)
T0_z = dist.Field(name = 'T0_z', bases = zbasis)
T0_zz = dist.Field(name = 'T0_zz', bases = zbasis)
T_ad_z = dist.Field(name = 'T_ad_z', bases = zbasis)
T_rad_z0 = dist.Field(name = 'T_rad_z0', bases = zbasis)

fH = dist.Field(name = 'fH', bases = zbasis)
fH2 = dist.Field(name = 'fH2', bases = zbasis)

# For setting up fields on rank 0
if rank == 0:
    dist_0 = d3.Distributor(coords['z'], comm = MPI.COMM_SELF, dtype = dtype)
    zbasis_0 = d3.ChebyshevT(coords['z'], size = Nz, bounds = (0, Lz), dealias = dealias)
    z_0 = dist_0.local_grid(zbasis_0)
    z_de_0 = dist_0.local_grid(zbasis_0, scale = dealias)

    k_0 = dist_0.Field(name = 'k_0', bases = zbasis_0)
    flux_of_z_0 = dist_0.Field(name = 'flux_of_z_0', bases = zbasis_0)
    k_z_0 = dist_0.Field(name = 'k_z_0', bases = zbasis_0)
    Q_0 = dist_0.Field(name = 'Q_0', bases = zbasis_0)
    T0_0 = dist_0.Field(name = 'T0_0', bases = zbasis_0)
    T0_z_0 = dist_0.Field(name = 'T0_z_0', bases = zbasis_0)
    T0_zz_0 = dist_0.Field(name = 'T0_zz_0', bases = zbasis_0)
    T_ad_z_0 = dist_0.Field(name = 'T_ad_z_0', bases = zbasis_0)
    T_rad_z0_0 = dist_0.Field(name = 'T_rad_z0_0', bases = zbasis_0)
    fH_0 = dist_0.Field(name = 'fH_0', bases = zbasis_0)
    fH2_0 = dist_0.Field(name = 'fH2_0', bases = zbasis_0)

# More model definitions
k_RZ = dH / P / S 
k_CZ = k_RZ * mu / (1 + mu + Pinv)
k_ad = k_RZ * (1 + mu) / (1 + mu + Pinv)
dk = k_RZ - k_CZ

grad_ad = (Qmag * S * P) * (1 + mu + Pinv)
grad_rad_top = (Qmag * S * P) * (1 + mu)

##### Function definitions #####

def step_func(z, z0, dw): # 0 to 1
    return (1 + erf((z - z0)/dw))/2

def step_func_c(z, z0, dw): # 1 to 0
    return (1 - erf((z - z0)/dw))/2

def Q_func(z, z0, dw, Qmag, dH):
    return Qmag * step_func(z, z0, dw) * step_func_c(z, z0 + dH, dw) 

def k_func(z, z0, dw, k_CZ, dk):
    return k_CZ + dk * step_func(z, z0, dw) 

##### Beginning of field setup #####

# Define k = k(z) and k_z = dz(k)
# Integrate to determine vertical flux due to Q and F_bot
if rank == 0:
    k_0.change_scales(dealias)
    k_0['g'] = k_func(z_de_0, L_CZ, dwk, k_CZ, dk)
    k_z_0.change_scales(dealias)
    k_z_0['g'] = dz(k_0).evaluate()['g']

    # Define Q(z)
    Q_0.change_scales(dealias)
    Q_0['g'] = Q_func(z_de_0, L_Q, dwQ, Qmag, dH)

    logger.info("Solving LBVP to get antiderivative")
    
    Q_0.change_scales(1)
    flux_of_z_0.change_scales(1)
    tau_0 = dist_0.Field(name = 'tau_0')

    problem_0 = d3.LBVP([flux_of_z_0, tau_0], namespace = locals())
    problem_0.add_equation("dz(flux_of_z_0) + lift(tau_0) = Q_0")
    problem_0.add_equation("flux_of_z_0(z = 0) = F_bot")
    solver_0 = problem_0.build_solver()
    solver_0.solve()

    k_0.change_scales(1)
    k_z_0.change_scales(1)
    Q_0.change_scales(1)
    flux_of_z_0.change_scales(1)
    k.change_scales(1)
    k_z.change_scales(1)
    Q.change_scales(1)
    flux_of_z.change_scales(1)
    k['g'] = k_0['g']
    k_z['g'] = k_z_0['g']
    Q['g'] = Q_0['g']
    flux_of_z['g'] = flux_of_z_0['g']

flux = reducer.global_min(flux_of_z(z = L_CZ).evaluate()['g'])

# Define T fields corresponding to adiabatic gradient and radiative flux
if rank == 0:
    T_ad_z_0.change_scales(dealias)
    T_rad_z0_0.change_scales(dealias)
    T_ad_z_0['g'] = -grad_ad # note the minus sign difference from "\nabla_{ad}"
    T_rad_z0_0['g'] = -(flux_of_z_0/k_0).evaluate()['g'] # grad_rad
    
    T_ad_z_0.change_scales(1)
    T_rad_z0_0.change_scales(1)
    T_ad_z.change_scales(1)
    T_rad_z0.change_scales(1) 
    T_ad_z['g'] = T_ad_z_0['g']
    T_rad_z0['g'] = T_rad_z0_0['g']

max_brunt = reducer.global_max(T_rad_z0['g'] - T_ad_z['g'])

# Initialize T0_zz as the derivative of grad_rad (s.t. T0_z would correspond to grad_rad)
# Integrate T0_zz to get T0_z
if rank == 0:
    # Get height at which the specified adiabatic gradient is 
    # matched by the specified value of grad_rad at L_CZ
    zs = np.linspace(0, Lz, 1000)
    k_match = k_func(zs, L_CZ, dwk, k_CZ, dk)
    T_rad_match = -flux/k_match
    z_match = np.interp(-grad_ad, T_rad_match, zs)

    T0_zz_0.change_scales(dealias)
    T0_zz_0['g'] = dz(T_rad_z0_0).evaluate()['g']
    T0_zz_0['g'] *= step_func(z_de_0, z_match - dwT, dwT)

    logger.info("Solving LBVP to get antiderivative")
    
    T0_zz_0.change_scales(1)
    tau_0 = dist_0.Field(name = 'tau_0')

    problem_0 = d3.LBVP([T0_z_0, tau_0], namespace = locals())
    problem_0.add_equation("dz(T0_z_0) + lift(tau_0) = T0_zz_0")
    problem_0.add_equation("T0_z_0(z = 0) = -grad_ad") # dz(T0) = adiabatic gradient at z=0
    solver_0 = problem_0.build_solver()
    solver_0.solve()

    T0_z_0.change_scales(1)
    T0_z.change_scales(1)
    T0_z['g'] = T0_z_0['g'] 

# Ensure T0_zz approaches grad_rad at z = Lz (top), where grad_rad = flux_of_z / k (horizontally avgd)
delta_rad = -grad_rad_top + grad_ad
delta_sim = reducer.global_max(T0_z(z = Lz).evaluate()['g']) + grad_ad

if rank == 0:

    T0_zz_0.change_scales(dealias)
    T0_zz_0['g'] *= delta_rad/delta_sim

    # Recalculate T0_z
    logger.info("Solving LBVP to get antiderivative")

    T0_zz_0.change_scales(1)
    tau_0 = dist_0.Field(name = 'tau_0')
    
    problem_0 = d3.LBVP([T0_z_0, tau_0], namespace = locals())
    problem_0.add_equation("dz(T0_z_0) + lift(tau_0) = T0_zz_0")
    problem_0.add_equation("T0_z_0(z = 0) = -grad_ad")
    solver_0 = problem_0.build_solver()
    solver_0.solve()

    T0_z_0.change_scales(1)
    T0_z.change_scales(1)
    T0_z['g'] = T0_z_0['g']

# Integrate T0_z to get T0
if rank == 0:
    logger.info("Solving LBVP to get antiderivative")

    T0_z_0.change_scales(1)
    tau_0 = dist_0.Field(name = 'tau_0')

    problem_0 = d3.LBVP([T0_0, tau_0], namespace = locals())
    problem_0.add_equation("dz(T0_0) + lift(tau_0) = T0_z_0")
    problem_0.add_equation("T0_0(z = Lz) = 1")
    solver_0 = problem_0.build_solver()
    solver_0.solve()
    
    T0_0.change_scales(1)
    T0.change_scales(1)
    T0['g'] = T0_0['g']
    
# Check that time-stationary heating and cooling roughly cancel at z = Lz
if rank == 0:
    fH_0.change_scales(dealias)
    fH_0['g'] = (Q_0 + k_z_0*T0_z_0 + k_0*T0_zz_0).evaluate()['g']

    logger.info("Solving LBVP to get antiderivative")

    fH_0.change_scales(1)
    tau_0 = dist_0.Field(name = 'tau_0')

    problem_0 = d3.LBVP([fH2_0, tau_0], namespace = locals())
    problem_0.add_equation("dz(fH2_0) + lift(tau_0) = fH_0")
    problem_0.add_equation("fH2_0(z = 0) = 0")
    solver_0 = problem_0.build_solver()
    solver_0.solve()

    fH2_0.change_scales(1)
    fH2.change_scales(1)
    fH2['g'] = fH2_0['g']

logger.info('right(integ(heating - cooling)): {:.3e}'.format(reducer.global_max(fH2(z = Lz).evaluate()['g'])))

# Set up locked fields
H.change_scales(dealias)
H['g'] = (Q + k_z*T0_z + k*T0_zz).evaluate()['g']
H = d3.Grid(H)
Ups.change_scales(dealias)
Ups['g'] = (T0_z - T_ad_z).evaluate()['g'] # = dz(T0) + grad_ad
Ups = d3.Grid(Ups)

##### End of field setup #####

logger.info("Setting up problem")

##### Problem #####

problem = d3.IVP([ux, uy, uz, wx, wy, wz, T1, T1_z, p, tau_ux1, tau_ux2, tau_uy1, tau_uy2, tau_uz1, tau_uz2, tau_T11, tau_T12], namespace = locals())
# First-order substitutions
problem.add_equation("T1_z - dz(T1) + lift(tau_T11) = 0")
problem.add_equation("wx - dy(uz) + dz(uy) + lift(tau_uy1) = 0")
problem.add_equation("wy - dz(ux) + dx(uz) + lift(tau_ux1) = 0")
problem.add_equation("wz - dx(uy) + dy(ux) = 0")
# Continuity
#problem.add_equation("dx(ux) + dy(uy) + dz(uz) = 0", condition="nx!=0")
problem.add_equation("dx(ux) + dy(uy) + dz(uz) + lift(tau_uz1) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("p = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("tau_uz1 = 0", condition="(nx == 0) and (ny == 0)")
# Momentum
problem.add_equation("dt(ux) + Rinv*(dy(wz)-dz(wy)) + dx(p) - Roinv*uy + lift(tau_ux2) = uy*wz-(uz*wy)")
problem.add_equation("dt(uy) + Rinv*(dz(wx)-dx(wz)) + dy(p) + Roinv*ux + lift(tau_uy2) = uz*wx-(ux*wz)")
problem.add_equation("dt(uz) + Rinv*(dx(wy)-dy(wx)) + dz(p) - T1       + lift(tau_uz2) = ux*wy-(uy*wx)", condition="(nx != 0) or  (ny != 0)")
problem.add_equation("uz = 0", condition="(nx == 0) and (ny == 0)")
problem.add_equation("tau_uz2 = 0", condition="(nx == 0) and (ny == 0)")
# Energy
problem.add_equation("dt(T1) - PrRinv*(dx(dx(T1)) + dy(dy(T1)) + dz(T1_z)) + lift(tau_T12) = -(ux*dx(T1) + uy*dy(T1) + uz*T1_z + uz*Ups)", condition="(nx != 0) or  (ny != 0)")
problem.add_equation("dt(T1) - dz(k*T1_z) + lift(tau_T12) = -(ux*dx(T1) + uy*dy(T1) + uz*T1_z + uz*Ups) + H", condition="(nx == 0) and (ny == 0)")
# BCs @ z = 0
problem.add_equation("ux(z = 0) = 0")
problem.add_equation("uy(z = 0) = 0")
problem.add_equation("uz(z = 0) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("T1_z(z = 0) = 0")
# BCs @ z = Lz
problem.add_equation("ux(z = Lz) = 0")
problem.add_equation("uy(z = Lz) = 0")
problem.add_equation("uz(z = Lz) = 0", condition="(nx != 0) or (ny != 0)")
problem.add_equation("T1(z = Lz) = 0")

##### Solver #####
# ncc_cutoff needs to be small to resolve k nccs with large dynamic range

logger.info("Building solver")
solver = problem.build_solver(timestepper, ncc_cutoff=1e-10)
solver.stop_sim_time = stop_sim_time

##### Initial conditions #####

if restart:
    file_handler_mode = 'append'
    write, init_dt = solver.load_state(start_dir)
    print("Read in solver.state from checkpoint, with init_dt = ", init_dt)
else:
    file_handler_mode = 'overwrite'
    # Set up T1 and T1_z with random perturbations
    T1.fill_random('g', seed = 101, distribution = 'normal', scale = 1e-3)
    T1.change_scales(dealias)
    T1['g'] *= np.sin(2*np.pi*z_de/Lz)

    T1_z.change_scales(dealias)
    T1_z['g'] = dz(T1).evaluate()['g']

##### Analysis tasks #####

# Substitutions 
T = T0 + T1
T1_bar = d3.integ(d3.integ(T1, coords['x']), coords['y']) / Lx / Ly
T_bar = T0 + T1_bar
T_fluc = T1 - T1_bar
T_fluc_norm = d3.integ(d3.integ(np.sqrt(T_fluc**2), coords['x']), coords['y']) / Lx / Ly

F_conv_bar = d3.integ(d3.integ(uz * T, coords['x']), coords['y']) / Lx / Ly
F_rad_bar = -k * dz(T_bar)
F_tot_bar_LHS = F_conv_bar + F_rad_bar
F_Q = flux_of_z - F_bot
F_tot_bar_RHS = flux_of_z

grad = -dz(T_bar)

#T = T0 + T1
#T1_bar = d3.Average(T1, (coords['x'], coords['y']))
#T_bar = T0 + T1_bar
#T_fluc = T1 - T1_bar
#T_fluc_norm = d3.Average(np.sqrt(T_fluc**2), (coords['x'], coords['y']))

#F_conv_bar = d3.Average(uz * T, (coords['x'], coords['y']))
#F_conv_bar = d3.Average(uz * (T0 + T1), (coords['x'], coords['y']))
#F_rad_bar = -k * dz(T_bar)
#F_tot_bar_LHS = F_conv_bar + F_rad_bar
#F_Q = flux_of_z - F_bot
#F_tot_bar_RHS = flux_of_z

#grad_rad_0 = (F_tot_bar_RHS)/k
#grad_rad = (F_tot_bar_LHS)/k
#grad = -dz(T_bar)

snapshots_dir = 'snapshots_' + output_suffix
snapshots = solver.evaluator.add_file_handler(filename = snapshots_dir, sim_dt = an_cad, mode=file_handler_mode)
y_slice = Ly / 2
#snapshots.add_task(T_fluc(y = y_slice), scales=dealias, name='T_fluc')
snapshots.add_task(uz(y = y_slice), scales=dealias, name='u_z')
snapshots.add_task(wy(y = y_slice), scales=dealias, name='w_y')
z_slice = Lz / 4
snapshots.add_task(wz(z = z_slice), scales=dealias, name='w_z')

profiles_dir = 'profiles_' + output_suffix
profiles = solver.evaluator.add_file_handler(profiles_dir, sim_dt = an_cad, mode=file_handler_mode)
#profiles.add_task(T_bar, name='T_bar')
#profiles.add_task(T1_bar, name='T1_bar')
#profiles.add_task(T_fluc_norm, name='T_fluc_norm')

#profiles.add_task(F_conv_bar, name='F_conv_bar')
#profiles.add_task(F_rad_bar, name='F_rad_bar')
#profiles.add_task(F_tot_bar_LHS, name='F_tot_bar_LHS')

###profiles.add_task(grad_rad_0/grad_ad, name='gradrad_0') # this is the "specified" grad_rad
###profiles.add_task(grad_rad/grad_ad, name='gradrad') # should track grad_rad in a time average sense / once at equilibrium
#profiles.add_task(grad/grad_ad, name='grad')

profiles_0_dir = 'profiles_0_' + output_suffix
profiles_0 = solver.evaluator.add_file_handler(profiles_0_dir, sim_dt = stop_sim_time/5, mode=file_handler_mode) # will output a few times
profiles_0.add_task(k, name='k')
profiles_0.add_task(T0, name='T0')
profiles_0.add_task(Q, name='heating')
profiles_0.add_task(k_z*T0_z + k*T0_zz, name='cooling')
profiles_0.add_task(F_Q, name='F_Q')
profiles_0.add_task(F_tot_bar_RHS, name='F_tot_bar_RHS')

scalars_dir = 'scalars_' + output_suffix
scalars = solver.evaluator.add_file_handler(scalars_dir, sim_dt = an_cad, mode=file_handler_mode)
scalars.add_task(R*d3.Average(ux**2 + uy**2 + uz**2), name = 'mean_Re')
#scalars.add_task(T1_bar(z = 0), name = 'mean_T1_bot')

##### Flow tools #####

t_ff = 1/np.sqrt(Qmag)
t_N2 = np.sqrt(1/max_brunt)
max_t_step = np.min((0.5*t_ff, t_N2))

if not restart:
    init_dt = max_t_step

CFL = d3.CFL(solver, initial_dt=init_dt, cadence=5, safety=safety, threshold=0.2,
             max_change=1.5, min_change=0.25, max_dt=max_t_step)
CFL.add_velocity(ux*ex + uy*ey + uz*ez)

flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(R*np.sqrt(ux**2 + uy**2 + uz**2), name = 'Re')
flow.add_property(T1(z = 0), name = 'T1_bot')

##### Checkpointing #####

checkpoints_dir = 'checkpoints_' + output_suffix
checkpoints = solver.evaluator.add_file_handler(checkpoints_dir, sim_dt=cp_cad, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

##### Main loop #####

t_step = init_dt
logger.info('Starting main loop with t_step = %e' %(t_step))
try:
    while solver.proceed:
        solver.step(t_step)
        #if (solver.iteration - 1) % 20 == 0:
        if (solver.iteration - 1) % 1 == 0:
            max_Re = flow.max('Re')
            avg_Re = flow.volume_integral('Re') / Lx / Ly / Lz
            avg_T1_bot = flow.grid_average('T1_bot')
            log_string = 'Iteration: {:7d}, '.format(solver.iteration)
            log_string += 'Time: {:8.10e}, dt: {:8.10e}, '.format(solver.sim_time, t_step)
            log_string += 'Re: {:8.10e}/{:8.10e}, '.format(max_Re, avg_Re)
            log_string += 'mean_T1_bot: {:8.10e}, '.format(avg_T1_bot)
            logger.info(log_string)
        t_step = CFL.compute_timestep()
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

