"""
Dedalus (d3) script to run a 2d horizontally-periodic simulation 
with convection and stable stratification in the Boussinesq limit,
following the non-dimensionalization given in Anders et al. (2022). 
This script implements a discontinuous (erf) convective flux profile. 

First run convection_ivp_2d_setup.py, which this script will pull its
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

### Parameters ###

# Prognostic
P = 1e0 # Penetration parameter
S = 1e3 # Stiffness
Pr = 5e-1 # Prandtl 
R = 4e2 # Reynolds (freefall)
mu = 1e-3 # F_H/F_bot
Pinv = 1/P 
Rinv = 1/R 
PrRinv = 1/(Pr*R)

# Model
L_Q = 1e-1 
dH = 2e-1 # width of internal heating layer: z in [L_Q, L_Q + dH]
Qmag = 1e0 # magnitude of Q inside heating layer
dwQ = 2e-2 # scale of Q jump
L_CZ = 1e0 # where to center jump in flux
dwk = 7.5e-2 # scale of k jump
dwT = 5e-2 # scale of T0_zz jump
F_bot = mu * Qmag * dH # radiative flux through z = 0

# Simulation
dtype = np.float64
Lx, Lz = 4, 2
Nx, Nz = 64, 256
dealias = 3/2
timestepper = d3.RK443
safety = 0.35 # CFL prefactor
stop_sim_time = 5e4 # freefall times
output_suffix = 'mu_{:.0e}'.format(mu) + '_R_{:.0e}'.format(R).replace('+','')
cp_cad = 100 # checkpoint cadence in freefall times

# Initialization
restart = False
setup_dir = 'setup_' + output_suffix

if not restart:
    start_dir = setup_dir + '/' + setup_dir + '_s1.h5'

else:
    cp_idx = 300
    bad_output_suffix = 'mu_{:.0e}'.format(mu) + 'R_{:.0e}'.format(R)
    start_dir = 'checkpoints_' + output_suffix + '/checkpoints_' + bad_output_suffix + '_s' + str(cp_idx) + '.h5'

### Bases ###
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype = dtype)
xbasis = d3.RealFourier(coords['x'], size = Nx, bounds = (0, Lx), dealias = dealias)
zbasis = d3.ChebyshevT(coords['z'], size = Nz, bounds = (0, Lz), dealias = dealias)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
dx = lambda A: d3.Differentiate(A, coords['x'])
dz = lambda A: d3.Differentiate(A, coords['z'])
x, z = dist.local_grids(xbasis, zbasis)
x_de, z_de = dist.local_grids(xbasis, zbasis, scales=(dealias, dealias))
ex, ez = coords.unit_vector_fields(dist)

reducer = flow_tools.GlobalArrayReducer(dist.comm_cart)

### Fields and substitutions ###

# Time-varying
ux = dist.Field(name = 'ux', bases = (xbasis, zbasis))
uz = dist.Field(name = 'uz', bases = (xbasis, zbasis))
w = dist.Field(name = 'w', bases = (xbasis, zbasis))
T1 = dist.Field(name = 'T1', bases = (xbasis, zbasis))
T1_z = dist.Field(name = 'T1_z', bases = (xbasis, zbasis))
p = dist.Field(name = 'p', bases = (xbasis, zbasis))

# Tau
tau_ux1 = dist.Field(name = 'tau_ux1', bases = xbasis)
tau_ux2 = dist.Field(name = 'tau_ux2', bases = xbasis)
tau_uz1 = dist.Field(name = 'tau_uz1', bases = xbasis)
tau_uz2 = dist.Field(name = 'tau_uz2', bases = xbasis)
tau_T11 = dist.Field(name = 'tau_T11', bases = xbasis)
tau_T12 = dist.Field(name = 'tau_T12', bases = xbasis)
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

# More model definitions
k_RZ = dH / P / S 
k_CZ = k_RZ * mu / (1 + mu + Pinv)
k_ad = k_RZ * (1 + mu) / (1 + mu + Pinv)
dk = k_RZ - k_CZ

grad_ad = (Qmag * S * P) * (1 + mu + Pinv)
grad_rad_top = (Qmag * S * P) * (1 + mu)

### Load in non-solver fields ###
logger.info("Loading in non-solver fields")
fields_path = setup_dir + '/setup_fields_' + output_suffix + '.h5'
fields_file = h5py.File(fields_path, 'r')

# Fields in problem
sendbuf = None
H.change_scales(dealias)
if rank == 0:
    sendbuf = np.array(fields_file['problem_fields']['H'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, H['g'], root = 0)
H = d3.Grid(H)

sendbuf = None
k.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['problem_fields']['k'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, k['g'], root = 0)

sendbuf = None
Ups.change_scales(dealias)
if rank == 0:
    sendbuf = np.array(fields_file['problem_fields']['Ups'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, Ups['g'], root = 0)
Ups = d3.Grid(Ups)

# Other fields
sendbuf = None
flux_of_z.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['analysis_fields']['flux_of_z'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, flux_of_z['g'], root = 0)

sendbuf = None
k_z.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['analysis_fields']['k_z'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, k_z['g'], root = 0)

sendbuf = None
Q.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['analysis_fields']['Q'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, Q['g'], root = 0)

sendbuf = None
T0.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['analysis_fields']['T0'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, T0['g'], root = 0)

sendbuf = None
T0_z.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['analysis_fields']['T0_z'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, T0_z['g'], root = 0)

sendbuf = None
T0_zz.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['analysis_fields']['T0_zz'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, T0_zz['g'], root = 0)

sendbuf = None
T_ad_z.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['analysis_fields']['T_ad_z'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, T_ad_z['g'], root = 0)

sendbuf = None
T_rad_z0.change_scales(1)
if rank == 0:
    sendbuf = np.array(fields_file['analysis_fields']['T_rad_z0'], dtype = dtype)[0, :]
comm.Scatter(sendbuf, T_rad_z0['g'], root = 0)

### Problem ###
problem = d3.IVP([ux, uz, w, T1, T1_z, p, tau_ux1, tau_ux2, tau_uz1, tau_uz2, tau_T11, tau_T12], namespace = locals())
# First-order substitutions
problem.add_equation("T1_z - dz(T1) + lift(tau_T11) = 0")
problem.add_equation("w - dz(ux) + dx(uz) + lift(tau_ux1) = 0")
# Continuity
problem.add_equation("dx(ux) + dz(uz) + lift(tau_uz1) = 0", condition="nx!=0")
problem.add_equation("p = 0", condition="nx==0")
problem.add_equation("tau_uz1 = 0", condition="nx==0")
# Momentum
problem.add_equation("dt(ux) + Rinv*(-dz(w)) + dx(p) + lift(tau_ux2) = -(uz*w)")
problem.add_equation("dt(uz) + Rinv*(dx(w)) + dz(p) - T1 + lift(tau_uz2) = ux*w", condition="nx!=0")
problem.add_equation("uz = 0", condition="nx==0")
problem.add_equation("tau_uz2 = 0", condition="nx==0")
# Energy
problem.add_equation("dt(T1) - PrRinv*(dx(dx(T1)) + dz(T1_z)) + lift(tau_T12) = -(ux*dx(T1) + uz*T1_z + uz*Ups)", condition="nx!=0")
problem.add_equation("dt(T1) - dz(k*T1_z) + lift(tau_T12) = -(ux*dx(T1) + uz*T1_z + uz*Ups) + H", condition="nx==0")
# BCs @ z = 0
problem.add_equation("ux(z = 0) = 0")
problem.add_equation("uz(z = 0) = 0", condition="nx!=0")
problem.add_equation("T1_z(z = 0) = 0")
# BCs @ z = Lz
problem.add_equation("ux(z = Lz) = 0")
problem.add_equation("uz(z = Lz) = 0", condition="nx!=0")
problem.add_equation("T1(z = Lz) = 0")

### Solver ###
# ncc_cutoff needs to be small to resolve k nccs with large dynamic range
solver = problem.build_solver(timestepper, ncc_cutoff=1e-10)
solver.stop_sim_time = stop_sim_time

### Initial conditions (i.e., load in solver fields) ###
if not restart: 
    file_handler_mode = 'overwrite'
    write, init_dt = solver.load_state(start_dir)   
    logger.info("Read in solver.state from setup script") 

else: 
    file_handler_mode = 'append'
    write, init_dt = solver.load_state(start_dir)
    print("Read in solver.state from checkpoint, with init_dt = ", init_dt)
    #init_dt *= 0.5

### Analysis tasks ###

# Substitutions

T1_bar = d3.Average(T1, coords['x'])
T_bar = T0 + T1_bar

F_conv_bar = d3.Average(uz * (T0 + T1), coords['x'])
F_rad_bar = -k * dz(T_bar)
F_tot_bar_LHS = F_conv_bar + F_rad_bar
F_Q = flux_of_z - F_bot
F_tot_bar_RHS = flux_of_z

grad_rad_0 = (F_tot_bar_RHS)/k
grad_rad = (F_tot_bar_LHS)/k
grad = -dz(T_bar)

snapshots_dir = 'snapshots_' + output_suffix
snapshots = solver.evaluator.add_file_handler(filename = snapshots_dir, sim_dt = 1, mode=file_handler_mode)
snapshots.add_task(T0 + T1 - T_bar, name='T-T_bar')
snapshots.add_task(ux, name='u_x')
snapshots.add_task(uz, name='u_z')
snapshots.add_task(p, name='pressure')

profiles_dir = 'profiles_' + output_suffix
profiles = solver.evaluator.add_file_handler(profiles_dir, sim_dt = 1, mode=file_handler_mode)
profiles.add_task(T_bar, name='T_bar')
profiles.add_task(T1_bar, name='T1_bar')

profiles.add_task(F_conv_bar, name='F_conv_bar')
profiles.add_task(F_rad_bar, name='F_rad_bar')
profiles.add_task(F_tot_bar_LHS, name='F_tot_bar_LHS')

profiles.add_task(grad_rad_0/grad_ad, name='gradrad_0') # this is the "specified" grad_rad
profiles.add_task(grad_rad/grad_ad, name='gradrad') # should track grad_rad in a time average sense / once at equilibrium
profiles.add_task(grad/grad_ad, name='grad')

profiles_0_dir = 'profiles_0_' + output_suffix
profiles_0 = solver.evaluator.add_file_handler(profiles_0_dir, sim_dt = stop_sim_time/5, mode=file_handler_mode) # will output a few times
profiles_0.add_task(k, name='k')
profiles_0.add_task(T0, name='T0')
profiles_0.add_task(Q, name='heating')
profiles_0.add_task(k_z*T0_z + k*T0_zz, name='cooling')
profiles_0.add_task(F_Q, name='F_Q')
profiles_0.add_task(F_tot_bar_RHS, name='F_tot_bar_RHS')

scalars_dir = 'scalars_' + output_suffix
scalars = solver.evaluator.add_file_handler(scalars_dir, sim_dt = 1, mode=file_handler_mode)
scalars.add_task(R*d3.Average(ux**2 + uz**2), name = 'mean_Re')
scalars.add_task(T1_bar(z = 0), name = 'mean_T1_bot')

### Flow tools ###
max_brunt = reducer.global_max(T_rad_z0['g'] - T_ad_z['g'])
t_ff = 1/np.sqrt(Qmag)
t_N2 = np.sqrt(1/max_brunt)
max_t_step = np.min((0.5*t_ff, t_N2))

CFL = d3.CFL(solver, initial_dt=init_dt, cadence=1, safety=safety, threshold=0.2,
             max_change=1.5, min_change=0.25, max_dt=max_t_step)
CFL.add_velocity(ux*ex + uz*ez)

flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(R*np.sqrt(ux**2 + uz**2), name = 'Re')
flow.add_property(T1(z = 0), name = 'T1_bot')

### Checkpointing ###
checkpoints_dir = 'checkpoints_' + output_suffix
checkpoints = solver.evaluator.add_file_handler(checkpoints_dir, sim_dt=cp_cad, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

### Main loop ###
t_step = init_dt
logger.info('Starting main loop with t_step = %e' %(t_step))
try:
    while solver.proceed:
        solver.step(t_step)
        if (solver.iteration - 1) % 10 == 0:
            max_Re = flow.max('Re')
            avg_Re = flow.volume_integral('Re') / Lx / Lz
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
