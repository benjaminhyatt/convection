"""
Dedalus (d3) script to set up initial conditions and background fields 
for a 2d horizontally-periodic simulation with convection and stable 
stratification in the Boussinesq limit. 
This script implements a discontinuous (erf) convective flux profile. 
"""
import numpy as np
import h5py
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.special import erf
from dedalus.extras import flow_tools

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

# Simulation
dtype = np.float64
Lx, Lz = 4, 2
Nx, Nz = 64, 256 
dealias = 3/2 
output_suffix = 'mu_{:.0e}'.format(mu).replace('-','m') + '_R_{:.0e}'.format(R).replace('+','p')

# Model
L_Q = 1e-1
dH = 2e-1 # width of internal heating layer: z in [L_Q, L_Q + dH]
Qmag = 1e0 # magnitude of Q inside heating layer
dwQ = 2e-2 # scale of Q jump
L_CZ = 1e0 # where to center jump in flux
dwk = 7.5e-2 # scale of k jump
dwT = 5e-2 # scale of T0_zz jump
F_bot = mu * Qmag * dH # radiative flux through z = 0

### Function definitions ###
def step_func(z, z0, dw): # 0 to 1
    return (1 + erf((z - z0)/dw))/2

def step_func_c(z, z0, dw): # 1 to 0
    return (1 - erf((z - z0)/dw))/2

def Q_func(z, z0, dw, Qmag, dH):
    return Qmag * step_func(z, z0, dw) * step_func_c(z, z0 + dH, dw)

def k_func(z, z0, dw, k_CZ, dk):
    return k_CZ + dk * step_func(z, z0, dw)

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

# Define k = k(z) and k_z = dz(k)
k.change_scales(dealias)
k['g'] = k_func(z_de, L_CZ, dwk, k_CZ, dk)
k_z.change_scales(dealias)
k_z['g'] = dz(k).evaluate()['g']

# Define Q(z)
Q.change_scales(dealias)
Q['g'] = Q_func(z_de, L_Q, dwQ, Qmag, dH)

# Integrate to determine vertical flux due to Q and F_bot
flux_of_z.change_scales(1)
Q.change_scales(1)
logger.info("Solving LBVP to get antiderivative")
tau_z = dist.Field(name = 'tau_z')
problem_z = d3.LBVP([flux_of_z, tau_z], namespace = locals())
problem_z.add_equation("dz(flux_of_z) + lift(tau_z) = Q")
problem_z.add_equation("flux_of_z(z = 0) = F_bot")
solver_z = problem_z.build_solver()
solver_z.solve()
flux = reducer.global_min(flux_of_z(z = L_CZ).evaluate()['g'])

# Define T fields corresponding to adiabatic gradient and radiative flux
T_ad_z.change_scales(dealias)
T_rad_z0.change_scales(dealias)
T_ad_z['g'] = -grad_ad
T_rad_z0['g'] = -(flux_of_z/k).evaluate()['g'] # grad_rad
max_brunt = reducer.global_max(T_rad_z0['g'] - T_ad_z['g'])

# Get height at which the specified adiabatic gradient is 
# matched by the specified value of grad_rad at L_CZ
zs = np.linspace(0, Lz, 1000)
k_match = k_func(zs, L_CZ, dwk, k_CZ, dk)
T_rad_match = -flux/k_match
z_match = np.interp(-grad_ad, T_rad_match, zs)

# Initialize T0_zz as the derivative of grad_rad (s.t. T0_z would correspond to grad_rad)
T0_zz.change_scales(dealias)
T0_zz['g'] = dz(T_rad_z0).evaluate()['g']
T0_zz['g'] *= step_func(z_de, z_match - dwT, dwT)

# Integrate T0_zz to get T0_z
logger.info("Solving LBVP to get antiderivative")
tau_z = dist.Field(name = 'tau_z')
problem_z = d3.LBVP([T0_z, tau_z], namespace = locals())
problem_z.add_equation("dz(T0_z) + lift(tau_z) = T0_zz")
problem_z.add_equation("T0_z(z = 0) = -grad_ad")
solver_z = problem_z.build_solver()
solver_z.solve()

# Ensure T0_zz approaches grad_rad at z = Lz (top), where grad_rad = flux_of_z / k (horizontally avgd)
delta_rad = -grad_rad_top + grad_ad
delta_sim = reducer.global_max(T0_z(z = Lz).evaluate()['g']) + grad_ad
T0_zz['g'] *= delta_rad/delta_sim

# Recalculate T0_z
logger.info("Solving LBVP to get antiderivative")
tau_z = dist.Field(name = 'tau_z')
problem_z = d3.LBVP([T0_z, tau_z], namespace = locals())
problem_z.add_equation("dz(T0_z) + lift(tau_z) = T0_zz")
problem_z.add_equation("T0_z(z = 0) = -grad_ad")
solver_z = problem_z.build_solver()
solver_z.solve()

# Integrate T0_z to get T0
logger.info("Solving LBVP to get antiderivative")
tau_z = dist.Field(name = 'tau_z')
problem_z = d3.LBVP([T0, tau_z], namespace = locals())
problem_z.add_equation("dz(T0) + lift(tau_z) = T0_z")
problem_z.add_equation("T0(z = Lz) = 1")
solver_z = problem_z.build_solver()
solver_z.solve()

# Check that time-stationary heating and cooling roughly cancel at z = Lz
fH = dist.Field(name = 'fH', bases = zbasis)
fH2 = dist.Field(name = 'fH2', bases = zbasis)
fH.change_scales(dealias)
fH['g'] = (Q + k_z*T0_z + k*T0_zz).evaluate()['g']

logger.info("Solving LBVP to get antiderivative")
tau_z = dist.Field(name = 'tau_z')
problem_z = d3.LBVP([fH2, tau_z], namespace = locals())
problem_z.add_equation("dz(fH2) + lift(tau_z) = fH")
problem_z.add_equation("fH2(z = 0) = 0")
solver_z = problem_z.build_solver()
solver_z.solve()
logger.info('right(integ(heating - cooling)): {:.3e}'.format(reducer.global_max(fH2(z = Lz).evaluate()['g'])))

# Set up locked fields
H.change_scales(dealias)
H['g'] = (Q + k_z*T0_z + k*T0_zz).evaluate()['g']
H = d3.Grid(H)
Ups.change_scales(dealias)
Ups['g'] = (T0_z - T_ad_z).evaluate()['g']
Ups = d3.Grid(Ups)

# Set up T1 and T1_z with random perturbations
T1.fill_random('g', seed = 101, distribution = 'normal', scale = 1e-3)
T1.change_scales(dealias)
T1['g'] *= np.sin(2*np.pi*z_de/Lz)

T1_z.change_scales(dealias)
T1_z['g'] = dz(T1).evaluate()['g']

# Initialize problem, solver and output solver.state to be loaded in by main ivp script
timestepper = d3.RK443
t_ff = 1/np.sqrt(Qmag)
t_N2 = np.sqrt(1/max_brunt)
max_t_step = np.min((0.5*t_ff, t_N2))

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
solver = problem.build_solver(timestepper, ncc_cutoff=1e-10, warmup_iterations=5)
solver.stop_iteration = 10
t_step = max_t_step

setup_dir = 'setup_' + output_suffix
setup = solver.evaluator.add_file_handler(setup_dir, max_writes=1, iter=solver.stop_iteration+1)
#setup = solver.evaluator.add_file_handler(setup_dir, group = 'setup', max_writes=1)
setup.add_tasks(solver.state)
#solver.evaluator.evaluate_group('setup')

try:
    logger.info("Taking %d iterations with dt = %e to get solver to output initial state" %(solver.stop_iteration, t_step))
    while solver.proceed:
        solver.step(t_step)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

# Output non-solver background fields to separate h5 file
logger.info("Saving non-solver fields")
fields_path = setup_dir + '/setup_fields_' + output_suffix + '.h5'
fields_file = h5py.File(fields_path, 'w')

# In problem
problem_fields = fields_file.create_group('problem_fields')
k.change_scales(1)
problem_fields.create_dataset('H', data=H['g']) # locked on dealiased grid
problem_fields.create_dataset('k', data=k['g'])
problem_fields.create_dataset('Ups', data=Ups['g']) # locked on dealiased grid

# In analysis and/or setup
analysis_fields = fields_file.create_group('analysis_fields')
flux_of_z.change_scales(1)
k_z.change_scales(1)
Q.change_scales(1)
T0.change_scales(1)
T0_z.change_scales(1)
T0_zz.change_scales(1)
T_ad_z.change_scales(1)
T_rad_z0.change_scales(1)
analysis_fields.create_dataset('flux_of_z', data=flux_of_z['g'])
analysis_fields.create_dataset('k_z', data=k_z['g'])
analysis_fields.create_dataset('Q', data=Q['g'])
analysis_fields.create_dataset('T0', data=T0['g'])
analysis_fields.create_dataset('T0_z', data=T0_z['g'])
analysis_fields.create_dataset('T0_zz', data=T0_zz['g'])
analysis_fields.create_dataset('T_ad_z', data=T_ad_z['g'])
analysis_fields.create_dataset('T_rad_z0', data=T_rad_z0['g'])

fields_file.close()
logger.info("Finished setup successfully")
