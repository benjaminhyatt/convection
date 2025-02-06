"""
Dedalus (d3) script to run a 2d horizontally-periodic simulation 
with convection and stable stratification in the Boussinesq limit,
following the non-dimensionalization given in Anders et al. (2022). 
This script implements a discontinuous (erf) convective flux profile. 

This script is not optimized--once we have this as a working example,
we will likely change our formulation to reduce the number of fields 
and transforms to the extent feasible. 
"""
import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
from scipy.special import erf

from dedalus.extras import flow_tools

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

### Parameters ###
dtype = np.float64
Lx, Lz = 4, 2
Nx, Nz = 64, 256
dealias = 3/2
timestepper = d3.RK222
safety = 0.1 # CFL prefactor
stop_sim_time = 1e3

# Prognostic
P = 1e0 # Penetration parameter
S = 1e3 # Stiffness
Pr = 5e-1 # Prandtl 
R = 4e2 # Reynolds (freefall)
zeta = 1e-3 # also called mu, ratio between adiabatic flux at bottom to the internal heating flux from Q

Pinv = 1/P
Rinv = 1/R
PrRinv = 1/(Pr*R)

# Model values
L_Q = 1e-1 
dH = 2e-1 # width of internal heating layer: z in [L_Q, L_Q + dH]
Qmag = 1e0 # magnitude of Q inside heating layer
dwQ = 2e-2 # scale of Q jump
L_CZ = 1e0 # where to center jump in flux
dwk = 7.5e-2 # scale of k jump
dwT = 5e-2 # scale of T0_zz jump
F_bot = zeta * Qmag * dH # 1e-3 * 1e0 * 2e-1 = 2e-4

### Function definitions ###
def H_func(z, z0, dw): # 0 to 1
    return (1 + erf((z - z0)/dw))/2

def H_func_c(z, z0, dw): # 1 to 0
    return (1 - erf((z - z0)/dw))/2

def Q_func(z, z0, dw, Qmag, dH):
    return Qmag * H_func(z, z0, dw) * H_func_c(z, z0 + dH, dw)

def k_func(z, z0, dw, k_CZ, dk):
    return k_CZ + dk * H_func(z, z0, dw)

#def T_rad_func(f_g, k_g):
#    return -f_g / k_g

### Bases ###
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype = dtype)
xbasis = d3.RealFourier(coords['x'], size = Nx, bounds = (0, Lx), dealias = dealias)
zbasis = d3.ChebyshevT(coords['z'], size = Nz, bounds = (0, Lz), dealias = dealias)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
dz = lambda A: d3.Differentiate(A, coords['z'])
x, z = dist.local_grids(xbasis, zbasis)
x_de, z_de = dist.local_grids(xbasis, zbasis, scales=(dealias, dealias))
ex, ez = coords.unit_vector_fields(dist)

reducer = flow_tools.GlobalArrayReducer(dist.comm_cart)

### Fields and substitutions ###

# time-varying
u = dist.VectorField(coords, name = 'u', bases = (xbasis, zbasis))
T1 = dist.Field(name = 'T1', bases = (xbasis, zbasis))
p = dist.Field(name = 'p', bases = (xbasis, zbasis))

# tau
tau_u1 = dist.VectorField(coords, name = 'tau_u1', bases = xbasis)
tau_u2 = dist.VectorField(coords, name = 'tau_u2', bases = xbasis)
tau_T11 = dist.Field(name = 'tau_T11', bases = xbasis)
tau_T12 = dist.Field(name = 'tau_T12', bases = xbasis)
tau_p = dist.Field(name = 'tau_p')

# stationary
T0 = dist.Field(name = 'T0', bases = zbasis) 
Q = dist.Field(name = 'Q', bases = zbasis)
k = dist.Field(name = 'k', bases = zbasis)

flux_of_z = dist.Field(name = 'flux_of_z', bases = zbasis)
k_z = dist.Field(name = 'k_z', bases = zbasis)

#CZ_mask = dist.Field(name = 'CZ_mask', bases = zbasis)
#CZ_mask['g'] = H_func(z, 0.2, 0.05) * H_func_c(z, L_CZ, 0.05)

T0_z = dist.Field(name = 'T0_z', bases = zbasis)
T0_zz = dist.Field(name = 'T0_zz', bases = zbasis)

T_ad_z = dist.Field(name = 'T_ad_z', bases = zbasis)
T_rad_z0 = dist.Field(name = 'T_rad_z0', bases = zbasis)

# substitutions
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_T1 = d3.grad(T1)

T1_z = dz(T1)

k_RZ = dH / P / S 
k_CZ = k_RZ * zeta / (1 + zeta + Pinv)
k_ad = k_RZ * (1 + zeta) / (1 + zeta + Pinv)
dk = k_RZ - k_CZ

grad_ad = (Qmag * S * P) * (1 + zeta + Pinv)
grad_rad_top = (Qmag * S * P) * (1 + zeta)

# set fields on grid
k['g'] = k_func(z, L_CZ, dwk, k_CZ, dk)
Q['g'] = Q_func(z, L_Q, dwQ, Qmag, dH)
k_z.change_scales(dealias)
k_z['g'] = dz(k).evaluate()['g']
k_z.change_scales(1)

logger.info("Solving LBVP to get antiderivative")
dist_z = d3.Distributor(coords['z'], comm = MPI.COMM_SELF, dtype = dtype)
F = dist_z.Field(name = 'F', bases = zbasis)
f = dist_z.Field(name = 'f', bases = zbasis)
tau_z = dist_z.Field(name = 'tau_z')
Q.change_scales(1)
comm.Gather(Q['g'], f['g'], root = 0)
sendbuf = None
if rank == 0:
    problem_z = d3.LBVP([F, tau_z], namespace = locals())
    problem_z.add_equation("dz(F) + lift(tau_z) = f")
    problem_z.add_equation("F(z = 0) = F_bot")    
    solver_z = problem_z.build_solver()
    solver_z.solve()
    F.change_scales(1)
    sendbuf = F['g']
comm.Scatter(sendbuf, flux_of_z['g'], root = 0)
flux = reducer.global_min(flux_of_z(z = L_CZ).evaluate()['g'])
print("rank", rank, "flux_of_z['g']", flux_of_z['g'])
print("rank", rank, "flux", flux)
flux_of_z.change_scales(1)
k.change_scales(1)

T_ad_z['g'] = -grad_ad
T_rad_z0.change_scales(dealias)
T_rad_z0['g'] = -(flux_of_z/k).evaluate()['g']#T_rad_func(flux_of_z['g'], k['g']) # grad_rad
T_rad_z0.change_scales(1)
print("rank", rank, "T_rad_z0['g']", T_rad_z0['g'])
max_brunt = reducer.global_max(T_rad_z0['g'] - T_ad_z['g']) 

# gets height at which the specified adiabatic gradient is 
# matched by the specified value of grad_rad at L_CZ
zs = np.linspace(0, Lz, 1000)
k_match = k_func(zs, L_CZ, dwk, k_CZ, dk)
T_rad_match = -flux/k_match #T_rad_func(flux, k_match)
print("rank", rank, "T_rad_match.shape", T_rad_match.shape, "T_rad_match", T_rad_match)
z_match = np.interp(-grad_ad, T_rad_match, zs)
print("rank", rank, "z_match", z_match)

# initialize T0_zz as the derivative of grad_rad (s.t. T0_z would correspond to grad_rad)
T0_zz.change_scales(dealias)
#print("rank", rank, "shapes", dz(T_rad_z0).evaluate()['g'].shape, T0_zz['g'].shape)
T0_zz['g'] = dz(T_rad_z0).evaluate()['g'] 

T0_zz.change_scales(1)
print("rank", rank, "T0_zz['g']", T0_zz['g'])

# T0_zz will be about 92% of dz(grad_rad) at z_match, and is smoothly decreasing for z < z_match,
# such that dz(grad_rad) approaches a constant... 

#print("rank", rank, "shape", T0_zz['g'].shape)
#T0_zz.change_scales(dealias)

#T0_zz['g'] *= H_func(z_de, z_match - dwT, dwT) 
#print(H_func(z_de, z_match - dwT, dwT))
#print(H_func(z, z_match - dwT, dwT))

#T0_zz.change_scales(1)
#T0_zz['g'] *= H_func(z, z_match - dwT, dwT) 

H_func_g = dist.Field(name = 'H_func', bases = zbasis) 
H_func_g['g'] = H_func(z, z_match - dwT, dwT)
H_func_g.change_scales(dealias)
print("rank", rank, "H_func_g['g']", H_func_g['g'])

T0_zz.change_scales(dealias)
T0_zz['g'] = (T0_zz * H_func_g).evaluate()['g'] 
T0_zz.change_scales(1)
print("rank", rank, "T0_zz['g'] after H", T0_zz['g'])

# integrate T0_zz to get T0_z
logger.info("Solving LBVP to get antiderivative")
F = dist_z.Field(name = 'F', bases = zbasis)
f = dist_z.Field(name = 'f', bases = zbasis)
tau_z = dist_z.Field(name = 'tau_z')
T0_zz.change_scales(1)
comm.Gather(T0_zz['g'], f['g'], root = 0)
sendbuf = None
if rank == 0:
    problem_z = d3.LBVP([F, tau_z], namespace = locals())
    problem_z.add_equation("dz(F) + lift(tau_z) = f")
    problem_z.add_equation("F(z = 0) = -grad_ad")    
    solver_z = problem_z.build_solver()
    solver_z.solve()
    F.change_scales(1)
    sendbuf = F['g']
T0_z.change_scales(1)
comm.Scatter(sendbuf, T0_z['g'], root = 0)

# ensure T0_zz approaches grad_rad at z = Lz (top of domain), where grad_rad = flux_of_z / k (horizontally avgd)
delta_rad = -grad_rad_top + grad_ad
delta_sim = reducer.global_max(T0_z(z = Lz).evaluate()['g']) + grad_ad
T0_zz['g'] *= delta_rad/delta_sim # multiplies by about 0.07519

# recalculate T0_z
logger.info("Solving LBVP to get antiderivative")
F = dist_z.Field(name = 'F', bases = zbasis)
f = dist_z.Field(name = 'f', bases = zbasis)
tau_z = dist_z.Field(name = 'tau_z')
comm.Gather(T0_zz['g'], f['g'], root = 0)
sendbuf = None
if rank == 0:
    problem_z = d3.LBVP([F, tau_z], namespace = locals())
    problem_z.add_equation("dz(F) + lift(tau_z) = f")
    problem_z.add_equation("F(z = 0) = -grad_ad")    
    solver_z = problem_z.build_solver()
    solver_z.solve()
    F.change_scales(1)
    sendbuf = F['g']
T0_z.change_scales(1)
comm.Scatter(sendbuf, T0_z['g'], root = 0)

# integrate T0_z to get T0
logger.info("Solving LBVP to get antiderivative")
F = dist_z.Field(name = 'F', bases = zbasis)
f = dist_z.Field(name = 'f', bases = zbasis)
tau_z = dist_z.Field(name = 'tau_z')
T0_z.change_scales(1)
comm.Gather(T0_z['g'], f['g'], root = 0)
sendbuf = None
if rank == 0:
    problem_z = d3.LBVP([F, tau_z], namespace = locals())
    problem_z.add_equation("dz(F) + lift(tau_z) = f")
    problem_z.add_equation("F(z = Lz) = 1")    
    solver_z = problem_z.build_solver()
    solver_z.solve()
    F.change_scales(1)
    sendbuf = F['g']
comm.Scatter(sendbuf, T0['g'], root = 0)


Q.change_scales(1)
k.change_scales(1)
k_z.change_scales(1)
T0_z.change_scales(1)
T0_zz.change_scales(1)

# check that heating and cooling cancel out
fH = dist.Field(name = 'fH', bases = zbasis)
fH2 = dist.Field(name = 'fH2', bases = zbasis)
fH['g'] = Q['g'] +  k_z['g']*T0_z['g'] + k['g']*T0_zz['g']
logger.info("Solving LBVP to get antiderivative")
F = dist_z.Field(name = 'F', bases = zbasis)
f = dist_z.Field(name = 'f', bases = zbasis)
tau_z = dist_z.Field(name = 'tau_z')
fH.change_scales(1)
comm.Gather(fH['g'], f['g'], root = 0)
sendbuf = None
if rank == 0:
    problem_z = d3.LBVP([F, tau_z], namespace = locals())
    problem_z.add_equation("dz(F) + lift(tau_z) = f")
    problem_z.add_equation("F(z = 0) = 0")    
    solver_z = problem_z.build_solver()
    solver_z.solve()
    F.change_scales(1)
    sendbuf = F['g']
comm.Scatter(sendbuf, fH2['g'], root = 0)

logger.info('right(integ(heating - cooling)): {:.3e}'.format(reducer.global_max(fH2(z = Lz).evaluate()['g']))) # currently is around 2e-6, but should it be zero to floating point precision?


### Comparison to d2 ###
print("rank", rank, "k['g']", k['g'])
print("rank", rank, "T0['g']", T0['g'])
print("rank", rank, "T0_z['g']", T0_z['g'])
print("rank", rank, "T0_zz['g']", T0_zz['g'])

### Problem ###
problem = d3.IVP([u, T1, p, tau_u1, tau_u2, tau_T11, tau_T12, tau_p], namespace = locals())
problem.add_equation("trace(grad_u) + tau_p = 0") # continuity
problem.add_equation("dt(u) - Rinv*div(grad_u) + grad(p) - T1*ez + lift(tau_u2) = - u@grad(u)") # momentum
problem.add_equation("dt(T1) - PrRinv*div(grad_T1) + dz(lift(tau_T11)) + lift(tau_T12) = - u@grad(T1) - u@ez * (T0_z - T_ad_z)", condition='nx!=0') # energy
#problem.add_equation("dt(T1) + dz(lift(tau_T11)) + lift(tau_T12) - dz(k*T1_z) = - u@grad(T1) - u@ez * (T0_z - T_ad_z) + Q + k_z*T0_z + k*T0_zz", condition='nx==0')
problem.add_equation("dt(T1) + dz(lift(tau_T11)) + lift(tau_T12) - dz(k*T1_z) = - u@grad(T1) - u@ez * (T0_z - T_ad_z) + Q + dz(k)*T0_z + k*T0_zz", condition='nx==0')
problem.add_equation("u(z = 0) = 0") # no slip
problem.add_equation("u(z = Lz) = 0") # no slip
problem.add_equation("T1_z(z = 0) = 0") # fixed flux
problem.add_equation("T1(z = Lz) = 0") # fixed temp
problem.add_equation("integ(p) = 0") # pressure gauge

### Solver ###
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

### Initial conditions ###
T1.fill_random('g', seed = 101, distribution = 'normal', scale = 1e-3)
T1['g'] *= np.sin(2*np.pi*z/Lz)
### Analysis tasks ###

# substitutions
T_bar = d3.Average(T0 + T1, coords['x'])
T1_rms = np.sqrt(d3.integ(T1*T1))

snapshots = solver.evaluator.add_file_handler('snapshots_par', sim_dt = 0.25)
snapshots.add_task(T0 + T1 - T_bar, name='T-T_bar')
snapshots.add_task(T_bar, name='T_bar') # should be 1d, but want included with this data
snapshots.add_task(T1, name='temp')
snapshots.add_task(T1/T1_rms, name='temp-rms')
snapshots.add_task(ex@u, name='u_x')
snapshots.add_task(ez@u, name='u_z')
snapshots.add_task(p, name='pressure')

profiles = solver.evaluator.add_file_handler('profiles_par', sim_dt = 0.25)
# fluxes (horizontally-averaged)
# time stationary part
F_conv_bar_0 = d3.Average((u @ ez) * T0, coords['x'])
F_rad_bar_0 = -k * T0_z
F_Q = flux_of_z - F_bot
F_tot_bar_0_LHS = F_conv_bar_0 + F_rad_bar_0
F_tot_bar_0_RHS = flux_of_z
profiles.add_task(F_conv_bar_0, name='F_conv_bar_0')
profiles.add_task(F_rad_bar_0, name='F_rad_bar_0')
profiles.add_task(F_Q, name='F_Q')
profiles.add_task(F_tot_bar_0_LHS, name='F_tot_bar_0_LHS')
profiles.add_task(F_tot_bar_0_RHS, name='F_tot_bar_0_RHS')

# full
F_conv_bar = d3.Average((u @ ez) * (T0 + T1), coords['x'])
F_rad_bar = -k * dz(T_bar)
F_tot_bar_LHS = F_conv_bar + F_rad_bar
#F_tot_bar_RHS = -grad_ad * antiderivative(d3.Average(u@ez, coords['x'])) #not implemented: would need to keep track of an indefinite integral on each step

profiles.add_task(F_conv_bar, name='F_conv_bar')
profiles.add_task(F_rad_bar, name='F_rad_bar')
profiles.add_task(F_tot_bar_LHS, name='F_tot_bar_LHS') 
# for F_tot_bar_RHS, we will refer to the time stationary setup above

# temperature gradients
grad_rad_0 = (F_tot_bar_0_RHS)/k
grad_rad = (F_tot_bar_LHS)/k
grad = -dz(T_bar)
profiles.add_task(grad_rad_0/grad_ad, name='gradrad_0') # this is the "specified" grad_rad
profiles.add_task(grad_rad/grad_ad, name='gradrad') # should track grad_rad in a time average sense / once at equilibrium
profiles.add_task(grad/grad_ad, name='grad')

# these should all be time stationary, but double-checking for sanity
#profiles.add_task(Q, name='heating')
#profiles.add_task(k_z*T0_z + k*T0_zz, name='cooling')
#profiles.add_task(flux_of_z, name='Qflux')
#profiles.add_task(k, name='kprof')
#profiles.add_task(T_rad_z0, name='T_rad_z0prof')
#profiles.add_task(T0, name='T0')
#profiles.add_task(T0_z, name='T0_z')
#profiles.add_task(T0_zz, name='T0_zz')
#profiles.add_task(k_z*T0_z, name='k_zT0_z')
#profiles.add_task(k*T0_zz, name='kT0_zz')
#profiles.add_task(fH, name='fHprof')
#profiles.add_task(fH2, name='fH2prof')
#profiles.add_task(T0_z - T_ad_z, name='T0_z-T_ad_z')
# checking terms comprising dz(T_rad_z0)
#profiles.add_task((-flux_of_z/(k**2))*k_z, name='Qfluxk2k_z')
#profiles.add_task(-flux_of_z/(k**2), name='Qfluxk2')
#profiles.add_task(k_z, name='k_zprof')
#profiles.add_task(Q/k, name='Qzprof')
#profiles.add_task(k**(-1), name='k1')
#profiles.add_task(k**(-2), name='k2')


### Flow tools ###
t_ff = 1/np.sqrt(Qmag)
t_N2 = np.sqrt(1/max_brunt)
max_t_step = np.min((0.5*t_ff, t_N2))
#t_step = max_t_step
t_step = 1e-5
print("Beginning with t_step = ", t_step)

CFL = d3.CFL(solver, initial_dt=t_step, cadence=1, safety=safety, threshold=0.1,
             max_change=1.5, min_change=0.25, max_dt=max_t_step)
CFL.add_velocity(u)

flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(R*np.sqrt(u@u), name = 'Re')
flow.add_property(d3.integ(dz(T1)), name = 'T1_z')

### Main loop ###
try:
    logger.info('Starting main loop')
    while solver.proceed:
        t_step = CFL.compute_timestep()
        solver.step(t_step)
        if (solver.iteration - 1) % 20 == 0:
            max_Re = flow.max('Re')
            avg_Re = flow.grid_average('Re')
            avg_T1_z = flow.max('T1_z')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, avg(Re)=%f, integ(dz(T1))=%f' %(solver.iteration, solver.sim_time, t_step, max_Re, avg_Re, avg_T1_z))
            #print(d3.Average((u@ez), coords['x']).evaluate()['g'])
            #print((u@ez)(x=1).evaluate()['g'])
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
