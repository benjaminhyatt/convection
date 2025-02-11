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

#restart = sys.argv[1]

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

def T_rad_func(f_g, k_g):
    return -f_g / k_g

# F = integ(f)
#def antiderivative(f, F, b, c, cstr, c0, F0):
#
#    logger.info("Solving LBVP to get antiderivative")
#    d = lambda A: d3.Differentiate(A, c[cstr])
#    tau_1 = dist.Field(name = 'tau_1')
#    lb = b.derivative_basis(1)
#    lift_b = lambda A: d3.Lift(A, lb, -1)
#
#    problem_F = d3.LBVP([F, tau_1], namespace = locals())
#    problem_F.add_equation("d(F) + lift_b(tau_1) = f")
#    problem_F.add_equation("F("+cstr+"="+str(c0)+") = F0")
#
#    solver_F = problem_F.build_solver()
#    solver_F.solve()
#    f.change_scales(1)
#    return F['g']

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

T_ad = dist.Field(name = 'T_ad', bases = zbasis)
T_rad_z0 = dist.Field(name = 'T_rad_z0', bases = zbasis)

# substitutions
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_T1 = d3.grad(T1)

T1_z = dz(T1)

k_RZ = dH / P / S # 2e-1 / 1e0 / 1e3 = 2e-4
k_CZ = k_RZ * zeta / (1 + zeta + Pinv) # 2e-4 * 1e-3 / (1 + 1e-3 + 1 / 1e0) = approx 1e-7
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

#flux_of_z['g'] = antiderivative(Q, flux_of_z, zbasis, coords, 'z', 0, F_bot)
logger.info("Solving LBVP to get antiderivative")
F = dist.Field(name = 'F', bases = zbasis)
tau_z = dist.Field(name = 'tau_z')
Q.change_scales(1)
problem_z = d3.LBVP([F, tau_z], namespace = locals())
problem_z.add_equation("dz(F) + lift(tau_z) = Q")
problem_z.add_equation("F(z = 0) = F_bot")    
solver_z = problem_z.build_solver()
solver_z.solve()
F.change_scales(1)
flux_of_z.change_scales(1)
flux_of_z['g'] = F['g']
# F_tot at height z=L_CZ (I think evaluate sends the result to rank 0, hence the reduction, but no need for it to be min in particular)
flux = reducer.global_min(flux_of_z(z = L_CZ).evaluate()['g'])

flux_of_z.change_scales(1)
k.change_scales(1)

T_ad['g'] = -grad_ad
T_rad_z0['g'] = T_rad_func(flux_of_z['g'], k['g'])

max_brunt = reducer.global_max(T_rad_z0['g'] - T_ad['g'])

# gets height at which the specified adiabatic gradient is 
# matched by the specified value of grad_rad at L_CZ
zs = np.linspace(0, Lz, 1000)
k_match = k_func(zs, L_CZ, dwk, k_CZ, dk)
T_rad_match = T_rad_func(flux, k_match)
z_match = np.interp(-grad_ad, T_rad_match, zs)

# initialize T0_zz as the derivative of grad_rad (s.t. T0_z would correspond to grad_rad)
T0_zz.change_scales(dealias)
T0_zz['g'] = dz(T_rad_z0).evaluate()['g']
# T0_zz will be about 92% of dz(grad_rad) at z_match, and is smoothly decreasing for z < z_match,
# such that dz(grad_rad) approaches a constant... 
T0_zz['g'] *= H_func(z_de, z_match - dwT, dwT) 
T0_zz.change_scales(1)

# integrate T0_zz to get T0_z
#T0_z['g'] = antiderivative(T0_zz, T0_z, zbasis, coords, 'z', 0, -grad_ad)
logger.info("Solving LBVP to get antiderivative")
F = dist.Field(name = 'F', bases = zbasis)
tau_z = dist.Field(name = 'tau_z')
T0_zz.change_scales(1)
problem_z = d3.LBVP([F, tau_z], namespace = locals())
problem_z.add_equation("dz(F) + lift(tau_z) = T0_zz")
problem_z.add_equation("F(z = 0) = -grad_ad")    
solver_z = problem_z.build_solver()
solver_z.solve()
F.change_scales(1)
T0_z.change_scales(1)
T0_z['g'] = F['g']

# ensure T0_zz approaches grad_rad at z = Lz (top of domain), where grad_rad = flux_of_z / k (horizontally avgd)
delta_rad = -grad_rad_top + grad_ad
delta_sim = reducer.global_max(T0_z(z = Lz).evaluate()['g']) + grad_ad
T0_zz['g'] *= delta_rad/delta_sim # multiplies by about 0.07519

# recalculate T0_z
#T0_z['g'] = antiderivative(T0_zz, T0_z, zbasis, coords, 'z', 0, -grad_ad)
logger.info("Solving LBVP to get antiderivative")
F = dist.Field(name = 'F', bases = zbasis)
tau_z = dist.Field(name = 'tau_z')
problem_z = d3.LBVP([F, tau_z], namespace = locals())
problem_z.add_equation("dz(F) + lift(tau_z) = T0_zz")
problem_z.add_equation("F(z = 0) = -grad_ad")    
solver_z = problem_z.build_solver()
solver_z.solve()
F.change_scales(1)
T0_z.change_scales(1)
T0_z['g'] = F['g']

# integrate T0_z to get T0
#T0['g'] = antiderivative(T0_z, T0, zbasis, coords, 'z', Lz, 1e0)
logger.info("Solving LBVP to get antiderivative")
F = dist.Field(name = 'F', bases = zbasis)
tau_z = dist.Field(name = 'tau_z')
T0_z.change_scales(1)
problem_z = d3.LBVP([F, tau_z], namespace = locals())
problem_z.add_equation("dz(F) + lift(tau_z) = T0_z")
problem_z.add_equation("F(z = Lz) = 1")    
solver_z = problem_z.build_solver()
solver_z.solve()
F.change_scales(1)
T0.change_scales(1)
T0['g'] = F['g']

Q.change_scales(1)
k.change_scales(1)
k_z.change_scales(1)
T0_z.change_scales(1)
T0_zz.change_scales(1)

# check that heating and cooling cancel out
fH = dist.Field(name = 'fH', bases = zbasis)
fH2 = dist.Field(name = 'fH2', bases = zbasis)
fH['g'] = Q['g'] +  k_z['g']*T0_z['g'] + k['g']*T0_zz['g']
#fH2['g'] = antiderivative(fH, fH2, zbasis, coords, 'z', 0, 0)
logger.info("Solving LBVP to get antiderivative")
F = dist.Field(name = 'F', bases = zbasis)
tau_z = dist.Field(name = 'tau_z')
fH.change_scales(1)
problem_z = d3.LBVP([F, tau_z], namespace = locals())
problem_z.add_equation("dz(F) + lift(tau_z) = fH")
problem_z.add_equation("F(z = 0) = 0")    
solver_z = problem_z.build_solver()
solver_z.solve()
F.change_scales(1)
fH2.change_scales(1)
fH2['g'] = F['g']
logger.info('right(integ(heating - cooling)): {:.3e}'.format(reducer.global_max(fH2(z = Lz).evaluate()['g'])))

### Problem ###
problem = d3.IVP([u, T1, p, tau_u1, tau_u2, tau_T11, tau_T12, tau_p], namespace = locals())
problem.add_equation("trace(grad_u) + tau_p = 0") # continuity
problem.add_equation("dt(u) - Rinv*div(grad_u) + grad(p) - T1*ez + lift(tau_u2) = - u@grad(u)") # momentum
problem.add_equation("dt(T1) - PrRinv*div(grad_T1) + dz(lift(tau_T11)) + lift(tau_T12) = - u@grad(T1) - u@ez * (T0_z - T_ad)", condition='nx!=0') # energy
#problem.add_equation("dt(T1) + dz(lift(tau_T11)) + lift(tau_T12) - k_z*dz(T1) - k*dz(dz(T1)) = - u@grad(T1) - u@ez * (T0_z - T_ad) + Q + k_z*T0_z + k*T0_zz", condition='nx==0')
problem.add_equation("dt(T1) + dz(lift(tau_T11)) + lift(tau_T12) - dz(k*T1_z) = - u@grad(T1) - u@ez * (T0_z - T_ad) + Q + k_z*T0_z + k*T0_zz", condition='nx==0')
problem.add_equation("u(z = 0) = 0") # no slip
problem.add_equation("u(z = Lz) = 0") # no slip
problem.add_equation("dz(T1)(z = 0) = 0") # no flux
problem.add_equation("T1(z = Lz) = 0") # fixed temp
problem.add_equation("integ(p) = 0") # pressure gauge

### Solver ###
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

### Initial conditions ###
T1.fill_random('g', seed = 101, distribution = 'normal', scale = 1e-3)
T1['g'] *= np.sin(2*np.pi*z/Lz)

### Analysis tasks ToDo ###
#snapshots = solver.evaluator.add_file_handler('snapshots', iter=1)
snapshots = solver.evaluator.add_file_handler('snapshots_serial', sim_dt = 0.25)
snapshots.add_task(T0 + T1 - d3.integ(T0 + T1, coords['x']), name='T-T_bar')
snapshots.add_task(d3.integ(T0 + T1, coords['x']), name='T_bar') # should be 1d, but want included with this data
snapshots.add_task(T1, name='temp')
#snapshots.add_task(T1/(d3.integ(np.abs(T1), coords['x'])), name='temp-anom')
snapshots.add_task(T1/np.sqrt(d3.integ(T1*T1)), name='temp-rms')
snapshots.add_task(ex@u, name='u_x')
snapshots.add_task(ez@u, name='u_z')
snapshots.add_task(p, name='pressure')

#T = T0 + T1
#Tbar = d3.integ(T, coords['x'])
#Tp = T - Tbar
#snapshots.add_task(Tp, name='Tp')
#snapshots.add_task(Tp/(d3.integ(np.abs(Tp), coords['x'])), name='Tp-anom')

# normalize by an rms of T1

profiles = solver.evaluator.add_file_handler('profiles_serial', sim_dt = 1.)
# these should all be time stationary, but double-checking for sanity
profiles.add_task(Q, name='heating')
profiles.add_task(k_z*T0_z + k*T0_zz, name='cooling')
profiles.add_task(flux_of_z, name='Qflux')
profiles.add_task(k, name='kprof')
profiles.add_task(T_rad_z0, name='T_rad_z0prof')
profiles.add_task(T0, name='T0')
profiles.add_task(T0_z, name='T0_z')
profiles.add_task(T0_zz, name='T0_zz')
profiles.add_task(k_z*T0_z, name='k_zT0_z')
profiles.add_task(k*T0_zz, name='kT0_zz')
profiles.add_task(fH, name='fHprof')
profiles.add_task(fH2, name='fH2prof')
profiles.add_task(T0_z - T_ad, name='T0_z-T_ad')

#profiles.add_task(T0_zz_init, name='T0_zz_init')
#profiles.add_task(T0_zz_H, name='T0_zz_H')
#profiles.add_task(T0_zz_del, name='T0_zz_del')

# checking terms comprising dz(T_rad_z0)
profiles.add_task((-flux_of_z/(k**2))*k_z, name='Qfluxk2k_z')
profiles.add_task(-flux_of_z/(k**2), name='Qfluxk2')
profiles.add_task(k_z, name='k_zprof')
profiles.add_task(Q/k, name='Qzprof')
profiles.add_task(k**(-1), name='k1')
profiles.add_task(k**(-2), name='k2')


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
#flow.add_property(-Rinv*d3.div(grad_u), name='lapu')
#flow.add_property(ex@d3.grad(p), name='gradpx')
#flow.add_property(ez@d3.grad(p), name='gradpz')
#flow.add_property(-T1*ez, name='T1ez')
#flow.add_property(lift(tau_u2), name='lift')
#flow.add_property(d3.Average(max_brunt * (CZ_mask * np.sqrt(u@u))**(-1)), name = 'S')

logger.info("Reached main loop, stopping here")
### Main loop ###
#try:
#    logger.info('Starting main loop')
#    while solver.proceed:
#        t_step = CFL.compute_timestep()
#        solver.step(t_step)
#        if (solver.iteration - 1) % 20 == 0:
#            max_Re = flow.max('Re')
#            avg_Re = flow.grid_average('Re')
#            avg_T1_z = flow.max('T1_z')
#            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, avg(Re)=%f, integ(dz(T1))=%f' %(solver.iteration, solver.sim_time, t_step, max_Re, avg_Re, avg_T1_z))
#except:
#    logger.error('Exception raised, triggering end of main loop.')
#    raise
#finally:
#    solver.log_stats()
