"""
Dedalus (d3) script to run a 2d horizontally-periodic simulation 
with convection and stable stratification in the Boussinesq limit,
following the non-dimensionalization given in Anders et al. (2022). 
This script implements a discontinuous (erf) convective flux profile. 

This script is not optimized--once we have this as a working example,
we will likely change our formulation to reduce the number of fields 
and transforms to the extent feasible. 
"""
import sys
import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
from scipy.special import erf

from dedalus.extras import flow_tools

restart = sys.argv[1]

### Parameters ###
dtype = np.float64
Lx, Lz = 1, 2
Nx, Nz = 64, 256
dealias = 3/2
timestepper = d3.RK222
safety = 0.3 # CFL prefactor
stop_sim_time = 1e2

# Prognostic
P = 1e0 # Penetration parameter
S = 1e3 # Stiffness
Pr = 1e0 # Prandtl 
R = 4e2 # Reynolds (freefall)
zeta = 1e-3 # also called mu, ratio between adiabatic flux at bottom to the internal heating flux from Q

Pinv = 1/P
Rinv = 1/R
PRinv = Pinv*Rinv

# Model values
L_Q = 1e-1 
dH = 2e-1 # width of internal heating layer: z in [L_Q, L_Q + dH]
Qmag = 1e0 # magnitude of Q inside heating layer
dwQ = 2e-2 # scale of Q jump
L_CZ = 1e0 # where to center jump in flux
dwk = 7.5e-2 # scale of k jump
dwT = 5e-2 # scale of T0 jump

F_bot = zeta * Qmag * dH

#delta_pred = 0.

### Function definitions ###
def H_func(z, z0, dw):
    return (1 + erf((z - z0)/dw))/2

def H_func_c(z, z0, dw):
    return (1 - erf((z - z0)/dw))/2

def Q_func(z, z0, dw, Qmag, dH):
    return Qmag * H_func(z, z0, dw) * H_func_c(z, z0 + dH, dw)

def k_func(z, z0, dw, k_CZ, dk):
    return k_CZ + dk * H_func(z, z0, dw)

def grad_rad_func(f_g, k_g):
    return f_g / k_g

# F = integ(f)
def antiderivative(f, F, b, c, cstr, c0, F0, b1):
    logger.info("Performing LBVP to get antiderivative")
    d = lambda A: d3.Differentiate(A, c[cstr])
    tau_1 = dist.Field(name = 'tau_1', bases = b1)
    lb = b.derivative_basis(1)
    lift_b = lambda A: d3.Lift(A, lb, -1)

    problem_F = d3.LBVP([F, tau_1], namespace = locals())
    problem_F.add_equation("d(F) + lift_b(tau_1) = f")
    problem_F.add_equation("F("+cstr+"="+str(c0)+") = F0")

    solver_F = problem_F.build_solver()
    solver_F.solve()

    return F['g']

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
T0 = dist.Field(name = 'T0', bases = (xbasis, zbasis)) 
Q = dist.Field(name = 'Q', bases = (xbasis, zbasis))
k = dist.Field(name = 'k', bases = (xbasis, zbasis))

grad_rad = dist.Field(name = 'grad_rad', bases = (xbasis, zbasis))
Q_int = dist.Field(name = 'Q_int', bases = (xbasis, zbasis))
N2 = dist.Field(name = 'N2', bases = (xbasis, zbasis))
k_z = dist.Field(name = 'k_z', bases = (xbasis, zbasis))

CZ_mask = dist.Field(name = 'CZ_mask', bases = (xbasis, zbasis))
CZ_mask['g'] = H_func(z, 0.2, 0.05) * H_func_c(z, L_CZ, 0.05)

T0_z = dist.Field(name = 'T0_z', bases = (xbasis, zbasis))
T0_zz = dist.Field(name = 'T0_zz', bases = (xbasis, zbasis))

# substitutions
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_T1 = d3.grad(T1) + ez*lift(tau_T11)

#T1_z_bar = d3.Component(dz(T1))
#T1_z_bar = d3.Average(dz(T1), coords['x'])
#T1_zz_bar = d3.Average(dz(dz(T1)), coords['x'])

#T1_z_bar = d3.Average(dz(T1), 'x')
#T1_zz_bar = d3.Average(dz(dz(T1)), 'x')

#T1_z_bar = d3.integ(dz(T1), 'x')
#T1_zz_bar = d3.integ(dz(dz(T1)), 'x')

#k_z = dz(k)

#print(T1_z_bar.evaluate()['g'].shape)

#T0_z = dz(T0)
#T0_zz = dz(T0_z)

k_RZ = dH / P / S
k_CZ = k_RZ * zeta / (1 + zeta + Pinv)
k_ad = k_RZ * (1 + zeta) / (1 + zeta + Pinv)
dk = k_RZ - k_CZ

grad_ad = (Qmag * S * P) * (1 + zeta + Pinv)
grad_rad_top = (Qmag * S * P) * (1 + zeta)
delta_grad = grad_ad - grad_rad_top

#T1_z = dist.Field(name = 'T1_z', bases = (xbasis, zbasis)) # currently using for IC, but could use more heavily

# set fields on grid
Q['g'] = Q_func(z, L_Q, dwQ, Qmag, dH)
k['g'] = k_func(z, L_CZ, dwk, k_CZ, dk)

Q_int['g'] = antiderivative(Q, Q_int, zbasis, coords, 'z', 0, F_bot, xbasis)
#tau_Q = dist.Field(name = 'tau_Q', bases = xbasis)
#problem_Q = d3.LBVP([Q_int, tau_Q], namespace = locals())
#problem_Q.add_equation("dz(Q_int) + lift(tau_Q) = Q")
#problem_Q.add_equation("Q_int(z = 0) = F_bot")
#solver_Q = problem_Q.build_solver()
#solver_Q.solve()

grad_rad['g'] = grad_rad_func(Q_int['g'], k['g'])
N2['g'] = -grad_rad['g'] + grad_ad
max_brunt = reducer.global_max(N2['g'])

k_z.change_scales(dealias)
k_z['g'] = dz(k).evaluate()['g']
k_z.change_scales(1)
flux = reducer.global_min(Q_int(z = L_CZ)['g'])

zs = np.linspace(0, Lz, 1000)
k_match = k_func(zs, L_CZ, dwk, k_CZ, dk)
grad_rad_match = grad_rad_func(flux, k_match)
z_match = np.interp(-grad_ad, -grad_rad_match, zs)

T0_zz.change_scales(dealias)
T0_zz['g'] = dz(-grad_rad).evaluate()['g']
T0_zz['g'] *= H_func(z_de, z_match - dwT, dwT) # or I could change scales back first and then use z?
T0_zz.change_scales(1)
T0_z['g'] = antiderivative(T0_zz, T0_z, zbasis, coords, 'z', 0, -grad_ad, xbasis)

delta_sim = reducer.global_max(T0_z(z = Lz)['g']) + grad_ad
T0_zz['g'] *= delta_grad/delta_sim
T0_z['g'] = antiderivative(T0_zz, T0_z, zbasis, coords, 'z', 0, -grad_ad, xbasis)
T0['g'] = antiderivative(T0_z, T0, zbasis, coords, 'z', Lz, 1e0, xbasis)

Q.change_scales(1)
k.change_scales(1)
k_z.change_scales(1)
T0_z.change_scales(1)
T0_zz.change_scales(1)

# check that heating and cooling cancel out
fH = dist.Field(name = 'fH', bases = (xbasis, zbasis))
fH2 = dist.Field(name = 'fH2', bases = (xbasis, zbasis))
fH['g'] = Q['g'] +  k_z['g']*T0_z['g'] + k['g']*T0_zz['g']
fH2['g'] = antiderivative(fH, fH2, zbasis, coords, 'z', 0, 0, xbasis)
logger.info('right(integ(heating - cooling)): {:.3e}'.format(reducer.global_max(fH2(z = Lz)['g'])))

### Problem ###
problem = d3.IVP([u, T1, p, tau_u1, tau_u2, tau_T11, tau_T12, tau_p], namespace = locals())
problem.add_equation("trace(grad_u) + tau_p = 0") # continuity
problem.add_equation("dt(u) - Rinv*div(grad_u) + grad(p) - T1*ez + lift(tau_u2) = - u@grad(u)") # momentum
problem.add_equation("dt(T1) - PRinv*div(grad_T1) + lift(tau_T12) = - u@grad(T1) - u@ez * (grad_ad + T0_z)", condition='nx!=0') # energy
problem.add_equation("dt(T1) - PRinv*div(grad_T1) + lift(tau_T12) = - u@grad(T1) + k_z*dz(T1) + k*dz(dz(T1)) - u@ez * (grad_ad + T0_z) + Q + k_z*T0_z + k*T0_zz", condition='nx==0') # energy
# does the laplacian term contribute when nx = 0? 
# maybe I am forced to evaluate - k_z*dz(T1) - k*dz(dz(T1) on rhs (explicitly) if I want to parallelize?
# > is this because k and k_z act like non-constant coefficients in z? 
problem.add_equation("u(z = 0) = 0") # no slip
problem.add_equation("u(z = Lz) = 0") # no slip
problem.add_equation("dz(T1)(z = 0) = 0") # no flux
problem.add_equation("T1(z = Lz) = 0") # fixed temp
problem.add_equation("integ(p) = 0") # pressure gauge

### Solver ###
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

### Initial conditions ###
if not restart: 
   
   # predictive
   #T_z = -grad_ad + (grad_ad - grad_rad['g']) * H_func(z, L_CZ + delta_pred, dwT)
   #T1_z['g'] = T_z - T0_z['g']
   #T1['g'] = antiderivative(T1_z, T1, zbasis, coords, 'z', Lz, 0, xbasis)

   T1.fill_random('g', seed = 101, distribution = 'normal', scale = 1e-3)
   #T1['g'] *= np.sin(np.pi*z)
   #u.fill_random('g', seed = 101, distribution = 'normal', scale = 1e-3) # was trying to break symmetry... but it's not helping
#else: # ToDo


### Analysis tasks ToDo ###
slices = solver.evaluator.add_file_handler('slices', sim_dt=0.25, max_writes=50)
slices.add_task(T1, name='temp')
slices.add_task(ez@u, name='u_z')

#profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.25, max_writes=50)

### Flow tools ###

t_ff = 1/np.sqrt(Qmag)
t_N2 = np.sqrt(1/max_brunt)
max_t_step = np.min((0.5*t_ff, t_N2))
t_step = max_t_step

CFL = d3.CFL(solver, initial_dt=t_step, cadence=1, safety=safety, threshold=0.1,
             max_change=1.5, min_change=0.25, max_dt=max_t_step)
CFL.add_velocity(u)

flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(R*np.sqrt(u@u), name = 'Re')
flow.add_property(d3.integ(dz(T1)), name = 'T1_z')
flow.add_property(-Rinv*d3.div(grad_u), name='lapu')
flow.add_property(ex@d3.grad(p1), name='gradp1x')
flow.add_property(ez@d3.grad(p1), name='gradp1z')
flow.add_property(-T1*ez, name='T1ez')
flow.add_property(lift(tau_u2), name='lift')
#flow.add_property(d3.Average(max_brunt * (CZ_mask * np.sqrt(u@u))**(-1)), name = 'S')
# ToDo: P tracker?

### Main loop ###
# ToDo: (add more fancy tracking?)


try:
    logger.info('Starting main loop')
    while solver.proceed:
        #print(T1['g'].min(), T1['g'].max())
        t_step = CFL.compute_timestep()
        solver.step(t_step)
        if (solver.iteration - 1) % 20 == 0:
            max_Re = flow.max('Re')
            avg_Re = flow.grid_average('Re')
            avg_T1_z = flow.max('T1_z')
            print('-Rinv*lapu', flow.min('lapu'), flow.max('lapu'), flow.grid_average('lapu'))
            print('gradp1x', flow.min('gradp1x'), flow.max('gradp1x'), flow.grid_average('gradp1x'))
            print('gradp1z', flow.min('gradp1z'), flow.max('gradp1z'), flow.grid_average('gradp1z'))
            print('-T1ez', flow.min('T1ez'), flow.max('T1ez'), flow.grid_average('T1ez'))
            print('lift', flow.min('lift'), flow.max('lift'), flow.grid_average('lift'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, avg(Re)=%f, integ(dz(T1))=%f' %(solver.iteration, solver.sim_time, t_step, max_Re, avg_Re, avg_T1_z))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
