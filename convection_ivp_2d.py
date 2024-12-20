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


### Parameters ###
dtype = np.float64
Lx, Lz = 1, 2
Nx, Nz = 64, 256
dealias = 3/2
timestepper = d3.RK222
safety = 0.5 # CFL prefactor
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
dw = 5e-2 # scale of other jumps -- may not need?

F_bot = zeta * Qmag * dH

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

# F = integ(f) over basis b, coord c, with F(c0) = F0
def antiderivative(f, F, b, cstr, c0, F0):
    tau_1 = d3.Field(name = 'tau_1', bases = b)
    lift_b = lambda A: d3.Lift(A, b, -1)

    problem_F = d3.LBVP([F, tau_1], namespace = locals())
    problem_F.add_equation("d"+cstr+"(F) + lift_b(tau_1) = f")
    problem_F.add_equation("F("+c0+") = F0")

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
dz = lambda A: d3.Differentate(A, coords['z'])
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

### Fields and substitutions ###

# time-varying
u = dist.VectorField(coords, name = 'u', bases = (xbasis, zbasis))
T1 = dist.Field(name = 'T1', bases = (xbasis, zbasis))
p1 = dist.Field(name = 'p1', bases = (xbasis, zbasis))

# tau
tau_u1 = dist.VectorField(coords, name = 'tau_u1', bases = xbasis)
tau_u2 = dist.VectorField(coords, name = 'tau_u2', bases = xbasis)
tau_T11 = dist.Field(name = 'tau_T11', bases = xbasis)
tau_T12 = dist.Field(name = 'tau_T12', bases = xbasis)
tau_p = dist.Field(name = 'tau_p')

# stationary
T0 = dist.Field(name = 'T0', bases = (xbasis, zbasis)) 
p0 = dist.Field(name = 'p0', bases = (xbasis, zbasis)) # related to T0
Q = dist.Field(name = 'Q', bases = (xbasis, zbasis))
k = dist.Field(name = 'k', bases = (xbasis, zbasis))

grad_rad = dist.Field(name = 'grad_rad', bases = (xbasis, zbasis))
Q_int = dist.Field(name = 'Q_int', bases = (zbasis))
N2 = dist.Field(name = 'N2', bases = (xbasis, zbasis))

# substitutions
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_T1 = d3.grad(T1) + ez*lift(tau_T11)

T1_z_bar = d3.Average(dz(T1), coords['x'])
T1_zz_bar = d3.Average(dz(dz(T1)), coords['x'])
k_z = dz(k)

T0_z = dz(T0)
T0_zz = dz(T0_z)

k_RZ = dH / P / S
k_CZ = k_RZ * zeta / (1 + zeta + Pinv)
k_ad = k_RZ * (1 + zeta) / (1 + zeta + Pinv)
dk = k_RZ - k_CZ

grad_ad = (Qmag * S * P) * (1 + zeta + Pinv)

# set fields on grid
Q['g'] = Q_func(z, L_Q, dwQ, Qmag, dH)
k['g'] = k_func(z, L_CZ, dwk, k_CZ, dk)

T0['g'] = 

Q_int['g'] = antiderivative(Q, Q_int, zbasis, 'z', 0, F_bot)
grad_rad['g'] = grad_rad_func(Q_int['g'], k['g'])
N2['g'] = grad_rad['g'] - grad_ad

### Problem ###
problem = d3.IVP([u, T1, p1, tau_u1, tau_u2, tau_T11, tau_T12, tau_p], namespace = locals())
problem.add_equation("trace(grad_u) + tau_p = 0") # continuity
problem.add_equation("dt(u) - Rinv*div(grad_u) + grad(p1) - T1*ez + lift(tau_u2) = - u@grad(u)") # momentum
problem.add_equation("dt(T1) - PRinv*div(grad_T1) - k_z*T1_z_bar - k*T1_zz_bar + lift(tau_T12) = - u@grad(T1) + Q - grad_ad * u@ez + k_z*T0_z + k*T0_zz") # energy
problem.add_equation("u(z=0) = 0") # no slip
problem.add_equation("u(z=Lz) = 0") # no slip
problem.add_equation("dz(T1)(z=0) = 0") # fixed flux
problem.add_equation("T1(z=Lz) = 0") # fixed temp
problem.add_equation("integ(p) = 0") # pressure gauge

### Solver ###
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

### Initial conditions ###


### Analysis tasks ###


### Flow tools ###
t_ff = 1/np.sqrt(Qmag)
t_N2 = np.sqrt(1/np.max(N2))
max_t_step = np.min((0.5*t_ff, t_N2))
t_step = max_t_step

CFL = d3.CFL(solver, initial_dt=t_step, cadence=1, safety=safety, threshold=0.1,
             max_change=1.5, min_change=0.25, max_dt=max_t_step)
CFL.add_velocity(u)

### Main loop ###