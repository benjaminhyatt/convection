"""
Dedalus (d3) script to run a 2d horizontally-periodic simulation 
with convection and stable stratification in the Boussinesq limit,
following the non-dimensionalization given in Anders et al. (2022). 
This script implements a discontinuous (erf) convective flux profile. 

Usage:
"""
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

### Parameters ###
dtype = np.float64
Lx, Lz = 1, 2
Nx, Nz = 64, 256
dealias = 3/2
timestepper = d3.RK222
sim_stop_time = 1e2

# Prognostic
P = 1e0 # Penetration parameter
S = 1e3 # Stiffness
Pr = 1e0 # Prandtl 
R = 4e2 # Reynolds (freefall)
zeta = 1e-3 # also called mu, ratio between adiabatic flux at bottom to the internal heating flux from Q

# Model values
dH = 2e-1 # width of internal heating layer: z in [0.1, 0.1 + dH]
Qmag = 1e0 # magnitude of Q inside heating layer

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

# substitutions
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_T1 = d3.grad(T1) + ez*lift(tau_T11)

grad_T0 = d3.grad(T0) # we could also write out ourselves since it simplifies

grad_ad = (Qmag * S * P) * (1 + zeta + 1/P) 


### Problem ###

### Solver ###

### Initial conditions ###

### Analysis tasks ###

### Flow tools ###

### Main loop ###