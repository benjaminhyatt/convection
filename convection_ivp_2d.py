"""
Dedalus (d3) script to run a 2d horizontally-periodic simulation 
with convection and stable stratification in the Boussinesq limit,
following the non-dimensionalization given in Anders et al. (2022). 

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

P = 1e0 # Penetration parameter
S = 1e3 # Stiffness
Pr = 1e0 # Prandtl 
R = 4e2 # Reynolds (freefall)

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


### Fields ###
u = dist.VectorField(coords, name = 'u', bases = (xbasis, zbasis))
T0 = dist.Field(name = 'T0', bases = (xbasis, zbasis)) # time-stationary bg, const in x
T1 = dist.Field(name = 'T1', bases = (xbasis, zbasis))
p = dist.Field(name = 'p', bases = (xbasis, zbasis))

tau_u1 = dist.VectorField(coords, name = 'tau_u1', bases = xbasis)
tau_u2 = dist.VectorField(coords, name = 'tau_u2', bases = xbasis)
tau_T11 = dist.Field(name = 'tau_T11', bases = xbasis)
tau_T12 = dist.Field(name = 'tau_T12', bases = xbasis)
tau_p = dist.Field(name = 'tau_p')

### Substitutions ###
grad_u = d3.grad(u) + ez*lift(tau_u1)

T0_z = dz(T0)
T0_zz = dz(T0_z)

T = T0 + T1




### Problem ###

### Solver ###

### Initial conditions ###

### Analysis tasks ###

### Flow tools ###

### Main loop ###