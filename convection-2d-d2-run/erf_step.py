"""
Dedalus script for a two-layer, Boussinesq simulation.
The bottom of the domain is at z = 0.
The lower part of the domain is stable; the domain is Schwarzschild stable above z >~ 1.

There are 6 control parameters:
    Re      - The approximate reynolds number = (u / diffusivity) of the evolved flows
    Pr      - The Prandtl number = (viscous diffusivity / thermal diffusivity)
    P       - The penetration parameter; When P >> 1 there is lots of convective penetration (of order P); when P -> 0 there is none.
    S       - The stiffness: the characteristic ratio of N^2 (above the penetration region) compared to the square convective frequency.
    zeta    - The fraction of the convective flux carried by the adiabatic gradient at z = 0 (below the heating layer)
    Lz      - The height of the box
    aspect  - The aspect ratio (Lx = aspect * Lz)

Usage:
    erf_step.py [options] 
    erf_step.py <config> [options] 

Options:
    --Re=<Reynolds>            Freefall reynolds number [default: 1e2]
    --Pr=<Prandtl>             Prandtl number = nu/kappa [default: 0.5]
    --P=<penetration>          ratio of CZ convective flux / RZ convective flux [default: 1]
    --S=<stiffness>            The stiffness [default: 1e2]
    --zeta=<frac>              Fbot = zeta * F_conv [default: 1e-3]
    --Lz=<L>                   Depth of domain [default: 2]
    --aspect=<aspect>          Aspect ratio of domain [default: 2]
    --L_cz=<L>                 Height of cz-rz erf step [default: 1]

    --nz=<nz>                  Vertical resolution   [default: 256]
    --nx=<nx>                  Horizontal resolution [default: 64]
    --safety=<s>               CFL safety factor (timestepper: RK443) [default: 0.75]

    --run_time_wall=<time>     Run time, in hours [default: 119.5]
    --run_time_ff=<time>       Run time, in freefall times [default: 2e3]

    --restart=<restart_file>   Restart from checkpoint
    --seed=<seed>              RNG seed for initial conditoins [default: 42]

    --label=<label>            Optional additional case name label
    --root_dir=<dir>           Root directory for output [default: ./]

    --adiabatic_IC             If flagged, set the background profile as a pure adiabat (not thermal equilibrium in RZ)
    --predictive=<delta>       A guess for delta_P the penetration depth. The initial state grad(T) will be an erf from grad(T_ad) to grad(T_rad) centered at L_cz + delta_P
    --T_iters=<N>              Number of times to iterate background profile before pure timestepping [default: 10]
    --plot_model               If flagged, create and plt.show() some plots of the 1D atmospheric structure.

"""
import logging
import os
import sys
import time
from collections import OrderedDict
from configparser import ConfigParser
from pathlib import Path
import h5py
import numpy as np
from docopt import docopt
from mpi4py import MPI
from scipy.special import erf

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools  import post

logger = logging.getLogger(__name__)

def filter_field(field, frac=0.25):
    """
    Filter a field in coefficient space by cutting off all coefficient above
    a given threshold.  This is accomplished by changing the scale of a field,
    forcing it into coefficient space at that small scale, then coming back to
    the original scale.

    Inputs:
        field   - The dedalus field to filter
        frac    - The fraction of coefficients to KEEP POWER IN.  If frac=0.25,
                    The upper 75% of coefficients are set to 0.
    """
    dom = field.domain
    logger.info("filtering field {} with frac={} using a set-scales approach".format(field.name,frac))
    orig_scale = field.scales
    field.set_scales(frac, keep_data=True)
    field['c']
    field['g']
    field.set_scales(orig_scale, keep_data=True)

def global_noise(domain, seed=101, **kwargs):
    """
    Create a field fielled with random noise of order 1.  Modify seed to
    get varying noise, keep seed the same to directly compare runs.
    """
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
    slices = domain.dist.grid_layout.slices(scales=domain.dealias)
    rand = np.random.RandomState(seed=seed)
    noise = rand.standard_normal(gshape)[slices]

    # filter in k-space
    noise_field = domain.new_field()
    noise_field.set_scales(domain.dealias, keep_data=False)
    noise_field['g'] = noise
    return noise_field

def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

def set_equations(problem):
    kx_0  = "(nx == 0)"
    kx_n0 = "(nx != 0)"
    equations = ( (True, "True", "T1_z - dz(T1) = 0"),
                  (True, "True", "ωy - dz(u) + dx(w) = 0"),
                  (True, kx_n0,  "dx(u) + dy(v) + dz(w) = 0"), #Incompressibility
                  (True, kx_0,   "p = 0"), #Incompressibility
                  (True, "True", "dt(u) + (dy(ωz) - dz(ωy))/Re0  + dx(p)        = v*ωz - w*ωy "), #momentum-x
                  (True, kx_n0,  "dt(w) + (dx(ωy) - dy(ωx))/Re0  + dz(p) - T1   = u*ωy - v*ωx "), #momentum-z
                  (True, kx_0,   "w = 0"), #momentum-z
                  (True, kx_n0, "dt(T1) - Lap(T1, T1_z)/Pe0  = -UdotGrad(T1, T1_z) - w*(T0_z - T_ad_z)"), #energy eqn
                  (True, kx_0,  "dt(T1) - dz(k0*T1_z)        = -UdotGrad(T1, T1_z) - w*(T0_z - T_ad_z) + (Q + dz(k0)*T0_z + k0*T0_zz)"), #energy eqn
                )
    for solve, cond, eqn in equations:
        if solve:
            logger.info('solving eqn {} under condition {}'.format(eqn, cond))
            problem.add_equation(eqn, condition=cond)

    boundaries = ( (True, " left(T1_z) = 0", "True"),
                   (True, "right(T1) = 0", "True"),
                   (True, " left(u) = 0", "True"),
                   (True, "right(u) = 0", "True"),
                   (True, " left(w) = 0", kx_n0),
                   (True, "right(w) = 0", kx_n0),
                 )
    for solve, bc, cond in boundaries:
        if solve: 
            logger.info('solving bc {} under condition {}'.format(bc, cond))
            problem.add_bc(bc, condition=cond)

    return problem

def set_subs(problem):
    # Set up useful algebra / output substitutions
    problem.substitutions['Lap(A, A_z)']                   = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
    problem.substitutions['UdotGrad(A, A_z)']              = '(u*dx(A) + v*dy(A) + w*A_z)'
    problem.substitutions['GradAdotGradB(A, B, A_z, B_z)'] = '(dx(A)*dx(B) + dy(A)*dy(B) + A_z*B_z)'
    problem.substitutions['dy(A)'] = '0'
    problem.substitutions['ωx'] = '0'
    problem.substitutions['ωz'] = '0'
    problem.substitutions['v'] = '0'
    problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
    problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'
    problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'
    problem.substitutions['enstrophy'] = '(ωx**2 + ωy**2 + ωz**2)'
    problem.substitutions['vel_rms']   = 'sqrt(u**2 + v**2 + w**2)'
    problem.substitutions['Re']        = '(Re0*vel_rms)'
    problem.substitutions['Pe']        = '(Pe0*vel_rms)'
    problem.substitutions['T_z']       = '(T0_z + T1_z)'
    problem.substitutions['T']         = '(T0 + T1)'


    problem.substitutions['bruntN2_structure']   = 'T_z - T_ad_z'
    problem.substitutions['bruntN2']             = 'bruntN2_structure'

    #Fluxes
    problem.substitutions['F_rad']       = '-k0*T_z'
    problem.substitutions['T_rad_z']     = '-flux_of_z/k0'
    problem.substitutions['T_rad_z_IH']  = '-right(flux_of_z)/k0'
    problem.substitutions['F_conv']      = 'w*T'
    problem.substitutions['tot_flux']    = '(F_conv + F_rad)'
    return problem

def initialize_output(solver, data_dir, mode='overwrite', output_dt=2, iter=np.inf):
    analysis_tasks = OrderedDict()
    slices = solver.evaluator.add_file_handler(data_dir+'slices', sim_dt=output_dt, max_writes=40, mode=mode, iter=iter)
    slices.add_task("T1")
    slices.add_task("u")
    slices.add_task("w")
    slices.add_task("enstrophy")
    analysis_tasks['slices'] = slices

    profiles = solver.evaluator.add_file_handler(data_dir+'profiles', sim_dt=output_dt, max_writes=40, mode=mode)
    profiles.add_task("plane_avg(T)", name='T')
    profiles.add_task("plane_avg(T_z)", name='T_z')
    profiles.add_task("plane_avg(T1)", name='T1')
    profiles.add_task("plane_avg(sqrt((T1 - plane_avg(T1))**2))", name='T1_fluc')
    profiles.add_task("plane_avg(T1_z)", name='T1_z')
    profiles.add_task("plane_avg(u)", name='u')
    profiles.add_task("plane_avg(w)", name='w')
    profiles.add_task("plane_avg(vel_rms)", name='vel_rms')
    profiles.add_task("plane_avg(sqrt((v*ωz - w*ωy)**2 + (u*ωy - v*ωx)**2))", name='advection')
    profiles.add_task("plane_avg(enstrophy)", name="enstrophy")
    profiles.add_task("plane_avg(bruntN2)", name="bruntN2")
    profiles.add_task("plane_avg(bruntN2_structure)", name="bruntN2_structure")
    profiles.add_task("plane_avg(flux_of_z)", name="flux_of_z")
    profiles.add_task("plane_avg((Q + dz(k0)*T0_z + k0*T0_zz))", name="effective_heating")
    profiles.add_task("plane_avg(T_rad_z)", name="T_rad_z")
    profiles.add_task("plane_avg(T_rad_z)", name="T_rad_z_IH")
    profiles.add_task("plane_avg(T_ad_z)", name="T_ad_z")
    profiles.add_task("plane_avg(F_rad)", name="F_rad")
    profiles.add_task("plane_avg(F_conv)", name="F_conv")
    profiles.add_task("plane_avg(k0)", name="k0")
    profiles.add_task("plane_avg(dz(k0*T1_z))", name="heat_fluc_rad")
    profiles.add_task("plane_avg(-dz(F_conv))", name="heat_fluc_conv")
    analysis_tasks['profiles'] = profiles


    scalars = solver.evaluator.add_file_handler(data_dir+'scalars', sim_dt=output_dt*5, max_writes=np.inf, mode=mode)
    scalars.add_task("vol_avg(cz_mask*vel_rms**2)/vol_avg(cz_mask)", name="cz_vel_squared")
    scalars.add_task("vol_avg((1-cz_mask)*bruntN2)/vol_avg(1-cz_mask)", name="rz_brunt_squared")
    scalars.add_task("vol_avg(Re0*vel_rms)", name='mean_Re')
    scalars.add_task("vol_avg(T1_z)", name='mean_T1_z')    
    analysis_tasks['scalars'] = scalars

    checkpoint_min = 60
    checkpoint = solver.evaluator.add_file_handler(data_dir+'checkpoint', wall_dt=checkpoint_min*60, sim_dt=np.inf, iter=np.inf, max_writes=1, mode=mode)
    checkpoint.add_system(solver.state, layout = 'c')
    analysis_tasks['checkpoint'] = checkpoint   

    return analysis_tasks



def run_cartesian_instability():
    #############################################################################################
    ### 1. Read in command-line args, set up data directory
    aspect = 2
    nx = 64
    nz = 256
    Re0 = 4e2
    S = 1e3
    Pr = 5e-1
    P = 1e0
    L_cz = 1e0
    Lz = 2e0
    zeta = 1e-3 
    data_dir = './erf_step_1e-3'
    logger.info("saving run in: {}".format(data_dir))

    ########################################################################################
    ### 2. Organize simulation parameters
    invP = 1/P

    Pe0   = Pr*Re0
    Lx = aspect * Lz

    dH = 0.2
    Qmag = 1
    Fconv = dH * Qmag
    Fbot = zeta*Fconv

    #Model values
    k_rz = dH / (P * S) 
    k_cz = k_rz * ( zeta / (1 + zeta + invP) )
    k_ad = k_rz * ( (1 + zeta) / (1 + zeta + invP) )
    delta_k = k_rz - k_cz
    grad_ad = (Qmag * S * P) * (1 + zeta + invP)
    grad_rad_top = (Qmag * S * P) * (1 + zeta)
    delta_grad = grad_ad - grad_rad_top

    #Adjust to account for expected velocities. and larger m = 0 diffusivities.
    Pe0 /= (np.sqrt(Qmag))
    Re0 /= (np.sqrt(Qmag)) 

    logger.info("Running two-layer instability with the following parameters:")
    logger.info("   Re = {:.3e}, S = {:.3e}, resolution = {}x{}, aspect = {}".format(Re0, S, nx, nz, aspect))
    logger.info("   Pr = {:2g}".format(Pr))
    logger.info("   Re0 = {:.3e}, Pe0 = {:.3e}, Qmag ~ u^2 = {:.3e}".format(Re0, Pe0, Qmag))

    
    ###########################################################################################################3
    ### 3. Setup Dedalus domain, problem, and substitutions/parameters
    x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=3/2)
    bases = [x_basis, z_basis]
    domain = de.Domain(bases, grid_dtype=np.float64, mesh=None)
    reducer = flow_tools.GlobalArrayReducer(domain.distributor.comm_cart)
    z = domain.grid(-1)
    z_de = domain.grid(-1, scales=domain.dealias)

    #Establish variables and setup problem
    variables = ['T1', 'T1_z', 'p', 'u', 'w', 'ωy']
    problem = de.IVP(domain, variables=variables, ncc_cutoff=1e-10)

    # Set up background / initial state vs z.
    T0   = domain.new_field()
    T0_z = domain.new_field()
    T0_zz = domain.new_field()
    T_ad_z = domain.new_field()
    T_rad_z0 = domain.new_field()
    k0     = domain.new_field()
    k0_z     = domain.new_field()
    Q = domain.new_field()
    flux_of_z = domain.new_field()
    cz_mask = domain.new_field()
    for f in [T0, T0_z, T_ad_z, k0, Q, flux_of_z, T_rad_z0, cz_mask]:
        f.set_scales(domain.dealias)
    for f in [T_ad_z, k0]:
        f.meta['x']['constant'] = True

    cz_mask['g'] = zero_to_one(z_de, 0.2, width=0.05)*one_to_zero(z_de, L_cz, width=0.05)

    delta = 0.02
    k_shape = lambda z: zero_to_one(z, L_cz, width=0.075)
    k_func = lambda z: (k_cz + delta_k*k_shape(z))
    Q_func  = lambda z: Qmag*zero_to_one(z, 0.1, delta)*one_to_zero(z, 0.1+dH, delta)
    T_rad_func = lambda flux, k: -flux / k

    k0['g'] = k_func(z_de)
    k0.differentiate('z', out=k0_z)
    Q['g'] = Q_func(z_de)
    Q.antidifferentiate('z', ('left', Fbot), out=flux_of_z)
    flux = flux_of_z.interpolate(z=L_cz)['g'].min()

    T_ad_z['g'] = -grad_ad
    T_rad_z0['g'] = T_rad_func(flux_of_z['g'], k0['g'])
    max_brunt = reducer.global_max(T_rad_z0['g'] - T_ad_z['g'])
    logger.info("Max brunt T: {:.3e}".format(max_brunt))

    #find match point
    zs = np.linspace(0, Lz, 1000)
    kmatch = k_func(zs)
    Tradmatch = T_rad_func(flux, kmatch)
    z_match = np.interp(-grad_ad, Tradmatch, zs)

    #Construct T0_zz so that it gets around the discontinuity.
    width = 0.05
    T_rad_z0.differentiate('z', out=T0_zz)
    T0_zz['g'] *= zero_to_one(z_de.flatten(), z_match - width, width=width)
    T0_zz.antidifferentiate('z', ('left', -grad_ad), out=T0_z)

    #Erf has a width that messes up the transition; bump up T0_zz so it transitions to grad_rad at top.
    deltaT0z_rad = -grad_rad_top + grad_ad
    deltaT0z_sim = T0_z.interpolate(z=Lz)['g'].max() + grad_ad
    T0_zz['g'] *= deltaT0z_rad/deltaT0z_sim
    T0_zz.antidifferentiate('z', ('left', -grad_ad), out=T0_z)
    T0_z.antidifferentiate('z', ('right', 1), out=T0)

    #Check that heating and cooling cancel each other out.
    fH = domain.new_field()
    fH2 = domain.new_field()
    fH.set_scales(domain.dealias)
    fH['g'] = Q['g'] + k0_z['g']*T0_z['g'] + k0['g']*T0_zz['g']
    fH.antidifferentiate('z', ('left', 0), out=fH2)
    logger.info('right(integ(heating - cooling)): {:.3e}'.format(fH2.interpolate(z=Lz)['g'].max()))

    #Plug in default parameters
    problem.parameters['Pe0']    = Pe0
    problem.parameters['Re0']    = Re0
    problem.parameters['Lx']     = problem.parameters['Ly'] = Lx
    problem.parameters['Lz']     = Lz
    problem.parameters['k0']     = k0
    problem.parameters['T0']     = T0
    problem.parameters['T0_z']     = T0_z
    problem.parameters['T0_zz']    = T0_zz
    problem.parameters['T_ad_z'] = T_ad_z
    problem.parameters['Q'] = Q
    problem.parameters['flux_of_z'] = flux_of_z
    problem.parameters['cz_mask'] = cz_mask
    problem.parameters['max_brunt'] = max_brunt 

    problem = set_subs(problem)
    problem = set_equations(problem)

    ts = de.timesteppers.RK443
    solver = problem.build_solver(ts)
    logger.info('Solver built')

    ###########################################################################
    ### 4. Set initial conditions or read from checkpoint.
    mode = 'overwrite'
    T1 = solver.state['T1']
    T1_z = solver.state['T1_z']
    z_de = domain.grid(-1, scales=domain.dealias)
    for f in [T1, T1_z]:
        f.set_scales(domain.dealias, keep_data=True)

    noise = global_noise(domain, 101)
    T1['g'] += 1e-3*np.sin(2*np.pi*(z_de)/Lz)*noise['g']
    T1.differentiate('z', out=T1_z)
    dt = None

    ###########################################################################
    ### 5. Set simulation stop parameters, output, and CFL
    t_ff    = 1/np.sqrt(Qmag)
    t_therm = Pe0
    t_brunt   = np.sqrt(1/max_brunt)
    max_dt    = np.min((0.5*t_ff, t_brunt))
    logger.info('buoyancy and brunt times are: {:.2e} / {:.2e}; max_dt: {:.2e}'.format(t_ff, t_brunt, max_dt))
    if dt is None:
        dt = max_dt

    #cfl_safety = float(args['--safety'])
    cfl_safety = 0.35
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=cfl_safety,
                         max_change=1.5, min_change=0.25, max_dt=max_dt, threshold=0.2)
    CFL.add_velocities(('u', 'w'))

    #run_time_ff   = float(args['--run_time_ff'])
    run_time_ff = 5e4
    #run_time_wall = float(args['--run_time_wall'])
    run_time_wall = 48
    solver.stop_sim_time  = run_time_ff*t_ff
    solver.stop_wall_time = run_time_wall*3600.
 
    ###########################################################################
    ### 6. Setup output tasks; run main loop.
    analysis_tasks = initialize_output(solver, data_dir, mode=mode, output_dt=t_ff)

    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re", name='Re')
    flow.add_property("T1(z=0)", name='T1_bot')
    
#grad_departure_frac = 0.5

    ### for comparing to d3 ###
    #k0.set_scales(1, keep_data=True)
    #T0.set_scales(1, keep_data=True)
    #T0_z.set_scales(1, keep_data=True)
    #T0_zz.set_scales(1, keep_data=True)
    #np.save('d2_k0.npy', k0['g'])
    #np.save('d2_T0.npy', T0['g'])
    #np.save('d2_T0_z.npy', T0_z['g'])
    #np.save('d2_T0_zz.npy', T0_zz['g'])


    logger.info('Reached beginning of main loop')

    def main_loop(dt):
        Re_avg = 0
        #max_T_iters = int(args['--T_iters'])
    #    done_T_iters = 0

    #    transient_wait = 30 
    #    transient_start = None
    #    N = 70
    #    halfN = int(N/2)
    #    avg_dLz_dt = 0
    #    top_cz_times = np.zeros(N)
    #    top_cz_z = np.zeros(N)
    #    good_times = np.zeros(N, dtype=bool)
    #    last_height_t = 0

    #    L_cz0 = None
    #    zmax = 0
    #    tol = 1e-8 #tolerance on magnitude of dLcz/dt

        try:
            logger.info('Starting loop')
            start_iter = solver.iteration
            start_time = time.time()
    #        shell_at_top = False
            while solver.ok and np.isfinite(Re_avg):
                effective_iter = solver.iteration - start_iter
                solver.step(dt)

    #            if solver.sim_time > last_height_t + 1:
    #                last_height_t = int(solver.sim_time)
    #                #Get departure point from grad_ad
    #                grad = dense_handler['grad']['g'][0,:]
    #                cz_points = grad > grad_ad - grad_departure_frac*delta_grad
    #                if np.sum(cz_points) > 0:
    #                    zmax = z_dense.flatten()[cz_points].max()
    #                else:
    #                    zmax = 0
    #                zmax = reducer.reduce_scalar(zmax, MPI.MAX)
    #                #Track trajectory of grad_ad->grad_rad departure over time
    #                if Re_avg > 1:
    #                    if transient_start is None:
    #                        transient_start = int(solver.sim_time)
    #                    if solver.sim_time > transient_start + transient_wait:
    #                        top_cz_z[:-1] = top_cz_z[1:]
    #                        top_cz_times[:-1] = top_cz_times[1:]
    #                        good_times[:-1] = good_times[1:]
    #                        top_cz_z[-1] = zmax
    #                        top_cz_times[-1] = solver.sim_time
    #                        good_times[-1] = True
    #                        if L_cz0 is None:
    #                            L_cz0 = top_cz_z[-1]

    #                        if good_times[-2] == True:
    #                            dLz_dt = np.gradient(top_cz_z[good_times], top_cz_times[good_times])
    #                            avg_dLz_dt = np.mean(dLz_dt)

    #                #Adjust background thermal profile
    #                if done_T_iters < max_T_iters and np.sum(good_times) == N and np.abs(avg_dLz_dt) > tol:
    #                    dLz_dt_beg = np.mean(dLz_dt[:halfN])
    #                    dLz_dt_end = np.mean(dLz_dt[halfN:])
    #                    Lz_beg = np.mean(top_cz_z[:halfN])
    #                    Lz_end = np.mean(top_cz_z[halfN:])
    #                    delta_Lz = Lz_end - Lz_beg
    #                    delta_dLz_dt = dLz_dt_end - dLz_dt_beg
    #                    logger.info('Lz: {:.3f}/{:.3f}, deltaLz: {:.3e}'.format(Lz_beg, Lz_end, delta_Lz))
    #                    logger.info("dLz/dt: {:.3e}/{:.3e}, deltadLz/dt: {:.3e}".format(dLz_dt_beg, dLz_dt_end, delta_dLz_dt))
    #                    if delta_Lz != 0 and delta_dLz_dt != 0:
    #                        L_cz1 = Lz_beg - delta_Lz * (dLz_dt_beg/delta_dLz_dt)
    ##                        L_cz1 = L_cz0 + 2*(N + transient_wait)*avg_dLz_dt
    #                        mean_T_z = -(grad_ad - zero_to_one(z_de, L_cz1, width=0.05)*delta_grad)
    #                        mean_T1_z = mean_T_z - T0_z['g'][0,:]
    #                        T1_z['g'] -= flow.properties['mean_T1_z']['g']
    #                        T1_z['g'] += mean_T1_z
    #                        T1_z.antidifferentiate('z', ('right', 0), out=T1)

    #                        L_cz0 = L_cz1
    #                        good_times[:] = False
    #                        transient_start = None
    #                        done_T_iters += 1
    #                        logger.info('T_adjust {}/{}: Adjusting mean state to have L_cz = {:.4f}'.format(done_T_iters, max_T_iters, L_cz1))


                if effective_iter % 10 == 0:
                    Re_avg = flow.grid_average('Re')

                    log_string =  'Iteration: {:7d}, '.format(solver.iteration)
                    log_string += 'Time: {:8.3e} ({:8.3e} therm), dt: {:8.3e}, '.format(solver.sim_time/t_ff, solver.sim_time/Pe0,  dt/t_ff)
                    log_string += 'Re: {:8.3e}/{:8.3e}, '.format(flow.max('Re'), flow.volume_average('Re'))
                    log_string += 'mean_T1_bot: {:8.3e}/{:89.3e}, '.format(flow.max('T1_bot'), flow.grid_average('T1_bot'))
                    #log_string += 'mean_T1_z: {:8.3e}/{:8.3e}, '.format(flow.max('mean_T1_z'), flow.volume_average('mean_T1_z'))
                    
                    
                    #log_string += 'Pe: {:8.3e}/{:8.3e}, '.format(flow.grid_average('Pe'), flow.max('Pe'))
                    #log_string += 'CZ_zmax: {:.03f}, '.format(zmax)
                    #log_string += 'avg_dLz_dt: {:.01e}, '.format(avg_dLz_dt)
                    #log_string += 'stiffness: {:.01e}'.format(flow.grid_average('stiffness'))
                    logger.info(log_string)

                dt = CFL.compute_dt()

        except:
            raise
            logger.error('Exception raised, triggering end of main loop.')
        finally:
            end_time = time.time()
            main_loop_time = end_time-start_time
            n_iter_loop = solver.iteration-start_iter
            logger.info('Iterations: {:d}'.format(n_iter_loop))
            logger.info('Sim end time: {:f}'.format(solver.sim_time))
            logger.info('Run time: {:f} sec'.format(main_loop_time))
            logger.info('Run time: {:f} cpu-hr'.format(main_loop_time/60/60*domain.dist.comm_cart.size))
            logger.info('iter/sec: {:f} (main loop only)'.format(n_iter_loop/main_loop_time))
    #        try:
    #            final_checkpoint = solver.evaluator.add_file_handler(data_dir+'final_checkpoint', wall_dt=np.inf, sim_dt=np.inf, iter=1, max_writes=1)
    #            final_checkpoint.add_system(solver.state, layout = 'c')
    #            solver.step(1e-5*dt) #clean this up in the future...works for now.
    #            post.merge_process_files(data_dir+'/final_checkpoint/', cleanup=False)
    #        except:
    #            raise
    #            print('cannot save final checkpoint')
    #        finally:
    #            logger.info('beginning join operation')
    #            for key, task in analysis_tasks.items():
    #                logger.info(task.base_path)
    #                post.merge_analysis(task.base_path)
    #        domain.dist.comm_cart.Barrier()
        return Re_avg

    Re_avg = main_loop(dt)
    #if np.isnan(Re_avg):
    #    return False, data_dir
    #else:
    #    return True, data_dir
    return True

if __name__ == "__main__":
    #ended_well, data_dir = run_cartesian_instability(args)
    ended_well = run_cartesian_instability()
    #if MPI.COMM_WORLD.rank == 0:
    #    print('ended with finite Re? : ', ended_well)
    #    print('data is in ', data_dir)
