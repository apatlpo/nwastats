#!/usr/bin/env python
# coding: utf-8

import os
from tqdm import tqdm

import xarray as xr
import pandas as pd
import numpy as np
np.random.seed(1234)

import nwastats as st
colors = st.colors
day = 86400

#data_dir = "data/"
#data_dir = "/home1/scratch/aponte/"
data_dir = "/home/datawork-lops-osi/aponte/nwa/drifter_stats"

#overwrite=True # allways overwrite here

# ----------------------------------------------------------------------------------

## define kernels

#case = "3D_matern32_iso_matern12_pp_r0.0_u0.1"
case = "3D_matern52_iso_matern12_pp_r0.0_u0.1"
run_dir = os.path.join(data_dir, case)

uv = True # True if u and v are observed
parameter_eta_formulation = False # eta vs gamma formulation
noise = 0.01 # observation noise added to u/v
no_time = False # activates time inference
no_space = False # activates time inference
traj_decorrelation = False  # artificially decorrelate different moorings/drifters
#traj_decorrelation = True  # artificially decorrelate different moorings/drifters
enable_nu = False
#enable_nu = True # enable estimation of spectral slopes

if no_space:
    # makes little sense otherwise
    parameter_eta_formulation = True
    traj_decorrelation = True
assert not no_time, "need to implement decorrelation across time"

# run inference and store
production = True
#production = False
emcee = False

# inference number of platforms
Nxy = 8

# number of inference samples:
n_mcmc = 20_000 # default value
if enable_nu:
    # 20_000 is 28h of computations
    n_mcmc = 10_000 # default value
#n_mcmc = None

# output file suffix
out_suffix = None
#out_suffix = f"_n{n_mcmc}"


## load data
dsd = xr.open_dataset(os.path.join(run_dir, f"drifters_selection.nc"))
dsd["time"] = dsd["time"]/pd.Timedelta("1D")

dsd_full = (xr
    .open_zarr(os.path.join(run_dir, f"drifters.zarr"))
    .rename(lon="x", lat="y")
    .drop_vars("z")
)
#_t = (dsd_full["time"] - dsd_full["time"][0,0]).median("traj",skipna=True)/pd.Timedelta("1D")
_t = (dsd_full["time"]).median("trajectory",skipna=True)/pd.Timedelta("1D")
dsd_full = dsd_full.assign_coords(t=_t)
dsd_full["x"] = dsd_full["x"]/1e3
dsd_full["y"] = dsd_full["y"]/1e3

# load eulerian flow
dsf = xr.open_dataset(os.path.join(run_dir, "flow_selection.nc")).sortby("trajectory")
dsf["time"] = dsf["time"]/pd.Timedelta("1D")
ds_flow = xr.open_zarr(os.path.join(run_dir, f"flow.zarr"))
ds_flow_reduced = ds_flow.sel(x=dsf.x, y=dsf.y).compute()




##
# problem parameters (need to be consistent with data generation notebook)
# should be a copy of the code in st.prepare_inference

p = dsd.attrs
# !! to adjust
η = p["amplitude0"]  # streamfunction unit
#η = p["amplitude1"]  # potential unit
#
λx = p["lambda_x"]   # km
λy = p["lambda_y"]   # km
λt = p["lambda_t"]   # days
#
νs = p["nu_space"]
νt = p["nu_time"]

# derived velocity parameter
γ = η / λx 

Lx = float(ds_flow.x.max()-ds_flow.x.min())
Ly = float(ds_flow.y.max()-ds_flow.y.min())
p["dx"] = float(ds_flow.x.diff("x").median())
p["dy"] = float(ds_flow.y.diff("y").median())
# km, km this should not matter

# get 1D covariances
C, Ct, isotropy = st.get_cov_1D(p["cov_x"], p["cov_t"], enable_nu)
Cu, Cv, Cuv = C

# set covariance parameters
if no_time:
    if uv:
        covfunc = lambda x, xpr, params: st.kernel_2d_iso_uv(x, xpr, params, (Cu, Cv, Cuv))
    else:
        covfunc = lambda x, xpr, params: st.kernel_2d_iso_u(x, xpr, params, Cu)
    covparams = [η, λx]
    labels = ['σ','η','λx',]
elif no_space:
    if uv:
        if traj_decorrelation:
            covfunc = lambda x, xpr, params: st.kernel_1d_uv_traj(x, xpr, params, Ct)
        else:
            covfunc = lambda x, xpr, params: st.kernel_1d_uv(x, xpr, params, Ct)
    else:
        assert False, "not implemented"
    covparams = [U, λt]
    labels = ['σ','u','λt']
elif isotropy:
    if uv:
        if parameter_eta_formulation:
            covfunc = lambda x, xpr, params: st.kernel_3d_iso_uv(x, xpr, params, (Cu, Cv, Cuv, Ct))
            covparams = [η, λx, λt]
            labels = ['σ','η','λx','λt']
        else:
            def covfunc(x, xpr, params):
                # params contains (gamma=eta/ld, ld, lt) and needs to be converted to (eta, ld, lt)
                params = (params[0]*params[1], *params[1:])
                if traj_decorrelation:
                    return st.kernel_3d_iso_uv_traj(x, xpr, params, (Cu, Cv, Cuv, Ct))
                else:
                    #return st.kernel_3d_iso_uv_old(x, xpr, params, (Cu, Cv, Cuv, Ct))
                    return st.kernel_3d_iso_uv(x, xpr, params, (Cu, Cv, Cuv, Ct))
            if enable_nu:
                covparams = [γ, λx, λt, νs, νt]
                labels = ['σ','γ','λx','λt', 'νs', 'νt']
            else:
                covparams = [γ, λx, λt]
                labels = ['σ','γ','λx','λt']
    else:
        covfunc = lambda x, xpr, params: st.kernel_3d_iso_u(x, xpr, params, (Cu, Ct))
        covparams = [η, λx, λt]
        labels = ['σ','η','λx','λt']
else:
    covfunc = lambda x, xpr, params: st.kernel_3d(x, xpr, params, (Cx, Cy, Ct))
    covparams = [η, λx, λy, λt]
    labels = ['σ','η','λx','λy','λt']

truth = {k: v for k, v in zip(labels, [noise]+covparams)}

# inference parameters

# bounds
lowers, uppers = None, None
if enable_nu:
    lowers = [None]*3 + [νs-0.5, νt-0.49] + [None]
    uppers = [None]*3 + [νs+0.5, νt+0.5] + [None]
    #lowers = [None]*3 + [νs-1., νt-0.49] + [None]
    #uppers = [None]*3 + [νs+1., νt+1.] + [None]
print(lowers, uppers)

#
# dsteps = {16: 1/20, 8: 1/20, 4: 1/5, 2: 1/5, 1: 1/2} #v0
dsteps = {16: .05, 8: .1, 4: .15, 2: .2, 1: .3} # v1
if enable_nu:
    dsteps = {k: v*.5 for k, v in dsteps.items()}

s = dsteps[Nxy]
steps = [s]*4
if enable_nu:
    #steps = [s]*5+[1/2]
    steps = [s]*6


# ----------------------------------------------------------------------------------
## mooring inference

if True:

    print("Processing mooring")
    
    # load data
    ds = dsf.copy()
    dt0 = float(np.unique(ds.time.diff("time"))[0])
    print(f"initial time sampling rate {dt0:.3f} days")
    
    # decimate
    if no_time:
        npts = 100
        dl = 200
        ds = (ds_flow
            .isel(time=0)
            .sel(x=slice(500-dl,500+dl), y=slice(500-dl,500+dl))
            .stack(trajectory=["x", "y"])
            .rename(U="u", V="v")
        )
        ds.u.unstack().plot()
        #
        narrow_selection = np.random.choice(ds.trajectory.values, npts)
        ds = ds.sel(trajectory=narrow_selection)
        #ds.unstack().psi.plot()
    else:
        ds = ds.sel(time=slice(0, None, int(2/dt0)))   # temporally
        #ds = ds.isel(trajectory=slice(0, None, 2))    # spatially
        ds = ds.sel(trajectory=np.random.choice(ds.trajectory.values, Nxy, replace=False))
        #
        #da = ds.u
        #da  = da + np.arange(ds.trajectory.size)/5
        #da.plot(hue="trajectory", add_legend=False, figsize=(5,7));
        
    # store for plotting purposes latter
    ds_mo = ds
    
    # number of data points
    N = ds.u.size
    assert Nxy==ds.trajectory.size, "issue with Nxy"
    print(f"Number of data points = {N}, ({Nxy} spatial locations)")
    
    # problem parameters (need to be consistent with data generation notebook)
    if not no_time:
        Lt = float(ds.time.max()-ds.time.min()) # days
    
    # estimate effective resolution
    δx = Lx/Nxy
    δy = Ly/Nxy
    print("-- resolutions:")
    print(f"λx/δx = {λx/δx}")
    print(f"λy/δy = {λy/δy}")
    if not no_time:
        δt = float(ds.time.diff("time")[0])
        print(f"λt/δt = {λt/δt}")
    
    # estimate number of independent samples
    print("-- independent samples (conservative):")
    print(f"Lx/λx = {Lx/λx} ") # this is probably conservative
    print(f"Ly/λy = {Ly/λy} ") # this is probably conservative
    if not no_time:
        print(f"Lt/λt = {Lt/λt} ") # this is probably conservative
    
    u, v, x, y, t = xr.broadcast(ds.u, ds.v, ds.x, ds.y, ds.time)
    assert u.shape==v.shape==x.shape==y.shape==t.shape
    x = x.values.ravel()
    y = y.values.ravel()
    t = t.values.ravel()
    X = np.hstack([x[:,None], y[:,None], t[:,None]])
        
    u = u.values.ravel()[:, None]
    v = v.values.ravel()[:, None]
    
    # add noise
    u += np.random.randn(*u.shape)*noise
    v += np.random.randn(*v.shape)*noise
    
    if uv:
        X = np.vstack([X, X])
        U = np.vstack([u, v])
    else:
        U = u
    
    # dev
    #X = X[::10,:]
    #U = U[::10]
    #X = X[::5,:]
    #U = U[::5]
    
    print(X.shape, U.shape)
    
    
    ## run inference
    nc = os.path.join(run_dir, f"moorings_reference_inference.nc")
    if enable_nu:
        nc = nc.replace(".nc", "_nu.nc")
    if out_suffix is not None:
        nc = nc.replace(".nc", out_suffix+".nc")
    print(nc)
    
    if production:
        #dsE1 = st.inference_MH_old(
        dsE1 = st.inference_MH(
            X, U, noise, covparams, covfunc, labels, 
            lowers=lowers, uppers=uppers, steps=steps,
            n_mcmc = n_mcmc,
        )
        # to control number of samples: ..., n_mcmc = int(10e3) ...
        dsE1.to_netcdf(nc, mode="w")    
    else:
        dsE1 = xr.open_dataset(nc)
    # 10 min
    # accepted fraction: 5% (8 platforms)
    
    # default: 9 it/s 
    # enable_nu=True: 5s / it -> 5*9 = 45 time drop in performance
    
    # move noise last
    #dsE1 = dsE1.sel(parameter=['γ', 'λx', 'λt', 'σ'])


# ----------------------------------------------------------------------------------
## drifter inference

if True:

    print("Processing drifter")
    
    # load data
    #ds = xr.open_dataset("data/drifters_3D_selection.nc", decode_times=False)
    ds = dsd
    
    dt0 = float(np.median(ds.time.diff("time")))
    print(f"initial time sampling rate {dt0:.03f} days")
    
    # decimate
    ds = (
        ds
        .isel(time=slice(0, -50, int(2/dt0)))   # temporally
        .sel(trajectory=np.random.choice(ds.trajectory.values, Nxy, replace=False))
        #.isel(trajectory=slice(0, None, 2))    # spatially
    )
    
    # store for plotting purposes latter
    ds_dr = ds
    
    
    # number of data points
    N = ds.u.size
    Nxy = ds.trajectory.size
    print(f"Number of data points = {N}, ({Nxy} drifters)")
    
    # problem parameters (need to be consistent with data generation notebook)
    Lt = float(ds.time.max()-ds.time.min()) # days
    
    # estimate effective resolution
    δx = Lx/Nxy
    δy = Ly/Nxy
    δt = float(ds.time.diff("time")[0])
    print("-- resolutions:")
    print(f"λx/δx = {λx/δx}")
    print(f"λy/δy = {λy/δy}")
    print(f"λt/δt = {λt/δt}")
    
    # estimate number of independent samples
    print("-- independent samples (conservative):")
    print(f"Lx/λx = {Lx/λx} ") # this is probably conservative
    print(f"Ly/λy = {Ly/λy} ") # this is probably conservative
    print(f"Lt/λt = {Lt/λt} ") # this is probably conservative
    
    
    # massage inputs to inference problem
    u, v, x, y, t = xr.broadcast(ds.u, ds.v, ds.x, ds.y, ds.time)
    assert u.shape==x.shape==y.shape==t.shape
    x = x.values.ravel()
    y = y.values.ravel()
    t = t.values.ravel()
    
    X = np.hstack([x[:,None], y[:,None], t[:,None]])
    
    u = u.values.ravel()[:, None]
    v = v.values.ravel()[:, None]
    
    # add noise
    u += np.random.randn(*u.shape)*noise
    v += np.random.randn(*v.shape)*noise
    
    if uv:
        X = np.vstack([X, X])
        U = np.vstack([u, v])
    else:
        U = u
    
    # dev
    #X = X[::10,:]
    #U = U[::10]
    #X = X[::5,:]
    #U = U[::5]
    
    print(X.shape, U.shape)
    
    
    nc = os.path.join(run_dir, f"drifters_reference_inference.nc")
    if enable_nu:
        nc = nc.replace(".nc", "_nu.nc")
    if out_suffix is not None:
        nc = nc.replace(".nc", out_suffix+".nc")
    print(nc)
    
    if production:
        dsL1 = st.inference_MH(
            X, U, noise, covparams, covfunc, labels, 
            lowers=lowers, uppers=uppers, steps=steps,
            n_mcmc=n_mcmc,
        )
        # to control number of samples: ..., n_mcmc = int(10e3) ...
        dsL1.to_netcdf(nc, mode="w")
    else:
        dsL1 = xr.open_dataset(nc)
    
    # move noise last
    #dsL1 = dsL1.sel(parameter=['γ', 'λx', 'λt', 'σ'])


print("All done")