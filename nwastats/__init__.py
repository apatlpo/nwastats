import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

import dask
import dask.array as da

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from seaborn import color_palette

from scipy.special import kv, kvp, gamma
from gptide import cov
from gptide import GPtideScipy
from gptide import mcmc
import corner
import arviz as az

import xrft

day = 86400

# colors drifters/moorings
cpal = color_palette("colorblind")
colors = dict(
    mo = cpal[0],
    #mo = "#ff7f00",
    dr = cpal[1],
    #dr = "#377eb8",
    truth = "k",
    #truth = "deeppink",
    #truth = cpal[2],
    MAP = cpal[2],
)


# ------------------------------------- covariances ------------------------------------------

# https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function

# copy from https://github.com/TIDE-ITRH/gptide/blob/main/gptide/cov.py
def matern_general(dx, eta, nu, l):
    """General Matern base function"""
    
    cff1 = np.sqrt(2*nu)*np.abs(dx)/l
    K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu)
    K *= np.power(cff1, nu)
    K *= kv(nu, cff1)

    if isinstance(K, np.ndarray):
        K[np.isnan(K)] = np.power(eta, 2.)
    
    return K

def matern_general_d1(dx, eta, nu, l):
    """General Matern base function, first derivative"""
    
    cff0 = np.sqrt(2*nu)/l
    cff1 = cff0*np.abs(dx)
    K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu) * cff0
    K *= (
        nu*np.power(cff1, nu-1)*kv(nu, cff1)
        + np.power(cff1, nu)*kvp(nu, cff1, n=1)
    )
    K[np.isnan(K)] = 0.
    # but remember K'(d)/d converge toward K''(0) towards 0
    
    return K

def matern_general_d2(dx, eta, nu, l):
    """General Matern base function, second derivative"""
    
    cff0 = np.sqrt(2*nu)/l
    cff1 = cff0*np.abs(dx)
    K = np.power(eta, 2.) * np.power(2., 1-nu) / gamma(nu) * cff0**2
    K *= (
        nu*(nu-1)*np.power(cff1, nu-2)*kv(nu,cff1) 
        + 2*nu*np.power(cff1, nu-1)*kvp(nu,cff1, n=1)
        + np.power(cff1, nu)*kvp(nu, cff1, n=2)
    )
    K[np.isnan(K)] = -np.power(eta, 2.) * nu/(nu-1)/l**2
    
    return K

##
def matern32_d1(dx, eta, l):
    """Matern 3/2 function, first derivative"""
    
    cff0 = np.sqrt(3)/l
    cff1 = cff0*np.abs(dx)
    Kp = -np.power(eta, 2.) * cff0 * cff1 * np.exp(-cff1)
    
    return Kp

def matern32_d2(dx, eta, l):
    """Matern 3/2 function, second derivative"""
    
    cff0 = np.sqrt(3)/l
    cff1 = cff0*np.abs(dx)
    Kpp = np.power(eta, 2.) * cff0**2 * (-1+cff1) * np.exp(-cff1)
    
    return Kpp

##
def matern52_d1(dx, eta, l):
    """Matern 5/2 function, first derivative"""
    
    cff0 = np.sqrt(5)/l
    cff1 = cff0*np.abs(dx)
    Kp = -np.power(eta, 2.) * cff0 * cff1*(1+cff1)/3 * np.exp(-cff1)
    
    return Kp

def matern52_d2(dx, eta, l):
    """Matern 5/2 function, second derivative"""
    
    cff0 = np.sqrt(5)/l
    cff1 = cff0*np.abs(dx)
    Kpp = np.power(eta, 2.) * cff0**2 * (-1-cff1+cff1*cff1)/3 * np.exp(-cff1)
    
    return Kpp

def get_cov_1D(cov_x, cov_t, enable_nu):

    isotropy = False
    if cov_x == "matern12_xy":
        Cx = cov.matern12  # -2 spectral slope in 1D
        Cy = cov.matern12  # -2 spectral slope in 1D
        C = (Cx, Cy, None)
    elif cov_x == "matern32_xy":
        Cx = cov.matern32  # -4 isotropic spectral slope in 2D - not twice differentiable
        Cy = cov.matern32  # -4 isotropic spectral slope in 2D
        C = (Cx, Cy, None)
    elif cov_x == "matern2_xy":
        #Cov_x = cov.matern_general(np.abs(t_x - t_x.T), 1., 2, λx) # -5 spectral slope
        #Cov_y = cov.matern_general(np.abs(t_y - t_y.T), 1., 2, λy) # -5 spectral slope
        assert False, "not implemented"
        pass
    elif cov_x == "matern52_xy":
        Cx = cov.matern52  # -5 isotropic spectral slope
        Cy = cov.matern52  # -5 isotropic spectral slope
        C = (Cx, Cy, None)
    elif cov_x == "expquad":
        #jitter = -10
        #Cx = cov.expquad(t_x, t_x.T, λx) # + 1e-10 * np.eye(Nx)
        #Cy = cov.expquad(t_y, t_y.T, λy) # + 1e-10 * np.eye(Nx)
        assert False, "need revision"

    # isotropic cases
    isotropy = ("iso" in cov_x)
    #if cov_x == "matern2_iso" or True: # dev
    if enable_nu:
        # for covariances based on distances
        def Cu(x, y, d, nu, λ):
            C = -(
                y**2 * matern_general_d2(d, 1., nu, λ)
                + x**2 * matern_general_d1(d, 1., nu, λ) / d
            )/ d**2
            C[np.isnan(C)] = -matern_general_d2(d[np.isnan(C)], 1.0, nu, λ)
            return C
        def Cv(x, y, d, nu, λ):
            C = -(
                x**2 * matern_general_d2(d, 1., nu, λ)
                + y**2 * matern_general_d1(d, 1., nu, λ) / d
            ) / d**2
            C[np.isnan(C)] = -matern_general_d2(d[np.isnan(C)], 1.0, nu, λ)
            return C
        def Cuv(x, y, d, nu, λ):
            C = x*y*(
                    matern_general_d2(d, 1., nu, λ)
                    - matern_general_d1(d, 1., nu, λ) / d
                ) / d**2
            C[np.isnan(C)] = 0.
            return C
        C = (Cu, Cv, Cuv)
    elif cov_x == "matern2_iso":
        nu = 2
        #nu = 3/2 # dev
        # for covariances based on distances
        def Cu(x, y, d, λ):
            C = -(
                y**2 * matern_general_d2(d, 1., nu, λ)
                + x**2 * matern_general_d1(d, 1., nu, λ) / d
            )/ d**2
            C[np.isnan(C)] = -matern_general_d2(d[np.isnan(C)], 1.0, nu, λ)
            return C
        def Cv(x, y, d, λ):
            C = -(
                x**2 * matern_general_d2(d, 1., nu, λ)
                + y**2 * matern_general_d1(d, 1., nu, λ) / d
            ) / d**2
            C[np.isnan(C)] = -matern_general_d2(d[np.isnan(C)], 1.0, nu, λ)
            return C
        def Cuv(x, y, d, λ):
            C = x*y*(
                    matern_general_d2(d, 1., nu, λ)
                    - matern_general_d1(d, 1., nu, λ) / d
                ) / d**2
            C[np.isnan(C)] = 0.
            return C
        C = (Cu, Cv, Cuv)
    elif cov_x == "matern32_iso":
        # for covariances based on distances
        def Cu(x, y, d, λ):
            C = -(
                y**2 * matern32_d2(d, 1., λ)
                + x**2 * matern32_d1(d, 1., λ) / d
            )/ d**2
            C[np.isnan(C)] = -matern32_d2(d[np.isnan(C)], 1.0, λ)
            return C
        def Cv(x, y, d, λ):
            C = -(
                x**2 * matern32_d2(d, 1., λ)
                + y**2 * matern32_d1(d, 1., λ) / d
            ) / d**2
            C[np.isnan(C)] = -matern32_d2(d[np.isnan(C)], 1.0, λ)
            return C
        def Cuv(x, y, d, λ):
            C = x*y*(
                    matern32_d2(d, 1., λ)
                    - matern32_d1(d, 1., λ) / d
                ) / d**2
            C[np.isnan(C)] = 0.
            return C
        C = (Cu, Cv, Cuv)
    elif cov_x == "matern52_iso":
        # for covariances based on distances
        def Cu(x, y, d, λ):
            C = -(
                y**2 * matern52_d2(d, 1., λ)
                + x**2 * matern52_d1(d, 1., λ) / d
            )/ d**2
            C[np.isnan(C)] = -matern52_d2(d[np.isnan(C)], 1.0, λ)
            return C
        def Cv(x, y, d, λ):
            C = -(
                x**2 * matern52_d2(d, 1., λ)
                + y**2 * matern52_d1(d, 1., λ) / d
            ) / d**2
            C[np.isnan(C)] = -matern52_d2(d[np.isnan(C)], 1.0, λ)
            return C
        def Cuv(x, y, d, λ):
            C = x*y*(
                    matern52_d2(d, 1., λ)
                    - matern52_d1(d, 1., λ) / d
                ) / d**2
            C[np.isnan(C)] = 0.
            return C
        C = (Cu, Cv, Cuv)
    # dev
    #Cu, Cv, Cuv = (lambda x, y, d, λ: np.eye(*x.shape),)*3 

    if enable_nu:
        Ct = matern_general
    else:
        Ct = getattr(cov, cov_t)

    return C, Ct, isotropy, enable_nu

def kernel_3d(x, xpr, params, C):
    """
    3D kernel
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            lx: x length scale
            ly: y length scale
            lt: t length scale
            
    """
    eta, lx, ly, lt = params
    Cx, Cy, Ct = C
    
    # Build the covariance matrix
    C  = Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    C *= Cy(x[:,1,None], xpr.T[:,1,None].T, ly) 
    C *= Cx(x[:,0,None], xpr.T[:,0,None].T, lx)
    C *= eta**2
    
    return C

def kernel_3d_iso(x, xpr, params, C):
    """
    3D kernel
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            ld: spatial scale
            lt: t length scale
            
    """
    eta, ld, lt = params
    Cx, Ct = C
    
    # Build the covariance matrix
    C  = Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    d = np.sqrt( (x[:,0,None]  - xpr.T[:,0,None].T)**2 + (x[:,1,None]  - xpr.T[:,1,None].T)**2 )
    C *= Cx(d, ld)
    C *= eta**2
    
    return C

def kernel_3d_iso_uv(x, xpr, params, C):
    """
    3D spatially isotropic kernel, two velocity components
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            ld: spatial scale
            lt: t length scale
            
    """
    if len(params)==3:
        eta, ld, lt = params
        ps = (ld,)
        pt = (lt,)
    elif len(params)==5:
        eta, ld, lt, nu_s, nu_t = params
        ps = (ld, nu_s)
        pt = (lt, nu_t)
    Cu, Cv, Cuv, Ct = C
    
    # Build the covariance matrix
    n = x.shape[0]//2
    _x = x[:n,0,None] - xpr.T[:n,0,None].T
    _y = x[:n,1,None] - xpr.T[:n,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    #
    C = np.ones((2*n,2*n))
    #
    C[:n,:n] *= Cu(_x, _y, _d, *ps)
    C[n:,n:] *= Cv(_x, _y, _d, *ps)
    #assert False, "need to check two lines below is correct, e.g. isn't a transpose required or a sign change?"
    # it is correct: Cuv = Cuv.T, Cuv is even in terms of x and y
    C[:n,n:] *= Cuv(_x, _y, _d, *ps)
    C[n:,:n] = C[:n,n:]   # assumes X is indeed duplicated vertically
    #
    #_Cu  = Cu(_x, _y, _d, ld)
    #_Cv  = Cv(_x, _y, _d, ld)
    #_Cuv  = Cuv(_x, _y, _d, ld)
    #C *= np.block([[_Cu, _Cuv],[_Cuv, _Cv]])
    C *= Ct(x[:,2,None], xpr.T[:,2,None].T, *pt)
    C *= eta**2
    
    return C

def kernel_3d_iso_uv_traj(x, xpr, params, C):
    """
    3D spatially isotropic kernel, two velocity components
    decorrelate data with different id
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            ld: spatial scale
            lt: t length scale
            
    """
    if len(params)==3:
        eta, ld, lt = params
        ps = (ld,)
        pt = (lt,)
    elif len(params)==5:
        eta, ld, lt, nu_s, nu_t = params
        ps = (ld, nu_s)
        pt = (lt, nu_t)
    Cu, Cv, Cuv, Ct = C
    
    # Build the covariance matrix
    n = x.shape[0]//2
    _x = x[:n,0,None] - xpr.T[:n,0,None].T
    _y = x[:n,1,None] - xpr.T[:n,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    #
    C = np.ones((2*n,2*n))
    C[:n,:n] *= Cu(_x, _y, _d, *ps)
    C[n:,n:] *= Cv(_x, _y, _d, *ps)
    #assert False, "need to check two lines below is correct, e.g. isn't a transpose required or a sign change"
    C[:n,n:] *= Cuv(_x, _y, _d, *ps)
    C[n:,:n] = C[:n,n:]   # assumes X is indeed duplicated vertically
    #
    C *= Ct(x[:,2,None], xpr.T[:,2,None].T, *pt)
    # decorrelate different trajectories (moorings/drifters)
    C *= ( (x[:,3,None] - xpr.T[:,3,None].T)==0 ).astype(int)
    C *= eta**2
    
    return C

def kernel_3d_iso_u(x, xpr, params, C):
    """
    3D kernel, one velocity component
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 3
            eta: standard deviation
            ld: spatial scale
            lt: t length scale
            
    """
    eta, ld, lt = params
    Cu, Ct = C
    
    # Build the covariance matrix
    C  = Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    _x = x[:,0,None] - xpr.T[:,0,None].T
    _y = x[:,1,None] - xpr.T[:,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    C *= Cu(_x, _y, _d, ld)
    C *= eta**2
    
    return C

def kernel_2d_iso_uv(x, xpr, params, C):
    """
    2D kernel (no time), one velocity component
    
    Inputs:
        x: matrices input points [N,2]
        xpr: matrices output points [M,2]
        params: tuple length 2
            eta: standard deviation
            ld: spatial scale
            
    """
    eta, ld = params
    Cu, Cv, Cuv = C
    
    # Build the covariance matrix
    n, m = x.shape[0]//2, xpr.shape[1]//2
    _x = x[:n,0,None] - xpr.T[:m,0,None].T
    _y = x[:n,1,None] - xpr.T[:m,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    #
    C = np.ones((2*n,2*m))
    # test comment out
    C[:n,:m] *= Cu(_x, _y, _d, ld)
    C[:n,m:] *= Cuv(_x, _y, _d, ld)
    C[n:,:m] = C[:n,m:]   # assumes X is indeed duplicated vertically
    C[n:,m:] *= Cv(_x, _y, _d, ld)
    #
    C *= eta**2
    
    return C


def kernel_2d_iso_u(x, xpr, params, C):
    """
    2D kernel, one velocity component
    
    Inputs:
        x: matrices input points [N,2]
        xpr: matrices output points [M,2]
        params: tuple length 2
            eta: standard deviation
            ld: spatial scale
            
    """
    eta, ld = params
    Cu = C
    
    # Build the covariance matrix
    _x = x[:,0,None] - xpr.T[:,0,None].T
    _y = x[:,1,None] - xpr.T[:,1,None].T
    _d = np.sqrt( _x**2 + _y**2 )
    C = Cu(_x, _y, _d, ld)
    C *= eta**2
    
    return C

def kernel_1d(x, xpr, params, C):
    """
    1D kernel - temporal
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 4
            eta: standard deviation
            lx: x length scale
            ly: y length scale
            lt: t length scale
            
    """
    eta, lt = params
    Ct = C
    
    # Build the covariance matrix
    C  = Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    C *= eta**2
    
    return C

def kernel_1d_uv(x, xpr, params, C):
    """
    temporal kernel only, two velocity components
    
    Inputs:
        x: matrices input points [N,3]
        xpr: matrices output points [M,3]
        params: tuple length 2
            u: standard deviation
            lt: t length scale
            
    """
    u, lt = params
    Ct = C
    
    # Build the covariance matrix
    n = x.shape[0]//2
    #
    C = np.ones((2*n,2*n))
    C *= Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    # decorrelates u/v components
    C[:n,n:] *= 0
    C[n:,:n] *= 0
    #
    C *= u**2
    
    return C

def kernel_1d_uv_traj(x, xpr, params, C):
    """
    temporal kernel only, two velocity components
    decorrelate data with different id
    
    Inputs:
        x: matrices input points [N,4]
        xpr: matrices output points [M,4]
        params: tuple length 2
            u: standard deviation
            lt: t length scale
            
    """
    u, lt = params
    Ct = C
    
    # Build the covariance matrix
    n = x.shape[0]//2
    #
    C = np.ones((2*n,2*n))
    C *= Ct(x[:,2,None], xpr.T[:,2,None].T, lt)
    # decorrelates u/v components
    C[:n,n:] *= 0
    C[n:,:n] *= 0
    # decorrelate different trajectories (moorings/drifters)
    C *= ( (x[:,3,None] - xpr.T[:,3,None].T)==0 ).astype(int)
    #
    C *= u**2
    
    return C

def generate_covariances(model, N, d, λ, xchunks=None):
    """ Generate spatial and temporal covariances
    used for synthetic flow generation (via streamfunction/potential)
    """
    
    Cs, Xs, Ns, isotropy = generate_spatial_covariances(
        model[0], N[0], N[1], d[0], d[1], λ[0], λ[1],
        chunks=xchunks,
    )

    Ct, Xt, Nt = generate_temporal_covariance(
        model[1], N[2], d[2], λ[2],
    )
    
    X = (*Xs, Xt)
    Ns = (*Ns, Nt)
    if isotropy:
        C = (Cs, Ct)
    else:
        C = (*Cs, Ct)

    return C, X, N, isotropy
        
        
def generate_spatial_covariances(
    model, Nx, Ny, dx, dy, λx, λy,
    chunks=None,
):
    """ Generate spatial covariances"""
            
    print(f"Space covariance model: {model}")
    
    N = (Nx, Ny)
    t_x = np.arange(Nx)[:, None]*dx
    t_y = np.arange(Ny)[:, None]*dy

    if chunks is not None:
        t_x = da.from_array(t_x, chunks=(chunks[0],1))
        t_y = da.from_array(t_y, chunks=(chunks[1],1))

    # space
    Cov_x, Cov_y, Cov_d = (None,)*3
    isotropy=False
    if model == "matern12_xy":
        Cov_x = cov.matern12(t_x, t_x.T, λx) # -2 spectral slope
        Cov_y = cov.matern12(t_y, t_y.T, λy) # -2 spectral slope
    elif model == "matern32_xy":
        Cov_x = cov.matern32(t_x, t_x.T, λx) # -4 spectral slope
        Cov_y = cov.matern32(t_y, t_y.T, λy) # -4 spectral slope
    elif model == "matern2_xy":
        Cov_x = cov.matern_general(np.abs(t_x - t_x.T), 1., 2, λx) # -5 spectral slope
        Cov_y = cov.matern_general(np.abs(t_y - t_y.T), 1., 2, λy) # -5 spectral slope
    elif model == "matern52_xy":
        Cov_x = cov.matern52(t_x, t_x.T, λx) # -6 spectral slope
        Cov_y = cov.matern52(t_y, t_y.T, λy) # -6 spectral slope
    elif model == "expquad":
        jitter = 1e-10
        Cov_x = cov.expquad(t_x, t_x.T, λx) + 1e-10 * np.eye(Nx)
        Cov_y = cov.expquad(t_y, t_y.T, λy) + 1e-10 * np.eye(Nx)
        # relative amplitude of the jitter:
        #    - on first derivative: np.sqrt(jitter) * λx/dx
        #    - on second derivative: np.sqrt(jitter) * (λx/dx)**2
        # with jitter = -10, λx/dx=100, these signatures are respectively: 1e-3 and 1e-1

    C = (Cov_x, Cov_y)

    # for covariances based on horizontal distances
    isotropy = ("iso" in model)
    if isotropy:
        t_x2 = (t_x   + t_y.T*0).ravel()[:, None]
        t_y2 = (t_x*0 + t_y.T  ).ravel()[:, None]
        t_xy = np.sqrt( (t_x2 - t_x2.T)**2 + (t_y2 - t_y2.T)**2 )        
        if model == "matern2_iso":
            Cov_d = cov.matern_general(t_xy, 1., 2, λx)
            C = Cov_d
        elif model == "matern32_iso":
            Cov_d = cov.matern32(t_xy, 0., λx) # -4 spectral slope
            C = Cov_d
        elif model == "matern52_iso":
            Cov_d = cov.matern52(t_xy, 0., λx) # -5 spectral slope
            C = Cov_d
        else:
            assert False, model+" is not implemented"
        
    # Input data points
    xd = (np.arange(0,Nx)[:,None]-1/2)*dx
    yd = (np.arange(0,Ny)[:,None]-1/2)*dy
    X = xd, yd
    
    return C, X, N, isotropy


def generate_temporal_covariance(model, Nt, dt, λt):
    """ Generate temporal covariance"""
    
    print(f"Covariance models: time={model}")
    
    t_t = np.arange(Nt)[:, None]*dt

    if model=="matern12":
        Cov_t = cov.matern12(t_t, t_t.T, λt) # -2 high frequency slope
    elif model=="matern32":
        Cov_t = cov.matern32(t_t, t_t.T, λt) # -4 high frequency slope
        
    # Input data points
    td = (np.arange(0,Nt)[:,None]-1/2)*dt
    
    return Cov_t, td, Nt




# ------------------------------------- synthetic field generation --------------------------------------

def generate_uv(
    kind, N, C, X, amplitudes, 
    noise=0., 
    dask_time=True, 
    #time=True, isotropy=False,
    time_chunk=5,
    seed=1234,
):
    """ Generate velocity fields
    
    Parameters
    ----------
    kind: str
        "uv": generates u and v independantly
        "pp": generates psi (streamfunction) and (phi) independantly from which velocities are derived
    N: tuple
        Grid dimension, e.g.: (Nx, Ny, Nt), or (Nx, Ny)
    C: tuple
        Covariance arrays, e.g.: (Cov_x, Cov_y, Cov_t) or (Cov_x, Cov_y) or Cov_d
    xyt: tuple
        xd, yd, td coordinates
    amplitudes: tuple
        amplitudes (u/v or psi, phi) as a size two tuple
    dask_time: boolean, optional
        activate dask distribution along time
    time: boolean, optional
        activate generation of time series
    isotropy: boolean, optional
        horizontally isotropic formulation
    seed: int, optional
        random number generation seed
    """
    
    # massage inputs
    time, isotropy = None, None
    if len(N)==2:
        time=False
        if not isinstance(C, tuple):
            isotropy=True
            Cov_x = C
        else:
            isotropy=False
            Cov_x, Cov_y = C
        xd, yd = X
        coords = dict(x=("x", xd[:,0]), y=("y", yd[:,0]))
    else:
        time=True
        if len(C)==2:
            isotropy=True
            Cov_x, Cov_t = C
        elif len(C)==3:
            isotropy=False
            Cov_x, Cov_y, Cov_t = C
        xd, yd, td = X
        coords = dict(x=("x", xd[:,0]), y=("y", yd[:,0]), time=("time", td[:,0]))
        
    assert (time is not None) and (isotropy is not None), "input data not consistent with implementation"
    print(f" time = {time},  isotropy = {isotropy}  ")

    # prepare output dataset
    ds = xr.Dataset(coords=coords)
    ds["x"].attrs["units"] = "km"
    ds["y"].attrs["units"] = "km"
    if time:
        ds["time"].attrs["units"] = "days"

    # perform Cholesky decompositions
    if isinstance(Cov_x, np.ndarray):
        chol = np.linalg.cholesky
        dask_space = False
    elif isinstance(Cov_x, da.core.Array):
        chol = lambda x: da.linalg.cholesky(x, lower=True)
        dask_space = True
    Lx = chol(Cov_x)
    if not isotropy:
        Ly = chol(Cov_y)
    # start converting to dask arrays
    if time:
        Lt = np.linalg.cholesky(Cov_t)
        #Lt_dask = da.from_array(Lt).persist()
        if time_chunk is not None and time_chunk>0:
            dask_time = True
            if not dask_space:
                Lx = da.from_array(Lx, chunks=(-1, -1))
            Lt = da.from_array(Lt, chunks=(time_chunk, -1)).persist()
            t_chunk_contraction = Lt.chunksize[1]
        else:
            t_chunk_contraction = -1
    if isinstance(Lx, da.core.Array):
        Lx = Lx.persist()
        x_chunk_contraction = Lx.chunksize[1]
    else:
        x_chunk_contraction = -1

    # generate sample
    rstate = da.random.RandomState(seed)
    np.random.seed(seed)
    u0_noise, u1_noise = 0., 0.
    if time and not isotropy and dask_time:
        assert False, "need update"
        _chunks = (-1, x_chunk_contraction, t_chunk_contraction)
        U0 = rstate.normal(0, 1, size=N, chunks=_chunks)
        U1 = rstate.normal(0, 1, size=N, chunks=_chunks)
        if noise>0:
            u0_noise = noise * rstate.normal(0, 1, size=N, chunks=_chunks)
            u1_noise = noise * rstate.normal(0, 1, size=N, chunks=_chunks)
    elif time and not isotropy and not dask_time:
        assert False, "Not implemented"
    elif time and isotropy and dask_time:
        _N = (N[0]*N[1], N[2])
        _chunks = (x_chunk_contraction, t_chunk_contraction)
        U0 = rstate.normal(0, 1, size=_N, chunks=_chunks)
        U1 = rstate.normal(0, 1, size=_N, chunks=_chunks)
        if noise>0:
            u0_noise = noise * rstate.normal(0, 1, size=_N, chunks=_chunks)
            u1_noise = noise * rstate.normal(0, 1, size=_N, chunks=_chunks)
    elif time and isotropy and not dask_time:
        assert False, "Not implemented"
    # not time
    elif not time and not isotropy and dask_space:
        assert False, "need update"
        _chunks = (-1, x_chunk_contraction,)
        U0 = rstate.normal(0, 1, size=(N[0], N[1]), chunks=_chunks)
        U1 = rstate.normal(0, 1, size=(N[0], N[1]), chunks=_chunks)
        if noise>0:
            u0_noise = noise * rstate.normal(0, 1, size=(N[0], N[1]), chunks=_chunks)
            u1_noise = noise * rstate.normal(0, 1, size=(N[0], N[1]), chunks=_chunks)
    elif not time and not isotropy and not dask_space:
        U0 = np.random.normal(0, 1, size=(N[0], N[1]))
        U1 = np.random.normal(0, 1, size=(N[0], N[1]))
        if noise>0:
            u0_noise = noise * np.random.normal(0, 1, size=(N[0], N[1]))
            u1_noise = noise * np.random.normal(0, 1, size=(N[0], N[1]))
    elif not time and isotropy and dask_space:
        _chunks = (x_chunk_contraction,)
        U0 = rstate.normal(0, 1, size=(N[0]*N[1],), chunks=_chunks)
        U1 = rstate.normal(0, 1, size=(N[0]*N[1],), chunks=_chunks)
        if noise>0:
            u0_noise = noise * rstate.normal(0, 1, size=(N[0]*N[1],), chunks=_chunks)
            u1_noise = noise * rstate.normal(0, 1, size=(N[0]*N[1],), chunks=_chunks)
    elif not time and isotropy and not dask_space:
        U0 = np.random.normal(0, 1, size=(N[0]*N[1],))
        U1 = np.random.normal(0, 1, size=(N[0]*N[1],))
        if noise>0:
            u0_noise = noise * np.random.normal(0, 1, size=(N[0]*N[1],))
            u1_noise = noise * np.random.normal(0, 1, size=(N[0]*N[1],))
    
    # 2D
    #zg = η * Lx @ V @ Lt.T
    # 3D
    # with dask
    #zg = η * da.einsum("ij,kl,mn,jln->ikm", Lx, Ly, Lt_dask, V)
    # with opt_einsum
    import opt_einsum as oe
    if time and not isotropy:
        u0 = amplitudes[0] * oe.contract("ij,kl,mn,jln", Lx, Ly, Lt, U0)
        u1 = amplitudes[1] * oe.contract("ij,kl,mn,jln", Lx, Ly, Lt, U1)
    elif time and isotropy:
        u0 = amplitudes[0] * oe.contract("ij,kl,jl", Lx, Lt, U0)
        u1 = amplitudes[1] * oe.contract("ij,kl,jl", Lx, Lt, U1)            
    elif not time and not isotropy:
        u0 = amplitudes[0] * oe.contract("ij,kl,jl", Lx, Ly, U0)
        u1 = amplitudes[1] * oe.contract("ij,kl,jl", Lx, Ly, U1)
    elif not time and isotropy:
        u0 = amplitudes[0] * Lx @ U0
        u1 = amplitudes[1] * Lx @ U1
            
    # add noise
    u0 = u0 + u0_noise
    u1 = u1 + u1_noise
    
    _u0 = u0
    
    if isotropy and time:
        # final reshaping required
        u0 = u0.reshape(N)
        u1 = u1.reshape(N)
    elif isotropy and not time:
        # final reshaping required
        u0 = u0.reshape(N[0], N[1])
        u1 = u1.reshape(N[0], N[1])
        
    if time:
        dims = ("x", "y", "time")
    else:
        dims = ("x", "y",)
    if kind=="uv":
        ds["u"] = (dims, u0)
        ds["v"] = (dims, u1)
    elif kind=="pp":
        ds["psi"] = (dims, u0)
        ds["phi"] = (dims, u1)
        # rederive u
        dpsidx = ds.psi.differentiate("x")
        dpsidy = ds.psi.differentiate("y")
        dphidx = ds.phi.differentiate("x")
        dphidy = ds.phi.differentiate("y")
        ds["u"] = -dpsidy + dphidx
        ds["v"] =  dpsidx + dphidy

    ds = ds.transpose(*reversed(dims))
    ds.attrs.update(kind=kind)

    return ds

#
def amplitude_decifit(dx, λ, ν):
    """ scale of the rms deficit associted after second order differenciation  """
    S2 = λ**2*(ν-1)/ν * (1 - matern_general(2*dx, 1., ν, λ))/2/dx**2
    return np.sqrt(S2)


# ------------------------------------- inference -----------------------------------------

def prepare_inference(
    run_dir,
    uv, no_time, no_space, 
    parameter_eta_formulation, traj_decorrelation,
    enable_nu,
):

    # load eulerian flow
    flow_file = os.path.join(run_dir, "flow.zarr")
    dsf = xr.open_zarr(flow_file)
    dsf["time"] = dsf["time"]/pd.Timedelta("1D")
    U = dsf.attrs["U"] # useful for computation of alpha much latter        
    
    # problem parameters (need to be consistent with data generation notebook)
    p = dsf.attrs
    η = p["eta"]  # streamfunction amplitude
    #
    λx = p["lambda_x"]   # km
    λy = p["lambda_y"]   # km
    λt = p["lambda_t"]   # days
    #
    νs = p["nu_space"]
    νt = p["nu_time"]

    # derived velocity parameter
    γ = η / λx 

    # add grid information
    #Lx = float(dsf.x.max()-dsf.x.min())
    #Ly = float(dsf.y.max()-dsf.y.min())
    p["dx"] = float(dsf.x.diff("x").median())
    p["dy"] = float(dsf.y.diff("y").median())
    # km, km this should not matter

    # get 1D covariances
    C, Ct, isotropy = get_cov_1D(p["cov_x"], p["cov_t"], enable_nu)
    Cu, Cv, Cuv = C
    #if "matern32" in p["cov_x"]:
    #    p["nu_space"] = 3/2
    #else:
    #    assert False, "no nu could be infered"
    #if "matern12" in p["cov_t"]:
    #    p["nu_time"] = 1/2
    #else:
    #    assert False, "no nu could be infered"

    # correct for bias due to spatial differentiation and (psi to u/v)
    #_scale = amplitude_decifit(p["dx"], p["lambda_x"], p["nu_space"])
    #η *= _scale
    #γ *= _scale
    #p["eta"] = η
    #p["amplitude_deficit"] = _scale

    for k, v in p.items():
        print(k, v)
    
    # set covariance parameters
    if no_time:
        if uv:
            covfunc = lambda x, xpr, params: kernel_2d_iso_uv(x, xpr, params, (Cu, Cv, Cuv))
        else:
            covfunc = lambda x, xpr, params: kernel_2d_iso_u(x, xpr, params, Cu)
        covparams = [η, λx]
        labels = ['σ','η','λx',]
    elif no_space:
        if uv:
            if traj_decorrelation:
                covfunc = lambda x, xpr, params: kernel_1d_uv_traj(x, xpr, params, Ct)
            else:
                covfunc = lambda x, xpr, params: kernel_1d_uv(x, xpr, params, Ct)
        else:
            assert False, "not implemented"
        covparams = [U, λt]
        labels = ['σ','u','λt']
    elif isotropy:
        if uv:
            if parameter_eta_formulation:
                covfunc = lambda x, xpr, params: kernel_3d_iso_uv(x, xpr, params, (Cu, Cv, Cuv, Ct))
                covparams = [η, λx, λt]
                labels = ['σ','η','λx','λt']
            else:
                def covfunc(x, xpr, params):
                    # params contains (gamma=eta/ld, ld, lt) and needs to be converted to (eta, ld, lt)
                    params = (params[0]*params[1], *params[1:])
                    if traj_decorrelation:
                        return kernel_3d_iso_uv_traj(x, xpr, params, (Cu, Cv, Cuv, Ct))
                    else:
                        return kernel_3d_iso_uv(x, xpr, params, (Cu, Cv, Cuv, Ct))
                if enable_nu:
                    covparams = [γ, λx, λt, νs, νt]
                    labels = ['σ','γ','λx','λt', 'νs', 'νt']
                else:
                    covparams = [γ, λx, λt]
                    labels = ['σ','γ','λx','λt']
        else:
            covfunc = lambda x, xpr, params: kernel_3d_iso_u(x, xpr, params, (Cu, Ct))
            covparams = [η, λx, λt]
            labels = ['σ','η','λx','λt']
    else:
        covfunc = lambda x, xpr, params: kernel_3d(x, xpr, params, (Cx, Cy, Ct))
        covparams = [η, λx, λy, λt]
        labels = ['σ','η','λx','λy','λt']
    return dsf, covfunc, covparams, labels

## emcee
def inference_emcee(
    X, U, 
    noise, covparams,
    covfunc, labels,
    isotropy=True, no_time=False,
    **kwargs,
):
        
    # Initial guess of the noise and covariance parameters (these can matter)
    if no_time:
        η, λx = covparams
        noise_prior      = gpstats.truncnorm(noise, noise*2, noise/10, noise*10)     # noise
        covparams_priors = [gpstats.truncnorm(η, η*2, η/10, η*10),                   # eta
                            gpstats.truncnorm(λx, λx*2, λx/10, λx*10),               # λx
                           ]
    elif isotropy:
        η, λx, λt = covparams
        noise_prior      = gpstats.truncnorm(noise, noise*2, noise/10, noise*10)     # noise
        covparams_priors = [gpstats.truncnorm(η, η*2, η/10, η*10),                   # eta
                            gpstats.truncnorm(λx, λx*2, λx/10, λx*10),               # λx
                            gpstats.truncnorm(λt, λt*2, λt/10, λt*10),               # λt
                           ]
    else:
        η, λx, λy, λt = covparams
        noise_prior      = gpstats.truncnorm(noise, noise*2, noise/10, noise*10)     # noise
        covparams_priors = [gpstats.truncnorm(η, η*2, η/10, η*10),                   # eta
                            gpstats.truncnorm(λx, λx*2, λx/10, λx*10),               # λx
                            gpstats.truncnorm(λy, λy*2, λy/10, λy*10),               # λy
                            gpstats.truncnorm(λt, λt*2, λt/10, λt*10),               # λt
                           ]

    samples, log_prob, priors_out, sampler = mcmc.mcmc(
        X,
        U,
        covfunc,
        covparams_priors,
        noise_prior,
        nwarmup=100,
        niter=100,
        verbose=False,
    )

    # 40 points
    # with bessels: 2min30, 1.6s / iteration
    # without bessels: 19s, 5 iterations / second
    # with one bessel: 1min05 , 1.5 iteration / second
    # with eye instead of bessel: 20s, 4.7 iterations / second

    # mattern32 - analytical:      24s , 4 iterations per second
    # mattern32 - bessel:      1min32s , 1 iteration  per second

    # 1000 points, mattern32 - analytical: 1 hour total - 30s / iteration

    # should also store prior information
    ds = xr.Dataset(
        dict(samples=(("i", "parameter"), samples),
             priors=(("j", "parameter"), priors_out),
             log_prob=("i", log_prob.squeeze()),
            ),
        coords=dict(parameter=labels)
    )
    ds.attrs["inference"] = "emcee"

    # MAP
    i_map = int(ds["log_prob"].argmax())
    ds["MAP"] = ds["samples"].isel(i=i_map)
    ds.attrs["i_MAP"] = i_map
    #i = np.argmax(log_prob)
    #MAP = samples[i, :]

    return ds

# MH of inference

def inference_MH(
    X, U,
    noise, covparams,
    covfunc, labels,
    n_mcmc = 20_000,
    steps = None,
    tqdm_disable=False,
    no_time=False, no_space=False,
    **kwargs,
):

    # number of parameters infered
    N = len(labels)

    # default step sizes
    if steps is None:
        # ** isn't this too coarse ? **
        steps = [1/5]*(N-1) + [1/2]
    
    # The order of everything is eta, ld, lt, noise
    #step_sizes = np.array([.5, 5, .5, 0.005])
    #initialisations = np.array([12, 100, 5, 0.01])
    initialisations = np.array(covparams+[noise])
    step_sizes = np.array(
        [v*s for v, s in zip(initialisations, steps)]
    )
    lowers = np.repeat(0, N)
    uppers = initialisations * 10

    # setup objects
    samples = [np.empty(n_mcmc) for _ in range(N)]
    accept_samples = np.empty(n_mcmc)
    lp_samples = np.empty(n_mcmc)
    lp_samples[:] = np.nan
    # init samples
    for i, s in enumerate(samples):
        s[0] =  initialisations[i]
    accept_samples[0] = 0
    #
    covparams_prop = initialisations.copy()[0:N-1]
    # run mcmc
    gp_current = GPtideScipy(X, X, noise, covfunc, covparams)

    for i in tqdm(np.arange(1, n_mcmc), disable=tqdm_disable):
        
        proposed = np.array([
            np.random.normal(s[i-1], step, 1)
            for s, step in zip(samples, step_sizes)
        ])

        if ((proposed.T <= lowers) | (proposed.T >= uppers)).any():
            for s in samples:
                s[i] = s[i-1]
            lp_samples[i] = lp_samples[i-1]
            accept_samples[i] = 0
            continue

        if accept_samples[i-1] == True:
            gp_current = gp_proposed

        covparams_prop = proposed[:-1]
        gp_proposed = GPtideScipy(X, X, proposed[-1], covfunc, covparams_prop)

        lp_current = gp_current.log_marg_likelihood(U)
        lp_proposed = gp_proposed.log_marg_likelihood(U)

        alpha = np.min([1, np.exp(lp_proposed - lp_current)])
        u = np.random.uniform()

        if alpha > u:
            for s, p in zip(samples, proposed):
                s[i] = p
            accept_samples[i] = 1
            lp_samples[i] = lp_proposed
        else:
            for s, p in zip(samples, proposed):
                s[i] = s[i-1]
            accept_samples[i] = 0
            lp_samples[i] = lp_samples[i-1]

    samples = np.vstack([samples[-1]]+samples[:-1])
    ds = xr.Dataset(
        dict(
            samples=(("i", "parameter"), samples.T), 
            accept=("i", accept_samples),
            log_prob=("i", lp_samples),
            init=(("parameter",), np.roll(initialisations,1)), # need to swap parameter orders
            lower=(("parameter",), np.roll(lowers,1)),
            upper=(("parameter",), np.roll(uppers,1)),
        ),
        coords=dict(parameter=labels)
    )

    #return noise_samples, eta_samples, ld_samples, lt_samples, 
    accepted_fraction = float(ds["accept"].mean())
    print(f"accepted fraction = {accepted_fraction*100:.1f} %")

    # keep only accepted samples
    #ds = ds.where(ds.accept==1, drop=True)

    # MAP
    i_map = int(ds["log_prob"].argmax())
    ds["MAP"] = ds["samples"].isel(i=i_map)

    ds.attrs["accepted_fraction"] = accepted_fraction
    ds.attrs["inference"] = "MH"
    ds.attrs["i_MAP"] = i_map

    return ds

def select_traj_core(ds, Nxy, dx):
    if "time" in ds:
        ds = ds.isel(time=0)
    tolerance = .1
    if dx is not None:
        assert Nxy>1, "Nxy must be >1 if dx is specified"
        if Nxy==2:
            # select in a circular shell around the first point
            traj_selection = [np.random.choice(ds.trajectory.values, 1)[0]]
            p0 = ds.sel(trajectory=traj_selection[0])[["x", "y"]]
            d = np.sqrt( (ds["x"]-float(p0.x))**2 + (ds["y"]-float(p0.y))**2 )
            d = d.where( (d>dx*(1-tolerance)) & (d<dx*(1+tolerance)), drop=True )
            if d.trajectory.size==0:
                return
            traj_selection.append(np.random.choice(d.trajectory.values, 1)[0])
        else:
            #Nxy==3: equilateral triangle
            assert False, "Not implemented"
    else:
        traj_selection = np.random.choice(ds.trajectory.values, Nxy, replace=False)
    return traj_selection

def select_traj(*args, repeats=5):
    i, fail = 0, True
    while i<repeats and fail:
        s = select_traj_core(*args)
        if s is not None:
            fail = False
        i+=1
    if fail:
        assert False, "could not select trajectories"
    return s

def mooring_inference_preprocess(
    ds, seed, N, 
    dx=None, 
    **kwargs,
):

    Nt, Nxy = N
    
    # set random seed - means same mooring positions will be selected across different flow_scales
    np.random.seed(seed)
    # subsample temporally
    ds = ds.isel(time=np.linspace(0, ds.time.size-1, Nt, dtype=int))
    
    # randomly select mooring location
    ds = ds.stack(trajectory=["x", "y"])
    traj_selection = select_traj(ds[["x", "y"]], Nxy, dx)
    ds = ds.sel(trajectory=traj_selection)
    ds["trajectory_local"] = ("trajectory", np.arange(ds.trajectory.size))
    
    return ds.compute()
    

def mooring_inference(
    dsf, seed,
    covparams, covfunc, labels, N, noise,
    inference="MH", uv=True,
    flow_scale=None, dx=None,
    write=None, overwrite=True,
    preprocessed=False,
    **kwargs,
):
    """ run inference for moorings """

    Nt, Nxy = N

    # not used anymore
    if write is not None:
        run_dir, _ = write
        output_file = os.path.join(
            run_dir,
            f"moorings_s{seed}_Nxy{Nxy}.nc",
        )
        if flow_scale is not None:
            output_file = output_file.replace(".nc", f"_fs{flow_scale:.2f}.nc")
        if os.path.isfile(output_file) and not overwrite:
            return None
    
    if not preprocessed:
        # needs to be taken out when parallelized
        ds = mooring_inference_preprocess(
            dsf, seed, N, dx=dx,
        )
    else:
        ds = dsf

    u_scale = 1.
    if flow_scale is not None:
        u_scale = flow_scale
    # update noise covparams
    noise = noise * u_scale
    covparams[0] = covparams[0] * u_scale # amplitude
    
    # set up inference
    u, v, x, y, t, traj = xr.broadcast(ds.u, ds.v, ds.x, ds.y, ds.time, ds.trajectory_local)
    assert u.shape==v.shape==x.shape==y.shape==t.shape==traj.shape
    x = x.values.ravel()
    y = y.values.ravel()
    t = t.values.ravel()
    traj = traj.values.ravel()
    #X = np.hstack([x[:,None], y[:,None], t[:,None]])
    X = np.hstack([x[:,None], y[:,None], t[:,None], traj[:,None]])
    u = u.values.ravel()[:, None] * u_scale
    v = v.values.ravel()[:, None] * u_scale
    # add noise
    u += np.random.randn(*u.shape) * noise
    v += np.random.randn(*v.shape) * noise
    if uv:
        X = np.vstack([X, X])
        U = np.vstack([u, v])
    else:
        U = u        
        
    # reset seed here
    np.random.seed(seed)
    
    # run
    if inference=="MH":
        ds = inference_MH(
            X, U, noise, covparams, covfunc, labels, 
            **kwargs,
        )
    elif inference=="emcee":
        ds = inference_emcee(
            X, U, noise, covparams, covfunc, labels, 
            **kwargs,
        )
    ds["true_parameters"] = ("parameter", np.array([noise]+covparams))
    ds["seed"] = seed
    if flow_scale is not None:
        ds["flow_scale"] = flow_scale
    
    # store or not and return
    if write is not None:
        ds.to_netcdf(output_file, mode="w")
        return output_file
    else:
        return ds

def run_mooring_ensembles(
    Ne, 
    dsf,
    covparams, covfunc, labels, N, noise,
    step=1/5, **kwargs,
):
    """ wrap mooring_inference to run ensembles """

    dkwargs = dict(tqdm_disable=True, n_mcmc=20_000)
    dkwargs.update(**kwargs)

    # MH default
    dkwargs["steps"] = (step, step, step, 1/2)

    # preload data:
    D = [
        mooring_inference_preprocess(dsf, seed, N, **kwargs)
        for seed in range(Ne)
    ]
    mooring_inference_delayed = dask.delayed(mooring_inference)
    datasets = [
        mooring_inference_delayed(
            ds, seed, 
            covparams, covfunc, labels, N, noise, 
            preprocessed=True,
            **dkwargs,
        )
        for seed, ds in zip(range(Ne), D)
        #for seed in range(Ne)
    ]
    datasets = dask.compute(datasets)[0]
    ds = xr.concat(datasets, "ensemble")
    ds = ds.isel(i=slice(0,None,5)) # subsample MCMC
    return ds

def open_drifter_file(run_dir, flow_scale=None, filter=True, **kwargs):
    
    zarr_file = os.path.join(run_dir, f"drifters.zarr")
    if flow_scale is not None:
        zarr_file = zarr_file.replace(".zarr", f"_fs{flow_scale:.2f}.zarr")
    ds = xr.open_zarr(zarr_file).compute()
    
    ds = ds.rename(lon="x", lat="y")
    ds["x"] = ds["x"]/1e3
    ds["y"] = ds["y"]/1e3    
    ds = ds.assign_coords(t=ds["time"]/pd.Timedelta("1D"))

    # trajectory reaching the end of the simulation
    maxt = ds.t.max("obs").compute()
    Tmax = np.floor(float(ds.t.max()))
    print(f"Tmax = {Tmax:.2f} days")
    n0 = ds.trajectory.size
    dsf = ds.where( maxt>Tmax , drop=True)
    if filter:
        ds = dsf
    ns = dsf.trajectory.size
    survival_rate = ns/n0*100
    print(f"{survival_rate:.1f}% of trajectories survived")
    #
    dt = ds.t.differentiate("obs")*day
    ds["u"] = ds.x.differentiate("obs")/dt*1e3 # x are in km
    ds["v"] = ds.y.differentiate("obs")/dt*1e3 # y are in km
    #
    t = ds.t
    #ds = ds.drop(["t", "time"])
    ds = ds.drop(["time"])
    ds["obs"] = ds.t.max("trajectory")
    ds = (
        ds
        .drop("t")
        .rename(obs="time")
        .sel(time=slice(0,Tmax))
    )
    ds["survival_rate"] = survival_rate

    return ds

def drifter_preprocess(
    run_dir, N, seed, 
    dx = None,
    ds = None,
    **kwargs,
):
    
    Nt, Nxy = N
    
    # parcels dataset
    if ds is None:
        ds = open_drifter_file(run_dir, **kwargs)

    # set random seed
    np.random.seed(seed)
    
    # randomly select Nxy trajectories
    traj_selection = select_traj(ds, Nxy, dx)
    ds = ds.sel(trajectory=traj_selection)
    #ds["traj"] = ("trajectory", np.arange(ds.trajectory.size))

    # subsample temporally
    ds = ds.isel(time=np.linspace(0, ds.time.size-1, Nt, dtype=int))
    ds = ds.compute()

    return ds

def drifter_inference(
    run_dir, seed, 
    covparams, covfunc, labels, N, noise,
    inference="MH", uv=True, no_time=False,
    flow_scale=None, dx=None,
    write=None, overwrite=True,
    preprocessed=None,
    **kwargs,
):
    """ run inference for drifters """

    Nt, Nxy = N

    # not used anymore it seems
    if write is not None:
        #run_dir, _ = write
        output_file = os.path.join(
            run_dir,
            f"drifters_s{seed}_Nxy{Nxy}.nc",
        )
        if flow_scale is not None:
            output_file = output_file.replace(".nc", f"_f{flow_scale:.2f}.nc")    
        if os.path.isfile(output_file) and not overwrite:
            return

    if preprocessed is None:
        ds = drifter_preprocess(run_dir, N, seed, flowscale=flowscale)
    else:
        ds = preprocessed
    survival_rate = ds["survival_rate"]

    if flow_scale is not None:
        noise = noise * flow_scale # rescale noise
        # update covparams
        #covparams = covparams[:]
        covparams[0] = covparams[0] * flow_scale
    
    # massage inputs to inference problem
    if no_time:
        # this should not be necessary
        u, v, x, y = xr.broadcast(ds.u, ds.v, ds.x, ds.y)
        assert u.shape==v.shape==x.shape==y.shape
        x = x.values.ravel()
        y = y.values.ravel()
        X = np.hstack([x[:,None], y[:,None],])
    else:
        #u, v, x, y, t = xr.broadcast(ds.u, ds.v, ds.x, ds.y, ds.time)
        u, v, x, y, t, traj = xr.broadcast(ds.u, ds.v, ds.x, ds.y, ds.time, ds.trajectory)
        assert u.shape==v.shape==x.shape==y.shape==t.shape==traj.shape
        assert not np.isnan(u).any(), ("u nan",u)
        x = x.values.ravel()
        y = y.values.ravel()
        t = t.values.ravel()
        traj = traj.values.ravel()
        #X = np.hstack([x[:,None], y[:,None], t[:,None]])
        X = np.hstack([x[:,None], y[:,None], t[:,None], traj[:,None]])
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
    
    # reset seed here
    np.random.seed(seed)
    
    if inference=="MH":
        ds = inference_MH(
            X, U, noise, covparams, covfunc, labels, 
            no_time=no_time, 
            **kwargs,
        )
    elif inference=="emcee":
        ds = inference_emcee(
            X, U, noise, covparams, covfunc, labels, 
            no_time=no_time, 
            **kwargs,
        )
    ds["true_parameters"] = ("parameter", np.array([noise]+covparams))
    ds["seed"] = seed
    ds["survival_rate"] = survival_rate
    if flow_scale is not None:
        ds["flow_scale"] = flow_scale

    # store or not and return
    if write is not None:
        ds.to_netcdf(output_file, mode="w")
        return output_file
    else:
        return ds
    
def run_drifter_ensembles(
    run_dir, 
    Ne,
    covparams, covfunc, labels, N, noise,
    step=1/5,
    **kwargs,
):
    """ wrap drifter_inference to run ensembles """

    dkwargs = dict(tqdm_disable=True, n_mcmc=20_000)
    dkwargs.update(**kwargs)

    # MH
    dkwargs["steps"] = (step, step, step, 1/2)

    # preload data
    ds = open_drifter_file(run_dir, **kwargs)
    D = [
        drifter_preprocess(run_dir, N, seed, ds=ds, **kwargs)
        for seed in range(Ne)
    ]
    
    drifter_inference_delayed = dask.delayed(drifter_inference)
    datasets = [
        drifter_inference_delayed(
            run_dir,
            seed,
            covparams, covfunc, labels, N, noise,
            preprocessed=d,
            **dkwargs,
        ) 
        for d, seed in zip(D, range(Ne))
        #for seed in range(Ne)
    ]
    datasets = dask.compute(datasets)[0]
    ds = xr.concat(datasets, "ensemble")
    ds = ds.isel(i=slice(0,None,5)) # subsample MCMC
    return ds


# ------------------------------------- plotting ------------------------------------------

def plot_snapshot(ds, i=None, **kwargs):

    if i is not None:
        ds = ds.isel(time=i)
    
    if "psi" in ds:
        return plot_snapshot_pp(ds, **kwargs)
    

def plot_snapshot_pp(ds, darrow=20):
    
    fig, axes = plt.subplots(3,2,figsize=(15,15), sharex=True)
    
    dsa = ds.isel(x=slice(0,None,darrow), y=slice(0,None,darrow))

    ax = axes[0, 0]
    ds.psi.plot(ax=ax, cmap="RdBu_r")
    dsa.plot.quiver("x", "y", "u", "v", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("psi")

    ax = axes[0, 1]
    ds.phi.plot(ax=ax, cmap="RdBu_r")
    dsa.plot.quiver("x", "y", "u", "v", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("phi")
    
    ##
    ax = axes[1, 0]
    ds.u.plot(ax=ax, cmap="RdBu_r")
    #dsa.plot.quiver("x", "y", "u", "v", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("u")

    ax = axes[1, 1]
    ds.v.plot(ax=ax, cmap="RdBu_r")
    #dsa.plot.quiver("x", "y", "u", "v", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("v")
    
    ##
    divergence = ds.u.differentiate("x")/1e3 + ds.v.differentiate("y")/1e3
    vorticity = ds.v.differentiate("x")/1e3 - ds.u.differentiate("y")/1e3
    
    ax = axes[2, 0]
    divergence.plot(ax=ax, cmap="RdBu_r")
    #dsa.plot.quiver("x", "y", "u", "v", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("divergence")

    ax = axes[2, 1]
    vorticity.plot(ax=ax, cmap="RdBu_r")
    #dsa.plot.quiver("x", "y", "u", "v", ax=ax)
    ax.set_aspect("equal")
    ax.set_title("vorticity")    
    
    return fig, axes



def plot_spectra(ds, v, yref=1e-1, slopes=[-4,-5,-6], **kwargs):
    
    # compute spectra
    dkwargs = dict(dim=['x','y'], detrend='linear', window=True)
    E = xrft.power_spectrum(ds[v], **kwargs)
    E = E.compute()
    E_iso = xrft.isotropic_power_spectrum(ds[v], truncate=True, **dkwargs)
    print(E_iso)
    if "time" in E.dims:
        E = E.mean("time")
        E_iso = E_iso.mean("time")
    E = E.compute()    
    E_iso = E_iso.compute()    
    
    # plot in kx-ky space
    _E = E.where( (E.freq_x>0) & (E.freq_y>0), drop=True )
    fig, ax = plt.subplots(1,1)
    np.log10(_E).plot(**kwargs)
    np.log10(_E).plot.contour(levels=[-8, -4, 0], colors="w", linestyles="-")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{v}: kx-ky power spectrum")
    
    # plot isotropic
    #fy = 1e-3
    #_Ex = E.sel(freq_y=fy, method="nearest")
    #_Ex = _Ex.where(_Ex.freq_x>0, drop=True)

    fig, ax = plt.subplots(1,1)
    E_iso.plot(label="iso", lw=4, zorder=10)
    #_Ex.plot(label=f"E(f_y={fy:.1e})")
    #np.log10(_E).plot.contour(levels=[-8, -4, 0], colors="w", linestyles="-")

    _f = np.logspace(-2.5, min(-.5, float(np.log10(E_iso.freq_r.max()))), 10)
    for s in slopes:
        ax.plot(_f, yref * (_f/_f[0])**s, color="k")
        ax.text(_f[-1], yref * (_f[-1]/_f[0])**s, r"$f^{}$".format(int(s)))
    #ax.plot(_f, yref * (_f/_f[0])**-4, color="k")
    #ax.text(_f[-1], yref * (_f[-1]/_f[0])**-4, r"$f^{-3}$")
    #ax.plot(_f, yref * (_f/_f[0])**-6, color="k")
    #ax.text(_f[-1], yref * (_f[-1]/_f[0])**-6, r"$f^{-6}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"{v}: isotropic spectrum")

## inference result
def convert_to_az(d, labels, burn=0):
    output = {}
    for ii, ll in enumerate(labels):
        output.update({ll:d[burn:,ii]})
    return az.convert_to_dataset(output)

def plot_inference(ds, stack=False, corner_plot=True, xlim=True, burn=None):

    labels = ds["parameter"].values
    
    #samples = ds.samples.values
    #samples_az = convert_to_az(samples, labels)
    if burn:
        ds = ds.isel(i=slice(burn, None))
    if stack:
        samples = ds.stack(points=("i", "ensemble")).samples.values.T
        samples_az = convert_to_az(samples, labels)
        density_data = [samples_az[labels],]
    else:
        samples = [ds.samples.sel(ensemble=e) for e in ds.ensemble]
        samples_az = [convert_to_az(s, labels) for s in samples]
        density_data = [s[labels] for s in samples_az]
    
    #density_labels = ['posterior',]
    
    axs = az.plot_density(   density_data,
                             shade=0.1,
                             grid=(1, 5),
                             textsize=12,
                             figsize=(15,4),
                             #data_labels=tuple(density_labels),
                             hdi_prob=0.995)

    if "ensemble" in ds.dims:
        cov_params = ds.true_parameters.isel(ensemble=0).values
    else:
        cov_params = ds.true_parameters.values
    
    i=0
    _ds = ds.sel(ensemble=0)
    #for t, ax in zip([noise,]+list(covparams), axs[0]):
    for t, ax in zip(cov_params, axs[0]):
        #print(t, ax)
        ax.axvline(t, color="k", ls="-", label="truth") # true value
        if isinstance(xlim, tuple):
            ax.set_xlim(*xlim)
        elif xlim:
            ax.set_xlim(0, t*2)
        if i==0:
            ax.legend()
        i+=1 
    
    if corner_plot:
        samples = ds.stack(points=("i", "ensemble")).samples.values.T
        fig = corner.corner(
            samples, 
            show_titles=True,
            labels=labels,
            plot_datapoints=True,
            quantiles=[0.16, 0.5, 0.84],
        )
    
def traceplots(ds, MAP=True, burn=None):

    if burn:
        ds = ds.isel(i=slice(burn, None))
    
    fig, axes = plt.subplots(2,2, sharex=True, figsize=(15,6))
    
    for v, ax in zip(ds["parameter"], axes.flatten()[:ds.parameter.size]):
        ds.samples.sel(parameter=v).plot(ax=ax, hue="ensemble", add_legend=False)
        #if MAP:
        #    ax.axvline(ds.attrs["i_MAP"], color="b", lw=2)
        ax.set_title(v.values)
        ax.set_ylabel("")
        #print(v.values)

    fig, ax = plt.subplots(1,1, figsize=(15,3))
    ds["log_prob"].plot(ax=ax, hue="ensemble")
    #if MAP:
    #    ax.axvline(ds.attrs["i_MAP"], color="b", lw=2)
    ax.set_ylabel("")
    ax.set_title("log_prob")

# combined plot of inference performance
def plot_sensitivity_combined(
    dsm, dsr, MAP=True, 
    x=None, xlog2=False, x_width=0.2, x_off=0., x_scale=1., x_ticks_free=True, x_label=None,
    alpha=None, c=None,
    bounds=None,
    label=None,
    type="shading",
    velocity_deficit=True,
    legend_loc=0,
    alpha_normalize=None,
    **kwargs,
):
    
    fig, axes = plt.subplot_mosaic(
        [['(a)', '(b)', '(c)', '(d)']],
        layout='constrained',
        figsize=(10,3),
        dpi=300,
    )

    b0, b1 = None, None

    # x positions
    _x = dsm[x].values
    if xlog2:
        width = lambda p, w: inverse2(forward2(p)+w/2)-inverse2(forward2(p)-w/2)
        widths = width(_x, x_width) # 0.2
        x_scale = 1.2
    else:
        widths = x_width
    # other properties
    boxkwargs = dict(
        manage_ticks=False,
        showfliers=False,
        widths=widths, # 1
        patch_artist=True,
        medianprops=dict(color="k", lw=2),
    )

    labels = dsm.parameter.values
    labels = [r"${}_{}$".format(l[0], "s") if l=="λx" else l for l in labels]
    labels = [r"${}_{}$".format(l[0], "t") if l=="λt" else l for l in labels]

    if x_label is None:
        x_label = x
        
    # moorings
    ds = dsm
    if c is None:
        c = colors["mo"]
    for p, k, l in zip(ds.parameter.values, axes, labels):
        ax = axes[k]
        if MAP:
            da = ds["MAP"].sel(parameter=p)
        else:
            da = (ds["samples"]
                .sel(parameter=p)
                .isel(i=slice(burn,None))
                .mean("i")
            )
        if (p=="γ" or p=="η" or p=="u") and alpha_normalize:
            l = p+"/α"
            da = (da/da["α"]).rename(p+"/α")
        if type=="boxplot":
            _h = _boxplot(ax, da.values.T, da[x].values, boxprops=dict(facecolor=c, alpha=alpha), **boxkwargs, **kwargs)
        elif type=="shading":
            _h = _shadeplot(ax, da, da[x].values, color=c, alpha=alpha, **kwargs)
        if b0 is None:
            b0 = _h

    # drifters
    if dsr is not None:
        ds, c = dsr, colors["dr"]
        
        for p, k, l in zip(ds.parameter.values, axes, labels):
            ax = axes[k]
            
            if MAP:
                da = ds["MAP"].sel(parameter=p)
            else:
                da = (ds["samples"]
                    .sel(parameter=p)
                    .isel(i=slice(burn,None))
                    .mean("i")
                )
            if (p=="γ" or p=="η" or p=="u") and alpha_normalize:
                l = p+"/α"
                da = (da/da["α"]).rename(p+"/α")
            if type=="boxplot":
                _h = _boxplot(ax, da.values.T, da[x].values*x_scale+x_off, boxprops=dict(facecolor=c, alpha=alpha), **boxkwargs, **kwargs)
            elif type=="shading":
                _h = _shadeplot(ax, da, da[x].values, color=c, alpha=alpha, **kwargs)
            if b1 is None:
                b1 = _h

    # misc info
    for p, k, l in zip(ds.parameter.values, axes, labels):
        ax = axes[k]
        if MAP:
            da = ds["MAP"].sel(parameter=p)
        else:
            da = (ds["samples"]
                .sel(parameter=p)
                .isel(i=slice(burn,None))
                .mean("i")
            )
        if xlog2:
            ax.set_xscale('function', functions=(forward2,inverse2))

        # truth
        _s = 1.
        if (p=="γ" or p=="η"):
            if velocity_deficit:
                if "velocity_deficit" in dsm.attrs:
                    _s = dsm.attrs["velocity_deficit"]
                else:
                    _s = amplitude_decifit(
                        dsm.attrs["dx"], 
                        dsm.attrs["lambda_x"], 
                        dsm.attrs["nu_space"],
                    )
            if alpha_normalize:
                l = p+"/α"
                _s = _s/da["α"]
                
        truth = da["true_parameters"]*_s+da[x]*0
        if type=="boxplot":
            h_truth = ax.scatter(
                da[x].values, truth , 
                c=colors["truth"], edgecolors="k", 
                s=80, marker="*", label="truth", zorder=10,
            )
        elif type=="shading":
            h_truth = ax.plot(
                da[x].values, truth, 
                color=colors["truth"], lw=2, 
                label="truth", zorder=10,
            )

        ax.set_title(l)
        ax.set_xlabel(x_label)
        if not x_ticks_free:
            ax.set_xticks(_x)
        if ax==axes["(a)"]:
            #ax.legend()
            if isinstance(h_truth, list):
                h_truth = h_truth[0]
            if dsr is not None:
                _handles = [b0, b1, h_truth]
                _legend_labels = ['moorings', 'drifters', 'truth']
            else:
                _handles = [b0, h_truth]
                _legend_labels = [label, 'truth']
            ax.legend(_handles, _legend_labels, loc=legend_loc)

        if bounds is not None:
            # should adjust for gamma (alpha_normalize) - but works magically for now
            ax.set_ylim(bounds[p])
        add_parameter_bounds(ds.sel(parameter=p), ax)
        ax.grid()
    
    return fig, axes

# log2 x scaling
# https://stackoverflow.com/questions/65319997/how-to-set-the-tick-scale-as-the-power-of-2-in-matplotlib
def forward2(x):
    return np.log2(x)
def inverse2(x):
    return 2**x

def _boxplot(ax, da, positions, quantiles=None, **kwargs):
    bplot = ax.boxplot(
        da, 
        positions=positions, 
        **kwargs,
    )
    return bplot["boxes"][0]

def _shadeplot(ax, da, positions, quantiles=(1/4, 3/4), **kwargs):
    q0, q1 = quantiles
    qm = 1/2
    daq = da.quantile([q0, qm, q1], "ensemble")
    _kwargs = dict(**kwargs)
    _kwargs["alpha"] = 1
    ax.plot(positions, daq.sel(quantile=qm), **_kwargs)
    h = ax.fill_between(positions, daq.sel(quantile=q0), daq.sel(quantile=q1), **kwargs)
    return h
    
from matplotlib.patches import Rectangle

def add_parameter_bounds(ds, ax):
    """ add bounds as grey patches """
    lower = float(ds.lower.min())
    upper = float(ds.upper.max())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rect = Rectangle((x0,lower), x1-x0, lower-100, facecolor="0.5", zorder=0, alpha=0.5)
    ax.add_patch(rect)
    rect = Rectangle((x0,upper), x1-x0, upper+100, facecolor="0.5", zorder=0, alpha=0.5)
    ax.add_patch(rect)

def print_MAP_truth_difference(ds, dim):
    """ print MAP - truth difference along with normalized difference"""
    da = ds.MAP.median("ensemble")
    print("% "+dim+" | " + " | ".join(ds.parameter.values))
    for d in ds[dim]:
        _da = da.sel({dim: d})
        print(f"% {d.values} | "
              + " | ".join([f"{float(_da.sel(parameter=p)):.2f} / {float(_da.true_parameters.sel(parameter=p)):.2f}" for p in _da.parameter])
        )
        print(f"% normalized diff | "
              + " | ".join([f"{float((_da.sel(parameter=p)-_da.true_parameters.sel(parameter=p))/_da.sel(parameter=p))*100:.1f}"  for p in _da.parameter])
        )

def print_quantile_width(ds, dim, quantiles=(1/4, 3/4)):
    """ print quantile width along a dimension """
    
    daq = ds.MAP.quantile(quantiles, "ensemble")
    nwidth = (daq.isel(quantile=1) - daq.isel(quantile=0))/ds.true_parameters
    
    print("% "+dim+" | " + " | ".join(ds.parameter.values))
    for d in ds[dim]:
        _w = nwidth.sel(**{dim: d})
        print(f"% {d.values} | "+ " | ".join([f"{__w:.2f}" for __w in _w.values]))

def label_and_print(fig, axs, fig_name):
    """ add labels on figures and print into files """


    if axs is not None:
        for label, ax in axs.items():
            # label physical distance in and down:
            trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
            ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                    fontsize='medium', verticalalignment='top', fontfamily='serif',
                    bbox=dict(facecolor='1.0', edgecolor='none', pad=3.0))

    fig_dir = os.path.join(os.getcwd(), "figs")

    for fmt in ["eps", "png"]:
        _fig_name = os.path.join(fig_dir, fig_name+"."+fmt)
        fig.savefig(_fig_name)
        print(f"scp dunree:{_fig_name} .")
