#!/usr/bin/env python
# coding: utf-8

# # 3D (x-y-t) field inference: ensemble experiments
# 
# We start from a flow field generated with `drifter_3D.ipynb` and run multiple inferences with different draws of  mooring positions & drifters release positions as well as noise.

import os
import logging
from time import sleep, time

import xarray as xr
from pandas.api.types import is_number

import nwastats as st
day = 86400

#data_dir = "data/"
#data_dir = "/home1/scratch/aponte/"
data_dir = "/home/datawork-lops-osi/aponte/nwa/drifter_stats"

overwrite=True
overwrite=False
mooring, drifter = True, True
#mooring, drifter = False, True
#mooring, drifter = True, False

# flow case
U = "0.1"
#case = "3D_matern32_iso_matern12_pp_r0.0_u"+U
case = "3D_matern52_iso_matern12_pp_r0.0_u"+U
run_dir = os.path.join(data_dir, case)

uv = True # True if u and v are observed
parameter_eta_formulation = False # eta vs gamma formulation
noise = 0.01 # observation noise added to u/v
no_time = False # activates time inference
no_space = False # activates time inference
traj_decorrelation = False  # artificially decorrelate different moorings/drifters
enable_nu = False

# number of points used for inference#
Nxy, Nt = 8, 50

# size of the experiment ensemble
#Ne = 10 # dev
Ne = 100

# separation parameter
dx = None

# number of inference samples:
n_mcmc = 20_000 # default value

osuffix = None
#nu_low, nu_up = None, None

experiment = 5
if experiment==0:
    # ensemble run
    Nxy = [1, 2, 4, 8, 16]
    # dev
    #Nxy = [1, 2, 4, 8]
    #enable_nu = True
    #n_mcmc = 500
elif experiment==1:
    # decorralated platform experiment
    traj_decorrelation = True
    Nxy = [1, 2, 4, 8, 16]
elif experiment==2:
    # dx sensitivity run
    Nxy, Nt = 2, 50
    #dx = [5, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200, 300]  # 5 does not go through with 10% tolerance on distance
    dx = [10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200, 300]
elif experiment==3:
    # spiral deployments
    Nxy = [1, 2, 4, 8, 16]
    dx = ("spiral", 50, 300)
elif experiment==4:
    # nu_estimate experiment
    enable_nu = True # enable estimation of spectral slopes
    n_mcmc = 10_000
    # 13h30*2
    osuffix = "_lprior"
    #nu_low = [.1, .1]
    #nu_up = [5, 5]
elif experiment==5:
    # nu_estimate experiment
    enable_nu = True # enable estimation of spectral slopes
    n_mcmc = 10_000
    #n_mcmc = 100 # dev
    # 13h30*2
    #osuffix = "_lprior"
    dx = ("spiral", 50, 300)
    #nu_low = [.1, .1]
    #nu_up = [5, 5]


if no_space:
    # makes little sense otherwise
    traj_decorrelation = True
assert not no_time, "need to implement decorrelation across time"


# inference parameters
# dsteps = {16: 1/20, 8: 1/20, 4: 1/5, 2: 1/5, 1: 1/2} #v0
dsteps = {16: .05, 8: .1, 4: .15, 2: .2, 1: .3} # v1
if enable_nu:
    dsteps = {k: v*.5 for k, v in dsteps.items()}


# dask parameters

suffix="job0"
#suffix="dumm" # tmp

#dask_job_name = None
dask_job_name = "dask"+suffix

dask_num_jobs, dask_jobkw = 5, dict(processes=20, cores=20)
#dask_num_jobs, dask_jobkw = 10, dict(processes=10, cores=10)

dask_jobkw["walltime"] = "48:00:00"
dask_jobkw["job_name"] = dask_job_name


# ---------------------------------- main ------------------------------------

def run():

    logging.info(f"starting run: case={case} / experiment={experiment}")

    # ### prepare inference & common utils

    dsf, covfunc, covparams, labels = st.prepare_inference(
        run_dir,
        uv, no_time, no_space,
        parameter_eta_formulation, traj_decorrelation,
        enable_nu,
    )
    logging.info("dsf attributes: "+" ;".join([f"{k}/{v}" for k, v in dsf.attrs.items()]))

    # gather common kwargs
    kwargs = dict(
        dx=dx, no_time=no_time, no_space=no_space, 
    )
    if enable_nu:
        # expected order hardcoded below ... danger
        _covparams = dict(zip(labels[1:], covparams))
        #if not nu_low:
        #    nu_low = [_covparams["νs"]-0.5, _covparams["νt"]-0.49]
        #    nu_up = [_covparams["νs"]+0.5, _covparams["νt"]+0.5]
        #kwargs["lowers"] = [None]*3 + nu_low + [None]
        #kwargs["uppers"] = [None]*3 + nu_up + [None]
        kwargs["lowers"] = [None]*3 + [1, .1] + [None]
        #kwargs["uppers"] = [None]*3 + [3.5, 3.5] + [None]
        kwargs["uppers"] = [None]*3 + [5, 5] + [None]
        #logging.info("nu_low = "+str(nu_low[0])+","+str(nu_low[1]))
        logging.info("lowers = "+" , ".join([f"{v}" for v in kwargs["lowers"]]))
        logging.info("uppers = "+" , ".join([f"{v}" for v in kwargs["uppers"]]))
    
    kwargs["n_mcmc"] = n_mcmc
    #kwargs["n_mcmc"] = 1_000 # dev
    #kwargs["n_mcmc"] = 500 # dev
    #print("n_mcmc = "+str(kwargs["n_mcmc"])) # check

    # ---
    # ## mooring inference
    logging.info("starting mooring inference ...")

    def run_mooring(Nxy, dx):
        
        if dx is not None:
            if is_number(dx):
                logging.info(f" Nxy= {Nxy}, dx={dx} - start")
            elif isinstance(dx, tuple):
                logging.info(f" Nxy= {Nxy}, dx={dx[0]} - start")
        else:
            logging.info(f" Nxy= {Nxy} - start")
        
        # build output file name
        nc = os.path.join(run_dir, f"moorings_ensemble_Nxy{Nxy}.nc")
        if dx is not None:
            if is_number(dx):
                nc = nc.replace(".nc", f"_dx{dx:0.0f}.nc")
            elif isinstance(dx, tuple):
                nc = nc.replace(".nc", f"_{dx[0]}.nc")
        if traj_decorrelation:
            nc = nc.replace(".nc", f"_trajd.nc")
        if enable_nu:
            nc = nc.replace(".nc", f"_nu.nc")
        if osuffix is not None:
            nc = nc.replace(".nc", osuffix+".nc")

        # adjust step size dynamically
        steps = [dsteps[Nxy]]*4
        if enable_nu:
            steps = [dsteps[Nxy]]*6

        dkwargs = dict(**kwargs)
        dkwargs.update(dx=dx, steps=steps)
        
        # do not overwrite
        if not os.path.isfile(nc) or overwrite:
            ds = st.run_mooring_ensembles(
                Ne, dsf, covparams, covfunc, labels, (Nt, Nxy), noise,
                **dkwargs,
            )
            ds.to_netcdf(nc, mode="w")
            logging.info(f" output file: {nc}")
        else:
            logging.info(" ... skipping")
            ds = xr.open_dataset(nc)
            ds["parameter"] = ds.parameter.astype(str)
        logging.info(f" Nxy= {Nxy}, ... - end")
        return ds

    if mooring:
        if isinstance(dx, list):
            Dm = []
            for d in dx:
                Dm.append(run_mooring(Nxy, d))
            ds = Dm[0]
        elif isinstance(Nxy, list):
            Dm = []
            for n in Nxy:
                Dm.append(run_mooring(n, dx))
            ds = Dm[0]
        else:
            ds = run_mooring(Nxy, dx)
    else:
        logging.info(" ! skipping !")

    logging.info(" ... ending mooring inference")


    # ---
    # ## drifter inference
    logging.info("starting drifter inference ...")

    def run_drifter(Nxy, dx):
        
        if dx is not None:
            if is_number(dx):
                logging.info(f" Nxy= {Nxy}, dx={dx} - start")
            elif isinstance(dx, tuple):
                logging.info(f" Nxy= {Nxy}, dx={dx[0]} - start")
        else:
            logging.info(f" Nxy= {Nxy} - start")
        
        # build output file name    
        nc = os.path.join(run_dir, f"drifters_ensemble_Nxy{Nxy}.nc")
        if dx is not None:
            if is_number(dx):
                nc = nc.replace(".nc", f"_dx{dx:0.0f}.nc")
            elif isinstance(dx, tuple):
                nc = nc.replace(".nc", f"_{dx[0]}.nc")
        if traj_decorrelation:
            nc = nc.replace(".nc", f"_trajd.nc")
        if enable_nu:
            nc = nc.replace(".nc", f"_nu.nc")
        if osuffix is not None:
            nc = nc.replace(".nc", osuffix+".nc")

        # adjust step size dynamically
        steps = [dsteps[Nxy]]*4
        if enable_nu:
            steps = [dsteps[Nxy]]*6

        dkwargs = dict(**kwargs)
        dkwargs.update(dx=dx, steps=steps)
        
        if not os.path.isfile(nc) or overwrite:
            ds = st.run_drifter_ensembles(
                run_dir, Ne, covparams, covfunc, labels, (Nt, Nxy), noise, 
                **dkwargs,
            ) 
            ds.to_netcdf(nc, mode="w")
            logging.info(f" output file: {nc}")
        else:
            logging.info(" ... skipping")
            ds = xr.open_dataset(nc)
            ds["parameter"] = ds.parameter.astype(str)
        logging.info(f" Nxy= {Nxy}, ... - end")
        return ds

    if drifter:
        if isinstance(dx, list):
            Dr = []
            for d in dx:
                Dr.append(run_drifter(Nxy, d))
            ds = Dr[0]
        elif isinstance(Nxy, list):
            Dr = []
            for n in Nxy:
                Dr.append(run_drifter(n, dx))
            ds = Dr[0]
        else:
            ds = run_drifter(Nxy, dx)
    else:
        logging.info(" ! skipping !")

    logging.info(" ... ending drifter inference")


# ---------------------------------- dask ------------------------------------

def spin_up_cluster(
    ctype,
    jobs=None,
    processes=None,
    fraction=0.9,
    timeout=20,
    **kwargs,
):
    """Spin up a dask cluster ... or not
    Waits for workers to be up for distributed ones

    Paramaters
    ----------
    ctype: None, str
        Type of cluster: None=no cluster, "local", "distributed"
    jobs: int, optional
        Number of PBS jobs
    processes: int, optional
        Number of processes per job (overrides default in .config/dask/jobqueue.yml)
    timeout: int
        Timeout in minutes is cluster does not spins up.
        Default is 20 minutes
    fraction: float, optional
        Waits for fraction of workers to be up

    """

    if ctype is None:
        return
    elif ctype == "local":
        from dask.distributed import Client, LocalCluster

        dkwargs = dict(n_workers=14, threads_per_worker=1)
        dkwargs.update(**kwargs)
        cluster = LocalCluster(**dkwargs)  # these may not be hardcoded
        client = Client(cluster)
    elif ctype == "distributed":
        from dask_jobqueue import PBSCluster
        from dask.distributed import Client

        assert jobs, "you need to specify a number of dask-queue jobs"
        cluster = PBSCluster(processes=processes, **kwargs)
        cluster.scale(jobs=jobs)
        client = Client(cluster)

        if not processes:
            processes = cluster.worker_spec[0]["options"]["processes"]

        flag = True
        start = time()
        while flag:
            wk = client.scheduler_info()["workers"]
            logging.info("Number of workers up = {}".format(len(wk)))
            sleep(5)
            if len(wk) >= processes * jobs * fraction:
                flag = False
                logging.info("Cluster is up, proceeding with computations")
            now = time()
            if (now - start) / 60 > timeout:
                flag = False
                logging.info("Timeout: cluster did not spin up, closing")
                cluster.close()
                client.close()
                cluster, client = None, None

    return cluster, client


def dashboard_ssh_forward(client):
    """returns the command to execute on a local computer in order to
    have access to dashboard at the following address in a browser:
    http://localhost:8787/status
    """
    env = os.environ
    port = client.scheduler_info()["services"]["dashboard"]
    return f'ssh -N -L {port}:{env["HOSTNAME"]}:8787 {env["USER"]}@datarmor1-10g', port


def close_dask(cluster, client):
    logging.info("starts closing dask cluster ...")
    try:
        client.close()
        logging.info("client closed ...")
        # manually kill pbs jobs
        manual_kill_jobs()
        logging.info("manually killed jobs ...")
        # cluster.close()
        # logging.info("cluster closed ...")
    except:
        logging.exception("cluster.close failed ...")
        # manually kill pbs jobs
        manual_kill_jobs()

    logging.info("... done")


def manual_kill_jobs():
    """manually kill dask pbs jobs"""

    import subprocess, getpass

    # 
    jobname = dask_job_name
    if jobname is None:
        jobname = "dask-worke"
    jobname = jobname[:10]
    
    #
    username = getpass.getuser()
    #
    bashCommand = "qstat"
    try:
        output = subprocess.check_output(
            bashCommand, shell=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    #
    for line in output.splitlines():
        lined = line.decode("UTF-8")
        if username in lined and jobname in lined:
            pid = lined.split(".")[0]
            bashCommand = "qdel " + str(pid)
            logging.info(" " + bashCommand)
            try:
                boutput = subprocess.check_output(bashCommand, shell=True)
            except subprocess.CalledProcessError as e:
                # logging.info(e.output.decode())
                pass

# ---------------------------------- main ------------------------------------

if __name__=="__main__":


    # to std output
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # to file
    logging.basicConfig(
        filename="distributed_"+suffix+".log",
        level=logging.INFO,
        # level=logging.DEBUG,
        format='%(asctime)s %(message)s',
    )
    # level order is: DEBUG, INFO, WARNING, ERROR
    # encoding='utf-8', # available only in latests python versions


    # spin up cluster
    logging.info("start spinning up dask cluster, jobs={}".format(dask_num_jobs))
    cluster, client = spin_up_cluster(
        "distributed",
        jobs=dask_num_jobs,
        **dask_jobkw,
    )
    ssh_command, dashboard_port = dashboard_ssh_forward(client)
    logging.info("dashboard via ssh: " + ssh_command)
    logging.info(f"open browser at address of the type: http://localhost:{dashboard_port}")

    # main run
    run()

    # close dask
    close_dask(cluster, client)
