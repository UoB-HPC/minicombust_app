#!/bin/bash

## Job configuration
NODES=1
NGPUS=8
MPI_PER_NODE=112
WALLTIME=0:20:00

## Paths and Containers
CONTAINER=/lustre/fsw/coreai_devtech_all/hwaugh/containers/devtech.sqsh
MOUNT=/lustre/fsw/coreai_devtech_all/hwaugh/
WDIR=/lustre/fsw/coreai_devtech_all/hwaugh/repos/minicombust_app
AMGX_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/AMGX_vector_upload/install/
MPI_PATH=/usr/local/openmpi/
CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/cuda/

## MiniCombust Parameters
CELLS=304
PARTICLES=280000
ITERS=100

## System paramaters
JOB_TEMPLATE=job_templates/eos.job

## Profiling
NSYS_CMD="./nsight-systems-linux-public-DVS/bin/nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 --sample=none --cpuctxsw=none --trace=cuda,nvtx,mpi --force-overwrite=true"
NCU_CMD="ncu -f --set full  --kernel-name kernel_get_phi_gradient --launch-count 1 --launch-skip 6"
PROF_RANKS=0

## Other defaults
USE_ENROOT=0
INTERACTIVE=0
INTERACTIVE_RUN=0
NO_CONTAINER=0
