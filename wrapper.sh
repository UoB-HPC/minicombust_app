#!/bin/bash

export OMPI_MCA_pml=ucx                                                                                                           
export UCX_MEMTYPE_CACHE=n      
export UCX_MAX_RNDV_RAILS=1
export UCX_CUDA_COPY_DMABUF=no
# export UCX_IB_PREFER_NEAREST_DEVICE=y 
export OMPI_MCA_coll=^hcoll



export CUDA_VISIBLE_DEVICES=
if [ $OMPI_COMM_WORLD_RANK -gt $(( OMPI_COMM_WORLD_SIZE-8-1 )) ]; then
  export CUDA_VISIBLE_DEVICES=$(( OMPI_COMM_WORLD_RANK - (OMPI_COMM_WORLD_SIZE-8) ))
  echo "$OMPI_COMM_WORLD_RANK $CUDA_VISIBLE_DEVICES"
fi


# Use $PMI_RANK for MPICH and $SLURM_PROCID with srun.
if [ $OMPI_COMM_WORLD_RANK -gt $(( OMPI_COMM_WORLD_SIZE-2 ))  ]; then
	NSYS_CONFIG_DIRECTIVES='AgentLaunchTimeoutSec=120' ./nsight-systems-linux-public-DVS/bin/nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 --cpuctxsw=none --trace=osrt,openacc,cuda,nvtx,mpi --force-overwrite=true -o 3iter_gpu_$x "$@"
else
  "$@"
fi
