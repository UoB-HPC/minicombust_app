#!/bin/bash
#export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/AMGX/install/lib/:$LD_LIBRARY_PATH
export MELLANOX_VISIBLE_DEVICES=void
enroot create --name dt /lustre/fsw/coreai_devtech_all/hwaugh/containers/devtech.sqsh
enroot start --mount /lustre/fsw/coreai_devtech_all/hwaugh/:/lustre/fsw/coreai_devtech_all/hwaugh/ dt
