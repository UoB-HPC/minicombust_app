#!/bin/bash

export LD_LIBRARY_PATH=/home/scratch.hwaugh_gpu/repos/AMGX/install/lib/:$LD_LIBRARY_PATH
export MELLANOX_VISIBLE_DEVICES=void
enroot create --name nvhpc /home/scratch.hwaugh_gpu/containers/nvhpc24.3.sqsh
enroot start --mount /home/scratch.hwaugh_gpu/:/home/scratch.hwaugh_gpu/ nvhpc
