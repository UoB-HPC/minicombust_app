#!/bin/bash

export LD_LIBRARY_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/AMGX/install/lib/:$LD_LIBRARY_PATH

for x in {8,}
do
	cmd="mpirun -np 112 ./wrapper.sh ./bin/gpu_minicombust $(( 112-x )) 80000 200 -1 3"
	cmd="mpirun -np 9 ./wrapper.sh ./bin/gpu_minicombust 1 80000 200 -1 3"
	filename="${x}gpu_strong_112proc_80000part_200cell_100iter.log"
	echo "$cmd $filename "
	OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 $cmd | tee $filename
done


