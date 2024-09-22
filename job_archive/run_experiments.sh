#!/bin/bash

export LD_LIBRARY_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/AMGX_vector_upload/install/lib/:$LD_LIBRARY_PATH

kill $(pidof nsys)
kill $(pidof gpu_minicombust)

# for x in {8,}
# for x in {2,1}
# for x in {8,4,2,1}
# for x in {100,126,159,200,252,318,401,505}
for x in {1,}
#for x in {8,}
do
	echo "RUNNING EXPERIMENT with $x GPUS"
	export MINICOMBUST_GPUS=$x
	export MINICOMBUST_NODES=1

	export MINICOMBUST_RANKS=$(( MINICOMBUST_GPUS*14 ))
	export MINICOMBUST_PRANKS=$(( MINICOMBUST_RANKS - MINICOMBUST_GPUS ))
	# export MINICOMBUST_CELLS=484
	export MINICOMBUST_CELLS=241
	# export MINICOMBUST_CELLS=304
	export MINICOMBUST_PARTICLES=280000
	#export MINICOMBUST_CELLS=484
	#export MINICOMBUST_PARTICLES=1040001
	export MINICOMBUST_ITERS=10

	echo ""	
  	echo "MINICOMBUST_GPUS      $MINICOMBUST_GPUS "
	echo "MINICOMBUST_RANKS     $MINICOMBUST_RANKS "
	echo "MINICOMBUST_PRANKS    $MINICOMBUST_PRANKS "
	echo "MINICOMBUST_CELLS     $MINICOMBUST_CELLS "
	echo "MINICOMBUST_PARTICLES $MINICOMBUST_PARTICLES "
	echo "MINICOMBUST_ITERS     $MINICOMBUST_ITERS "	
	echo ""	

	cmd="mpirun -np $MINICOMBUST_RANKS ./wrapper.sh ./bin/gpu_minicombust $MINICOMBUST_PRANKS $MINICOMBUST_PARTICLES $MINICOMBUST_CELLS -1 $MINICOMBUST_ITERS"
	filename="STRONG-RANKS$MINICOMBUST_RANKS-GPUS$MINICOMBUST_GPUS-CELLS$MINICOMBUST_CELLS-PARTICLES-$MINICOMBUST_PARTICLES-ITERS$MINICOMBUST_ITERS.log"
	echo "$cmd"
	# echo "$filename"
	OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 $cmd | tee $filename
done

