#!/bin/bash

# OMPI_MCA_pml=ucx                                                                                                           
# UCX_MEMTYPE_CACHE=n      
# UCX_MAX_RNDV_RAILS=1
# UCX_IB_PREFER_NEAREST_DEVICE=y 
# OMPI_MCA_coll=^hcoll

# UCX_TLS=cuda_copy,cuda_ipc,gdr_copy
# UCX_TLS=self,rc,shm,cuda,tcp,cuda_copy,cuda_ipc,gdr_copy

export CUDA_VISIBLE_DEVICES=
if [ $OMPI_COMM_WORLD_RANK -gt $(( MINICOMBUST_PRANKS-1 )) ]; then
  export CUDA_VISIBLE_DEVICES=$(( OMPI_COMM_WORLD_RANK - MINICOMBUST_PRANKS ))
  # UCX_CUDA_COPY_DMABUF=no
  echo "RANK $OMPI_COMM_WORLD_RANK VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  echo ""
  echo "OMPI_MCA_pml                 $OMPI_MCA_pml "
  echo "UCX_TLS                      $UCX_TLS "
  echo "UCX_MEMTYPE_CACHE            $UCX_MEMTYPE_CACHE "
  echo "UCX_MAX_RNDV_RAILS           $UCX_MAX_RNDV_RAILS "
  echo "UCX_IB_PREFER_NEAREST_DEVICE $UCX_IB_PREFER_NEAREST_DEVICE "
  echo "OMPI_MCA_coll                $OMPI_MCA_coll "
  echo "UCX_CUDA_COPY_DMABUF         $UCX_CUDA_COPY_DMABUF "
  echo ""

fi

export MINICOMBUST_FRANKS=$(( MINICOMBUST_NODES * MINICOMBUST_GPUS ))
export MINICOMBUST_RANK_ID=$OMPI_COMM_WORLD_RANK

# Use $PMI_RANK for MPICH and $SLURM_PROCID with srun.
if [ $OMPI_COMM_WORLD_RANK = $MINICOMBUST_PRANKS  ] || [ $OMPI_COMM_WORLD_RANK = 0 ]; then
	outfile="RANK$OMPI_COMM_WORLD_RANK-RANKS$MINICOMBUST_RANKS-GPUS$MINICOMBUST_GPUS-CELLS$MINICOMBUST_CELLS-PARTICLES-$MINICOMBUST_PARTICLES-ITERS$MINICOMBUST_ITERS"
  prof_cmd="valgrind --leak-check=yes --show-reachable=yes --track-origins=yes --log-file=$OMPI_COMM_WORLD_RANK.log " 
  prof_cmd="compute-sanitizer " 
  prof_cmd="" 
  prof_cmd="ncu -f --set full  --kernel-name kernel_interpolate_phi_to_nodes --launch-count 5 --launch-skip 1 -o $outfile " 
  prof_cmd="./nsight-systems-linux-public-DVS/bin/nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 --sample=none --cpuctxsw=none --trace=cuda,nvtx,mpi --force-overwrite=true -o $outfile " 
	echo "$OMPI_COMM_WORLD_RANK rank prof_cmd:  $prof_cmd"
	echo "cmd:  $prof_cmd"
	echo ""
  NSYS_CONFIG_DIRECTIVES='AgentLaunchTimeoutSec=120' $prof_cmd "$@"
else
  prof_cmd="valgrind --leak-check=yes --show-reachable=yes --track-origins=yes --log-file=$OMPI_COMM_WORLD_RANK.log "  
  prof_cmd="" 
  $prof_cmd "$@"
fi
