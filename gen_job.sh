#!/bin/bash

NODES=1
NGPUS=8
MPI_PER_NODE=112
CELLS=304
PARTICLES=280000
ITERS=100
RESULTS_DIR=$PWD/results/$(date +%F)/
CONTAINER=/lustre/fsw/coreai_devtech_all/hwaugh/containers/devtech.sqsh
JOB_TEMPLATE=job_templates/eos.job
MOUNT=/lustre/fsw/coreai_devtech_all/hwaugh/
WDIR=/lustre/fsw/coreai_devtech_all/hwaugh/repos/minicombust_app
WALLTIME=0:20:00
USE_ENROOT=0
INTERACTIVE=0
INTERACTIVE_RUN=0
AMGX_PATH=/lustre/fsw/coreai_devtech_all/hwaugh/repos/AMGX_vector_upload/install/lib/
GLOBAL_PROF_CMD=
PROF_RANKS=0

while [ : ]; do
  case "$1" in
    --mpi_procs)
        export MPI_PER_NODE=$2
        shift 2
        ;;
    --nodes)
        export NODES=$2
        shift 2
        ;;
    --gpus)
        export NGPUS=$2
        shift 2
        ;;
    --container)
        export CONTAINER=$2
        shift 2
        ;;
    --results_name)
        export RESULTS_DIR=$RESULTS_DIR/$2
        shift 2
        ;;
    --job_template)
        export JOB_TEMPLATE=$2
        shift 2
        ;;
    --walltime)
        export WALLTIME=$2
        shift 2
        ;;
    --enroot)
        USE_ENROOT=1
        shift
        ;;
    --nsys)
        GLOBAL_PROF_CMD="./nsight-systems-linux-public-DVS/bin/nsys profile -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 --sample=none --cpuctxsw=none --trace=cuda,nvtx,mpi --force-overwrite=true -o"
        shift
        ;;
    --ncu)
        GLOBAL_PROF_CMD="ncu -f --set full  --kernel-name kernel_get_phi_gradient --launch-count 1 --launch-skip 6 -o"
        shift
        ;;
    --prof_ranks)
        PROF_RANKS="$2"
        shift 2
        ;;
    --interactive_run)
        export INTERACTIVE=1
        export INTERACTIVE_RUN=1
        USE_ENROOT=1
        shift
        ;;
    --interactive)
        export INTERACTIVE=1
        export INTERACTIVE_RUN=0
        USE_ENROOT=1
        shift
        ;;
    --) 
        shift; 
        break 
        ;;
  esac
done

RANKS=$((  MPI_PER_NODE*NODES ))
PRANKS=$(( RANKS - NGPUS*NODES ))
FRANKS=$(( RANKS - PRANKS ))

print_config () {
    rm -f $OUTFILE.log 
    echo "MiniCombust configuration: "                       | tee -a $OUTFILE.log 
    echo "  nodes:                 $NODES"                   | tee -a $OUTFILE.log
    echo "  ngpus:                 $NGPUS"                   | tee -a $OUTFILE.log
    echo "  ranks:                 $RANKS"                   | tee -a $OUTFILE.log
    echo "  pranks:                $PRANKS"                  | tee -a $OUTFILE.log
    echo "  franks:                $FRANKS"                  | tee -a $OUTFILE.log
    echo "  mpi_procs_per_node:    $MPI_PER_NODE"            | tee -a $OUTFILE.log
    echo "  cells:                 $CELLS"                   | tee -a $OUTFILE.log
    echo "  particles:             $PARTICLES"               | tee -a $OUTFILE.log
    echo "  iters:                 $ITERS"                   | tee -a $OUTFILE.log
    echo "  container:             $CONTAINER"               | tee -a $OUTFILE.log
    echo "  results_dir:           $RESULTS_DIR"             | tee -a $OUTFILE.log              
    echo "  job_template:          $JOB_TEMPLATE"            | tee -a $OUTFILE.log       
    echo "  mount:                 $MOUNT"                   | tee -a $OUTFILE.log       
    echo "  wdir:                  $WDIR"                    | tee -a $OUTFILE.log       
    echo "  walltime:              $WALLTIME"                | tee -a $OUTFILE.log       
    echo "  amgx_path:             $AMGX_PATH"               | tee -a $OUTFILE.log      
    echo "  profile_ranks:         $PROF_RANKS"              | tee -a $OUTFILE.log      
    echo "  create_cmd:            $CREATE_CMD"              | tee -a $OUTFILE.log       
    echo "  global_prof_cmd:       $GLOBAL_PROF_CMD"         | tee -a $OUTFILE.log      
    echo "  inner_cmd:             $INNER_CMD"               | tee -a $OUTFILE.log       
    echo "  outer_cmd:             $OUTER_CMD"               | tee -a $OUTFILE.log       
    echo "" 
    echo "" 
}

mkdir -p $RESULTS_DIR

CREATE_CMD=""
INNER_CMD="./wrapper.sh ./bin/gpu_minicombust $PRANKS $PARTICLES $CELLS -1 $ITERS"
OUTER_CMD="source unset.sh; srun -N${NODES} --ntasks-per-node=${MPI_PER_NODE} --mem-bind=none --cpu-bind=none --mpi=pmix --container-image=${CONTAINER} --distribution=cyclic:cyclic --container-mounts=${MOUNT}:${MOUNT} bash -c"

if [ $USE_ENROOT -eq 1 ]
then
    CREATE_CMD="enroot create --name minicombust -- $CONTAINER || true > /dev/null 2>&1 "
    OUTER_CMD="enroot start --mount $MOUNT:$MOUNT minicombust bash -c"
    INNER_CMD="mpirun -bind-to none -np ${RANKS} ${INNER_CMD}"
fi

OUTFILE="$RESULTS_DIR/NODES${NODES}-RANKS$RANKS-GPUS$NGPUS-CELLS$CELLS-PARTICLES$PARTICLES-ITERS$ITERS"
GLOBAL_PROF_CMD="$GLOBAL_PROF_CMD $OUTFILE"

print_config

cp $JOB_TEMPLATE $OUTFILE.job

sed -i "s/#NODES#/$NODES/g"                        $OUTFILE.job
sed -i "s/#NGPUS#/$NGPUS/g"                        $OUTFILE.job
sed -i "s/#MPI_PER_NODE#/$MPI_PER_NODE/g"          $OUTFILE.job
sed -i "s@#OUTFILE#@$OUTFILE@g"                    $OUTFILE.job
sed -i "s@#CONTAINER#@$CONTAINER@g"                $OUTFILE.job
sed -i "s@CREATE_CMD@$CREATE_CMD@g"                $OUTFILE.job
sed -i "s@INNER_CMD@$INNER_CMD@g"                  $OUTFILE.job
sed -i "s@OUTER_CMD@$OUTER_CMD@g"                  $OUTFILE.job
sed -i "s@#WALLTIME#@$WALLTIME@g"                  $OUTFILE.job
sed -i "s@#RANKS#@$RANKS@g"                        $OUTFILE.job
sed -i "s@#PRANKS#@$PRANKS@g"                      $OUTFILE.job
sed -i "s@#CELLS#@$CELLS@g"                        $OUTFILE.job
sed -i "s@#PARTICLES#@$PARTICLES@g"                $OUTFILE.job
sed -i "s@#ITERS#@$ITERS@g"                        $OUTFILE.job
sed -i "s@OUTFILE@$OUTFILE@g"                      $OUTFILE.job
sed -i "s@#AMGX_PATH#@$AMGX_PATH@g"                $OUTFILE.job
sed -i "s@#PROF_CMD#@$GLOBAL_PROF_CMD@g"           $OUTFILE.job
sed -i "s@#PROF_RANKS#@$PROF_RANKS@g"              $OUTFILE.job

if [ $INTERACTIVE -eq 1 ]
then
    if [ $INTERACTIVE_RUN -eq 1 ]
    then
        chmod +x $OUTFILE.job
        bash $OUTFILE.job
    else
        echo "Running CREATE_CMD: $CREATE_CMD"
        $CREATE_CMD
        enroot start --mount $MOUNT minicombust
    fi
else
    echo "Submitting job"
    sbatch $OUTFILE.job
fi
