#!/bin/bash

source defaults.sh

## Output
BUILD=0
NSYS=0
NCU=0
RESULTS_DIR=$PWD/results/$(date +%F)/
GLOBAL_PROF_CMD=

while [ : ]; do
  case "$1" in
    --build)
        export BUILD=1
        shift
        ;;
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
        NSYS=1
        GLOBAL_PROF_CMD="$NSYS_CMD"
        shift
        ;;
    --ncu)
        NCU=1
        GLOBAL_PROF_CMD="$NCU_CMD"
        shift
        ;;
    --prof_ranks)
        PROF_RANKS="$2"
        shift 2
        ;;
    --interactive)
        export INTERACTIVE=1
        export INTERACTIVE_RUN=1
        shift
        ;;
    --jump_into_container)
        export INTERACTIVE=1
        export INTERACTIVE_RUN=0
        export USE_ENROOT=1
        shift
        ;;
    --no_container)
        export NO_CONTAINER=1
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
    echo "  cuda_path:             $CUDA_PATH"               | tee -a $OUTFILE.log      
    echo "  mpi_path:              $MPI_PATH"                | tee -a $OUTFILE.log      
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
OUTER_CMD="source unset.sh; srun --overlap -N${NODES} --ntasks-per-node=${MPI_PER_NODE} --mem-bind=none --cpu-bind=none --mpi=pmix --container-image=${CONTAINER} --distribution=cyclic:cyclic --container-mounts=${MOUNT}:${MOUNT} bash -c"

if [ $USE_ENROOT -eq 1 ]
then
    CREATE_CMD="enroot create --name minicombust -- $CONTAINER || true > /dev/null 2>&1 "
    OUTER_CMD="enroot start --mount $MOUNT:$MOUNT minicombust bash -c"
    INNER_CMD="mpirun -bind-to none -np ${RANKS} ${INNER_CMD}"
fi

if [ $BUILD -eq 1 ]
then
    INNER_CMD="source build.sh"
    OUTER_CMD="source unset.sh; srun --overlap -N1 --ntasks-per-node=1 --mem-bind=none --cpu-bind=none --mpi=pmix --container-image=${CONTAINER} --distribution=cyclic:cyclic --container-mounts=${MOUNT}:${MOUNT} bash -c"
fi

if [ $NO_CONTAINER -eq 1 ]
then
    OUTER_CMD="bash -c"
fi

OUTFILE="$RESULTS_DIR/NODES${NODES}-RANKS$RANKS-GPUS$NGPUS-CELLS$CELLS-PARTICLES$PARTICLES-ITERS$ITERS"

if [ $NSYS -eq 1 ] || [ $NCU -eq 1 ];
then
    GLOBAL_PROF_CMD="$GLOBAL_PROF_CMD -o $OUTFILE"
fi

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
sed -i "s@#MPI_PATH#@$MPI_PATH@g"                  $OUTFILE.job
sed -i "s@#CUDA_PATH#@$CUDA_PATH@g"                $OUTFILE.job
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
