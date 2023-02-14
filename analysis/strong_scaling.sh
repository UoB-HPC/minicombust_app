#!/bin/bash


if [ $# -ne 5 ];
then 
        echo "Wrong number of parameters. Usage: ./particle_weak_scaling.sh LOW_PROCS HIGH_PROCS MAX_PPN PARTICLES CELL_MODIFIER"
        exit 1
fi



HEADER="cores       real_time  total_time     performance   mem_bandwidth"

echo $HEADER > results/tx2-strong-$5cells_modifier-$4particles-interpolate_nodal_data.log
echo $HEADER > results/tx2-strong-$5cells_modifier-$4particles-particle_interpolation_data.log
echo $HEADER > results/tx2-strong-$5cells_modifier-$4particles-solve_spray_equations.log
echo $HEADER > results/tx2-strong-$5cells_modifier-$4particles-update_particle_positions.log
echo $HEADER > results/tx2-strong-$5cells_modifier-$4particles-emitted_particles.log
echo $HEADER > results/tx2-strong-$5cells_modifier-$4particles-updated_flow_field.log
echo $HEADER > results/tx2-strong-$5cells_modifier-$4particles-minicombust.log

for (( PROCS=$1; PROCS <= $2; PROCS *= 2 ))
do
    TEMPLATE=jobs/templates/isambard_tx2_strong
    FNAME=jobs/strong-$4particles-$5cells_modifier-${PROCS}procs.job
    cp $TEMPLATE $FNAME

    NODES=$(( PROCS / $3 ))
    PPN=$3
    if [ $PROCS -lt $3 ];
    then
        NODES=1
        PPN=$PROCS
    fi
    
    sed -i "s/MC_NODES/$NODES/g" $FNAME
    sed -i "s/MC_PROCS/$PROCS/g" $FNAME
    sed -i "s/MC_PPN/$PPN/g"     $FNAME
    sed -i "s/MC_MAX_PPN/$3/g"   $FNAME
    sed -i "s/MC_PARTICLES/$4/g" $FNAME
    sed -i "s/MC_CELLS/$5/g"     $FNAME
    
    echo "Submitting job with $PROCS processes..."
    qsub jobs/strong-$4particles-$5cells_modifier-${PROCS}procs.job
done

echo ""
echo ""
echo "Once all jobs are finished, run:"
echo "  python analysis/strong_graphs.py $4 $5"
