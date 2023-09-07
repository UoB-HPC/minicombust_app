#!/bin/bash
#PBS -q arm
#PBS -l select=MC_NODES:ncpus=MC_MAX_PPN
#PBS -l walltime=00:04:00
#PBS -o results/isambard-MC_PROCSprocs-MC_PARTICLESparticles-MC_CELLScells_modifier.log
#PBS -N strong-MC_PROCS
#PBS -j oe

cd /home/br-hwaugh/repos/minicombust_app_tx2_gcc12/

source env.sh

module unload cray-python/3.8.5.1
module load valgrind4hpc/2.10.2
TMPDIR=$HOME/scratch

# binding=`./get_binding.sh MC_PROCS`
binding="numa_node"
echo "Running: aprun -n MC_PROCS -N MC_PPN --cc $binding ./bin/minicombust $(( 2 * MC_PROCS / 4 )) MC_PARTICLES MC_CELLS -1"
# valgrind4hpc --tool=memcheck -o outMC_PROCS_MC_PARTICLES_MC_CELLS.log --valgrind-args="--leak-check=full" --num-ranks MC_PROCS ./bin/minicombust -- $(( 3 * MC_PROCS / 4 )) MC_PARTICLES MC_CELLS -1
aprun -n MC_PROCS -N MC_PPN --cc $binding ./bin/minicombust $(( 1 * MC_PROCS / 16 )) MC_PARTICLES MC_CELLS -1
module load cray-python/3.8.5.1
echo "Running: python analysis/get_roofline_cmd.py TX2 MC_NODESnodes MC_PPN | tee /tmp/roofline.log"
python analysis/get_roofline_cmd.py TX2 MC_NODESnodes MC_PROCS | tee /tmp/roofline_MC_PROCS.log

cat /tmp/roofline_MC_PROCS.log | grep "    interpolate_nodal_data"      >> results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-interpolate_nodal_data.log
cat /tmp/roofline_MC_PROCS.log | grep "    particle_interpolation_data" >> results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-particle_interpolation_data.log
cat /tmp/roofline_MC_PROCS.log | grep "    solve_spray_equations"       >> results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-solve_spray_equations.log
cat /tmp/roofline_MC_PROCS.log | grep "    update_particle_positions"   >> results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-update_particle_positions.log
cat /tmp/roofline_MC_PROCS.log | grep "    emitted_particles"           >> results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-emitted_particles.log
cat /tmp/roofline_MC_PROCS.log | grep "    updated_flow_field"          >> results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-updated_flow_field.log
cat /tmp/roofline_MC_PROCS.log | grep "    minicombust"                 >> results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-minicombust.log


sed -i "s/ *interpolate_nodal_data/MC_PROCS/g"       results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-interpolate_nodal_data.log
sed -i "s/ *particle_interpolation_data/MC_PROCS/g"  results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-particle_interpolation_data.log
sed -i "s/ *solve_spray_equations/MC_PROCS/g"        results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-solve_spray_equations.log
sed -i "s/ *update_particle_positions/MC_PROCS/g"    results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-update_particle_positions.log
sed -i "s/ *emitted_particles/MC_PROCS/g"            results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-emitted_particles.log
sed -i "s/ *updated_flow_field/MC_PROCS/g"           results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-updated_flow_field.log
sed -i "s/ *minicombust/MC_PROCS/g"                  results/tx2-strong-MC_CELLScells_modifier-MC_PARTICLESparticles-minicombust.log