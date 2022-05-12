#!/bin/bash

for x in {1..200}; 
do
    FILE=out/${x}_particles.log 
    echo "Number of particles per timestep " $x;
    ./bin/minicombust $x | tee $FILE 
    python analysis/get_roofline_cmd.py CASCADE_LAKE 1core out/performance.csv | tee -a $FILE 
done


grep -r "Program Runtime" out/*_particles.log | tee out/runtimes.log
grep -r "Avg Particles" out/*_particles.log | tee out/avg_num_particle.log
grep -r "interpolate_nodal_data:" out/*_particles.log | tee out/interpolate_nodal_data.log
grep -r "particle_interpolation_data:" out/*_particles.log | tee out/particle_interpolation_data.log
grep -r "solve_spray_equations:" out/*_particles.log | tee out/solve_spray_equations.log
grep -r "update_particle_positions:" out/*_particles.log | tee out/update_particle_positions.log
grep -r "emitted_particles:" out/*_particles.log | tee out/emitted.log
