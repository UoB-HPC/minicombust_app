#include <stdio.h>

#include "particles/ParticleSolver.hpp"


namespace minicombust::particles 
{
    template<class T> 
    void ParticleSolver<T>::update_flow_field()
    {
        printf("\tRunning fn: update_flow_field.\n");

    }
            
    template<class T> 
    void ParticleSolver<T>::particle_release()
    {
        // TODO: Reuse decaying particle space
        printf("\tRunning fn: particle_release.\n");
        particle_dist->emit_particles(particles + current_particle);
        current_particle += particle_dist->particles_per_timestep;
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
        printf("\tRunning fn: solve_spray_equations.\n");
    }

    template<class T> 
    void ParticleSolver<T>::update_particle_positions()
    {
        printf("\tRunning fn: update_particle_positions.\n");

        // Update particle positions
        global_mesh->clear_particles_per_cell_array();
        for (int p = 0; p < current_particle; p++)
        {
            particles[p].timestep();
            global_mesh->particles_per_cell[particles[p].cell1] += 1;
        }

        // Algorithm for finding nearest point to particle.
        //  - Store the starting cell of each particle.
        //  - Is the particle in the current cell anymore?
        //  - Which face has it intersected with?
        //  - What is the cell id of the cell that shares this face?
        //  - Double check, is particle in this new cell? 
        //  - Yes, stop. No, repeat from new current cell.


    }

    template<class T> 
    void ParticleSolver<T>::update_spray_source_terms()
    {
        printf("\tRunning fn: update_spray_source_terms.\n");
    }

    template<class T> 
    void ParticleSolver<T>::map_source_terms_to_grid()
    {
        printf("\tRunning fn: map_source_terms_to_grid.\n");
    }

    template<class T> 
    void ParticleSolver<T>::interpolate_data()
    {
        printf("\tRunning fn: interpolate_data.\n");
    }

    template<class T> 
    void ParticleSolver<T>::timestep()
    {
        printf("Start particle timestep\n");
        // update_flow_field();
        particle_release();
        // solve_spray_equations();
        update_particle_positions();
        // update_spray_source_terms();
        // map_source_terms_to_grid();
        printf("Stop particle timestep\n");
    }

}   // namespace minicombust::particles 