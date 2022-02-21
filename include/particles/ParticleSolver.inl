#include <stdio.h>

#include "particles/ParticleSolver.hpp"


namespace minicombust::particles 
{
    template<class T> 
    void ParticleSolver<T>::update_flow_field()
    {
        printf("\tRunning function update_flow_field.\n");

    }
            
    template<class T> 
    void ParticleSolver<T>::particle_release()
    {
        // TODO: Reuse decaying particle space
        printf("\tRunning function particle_release.\n");
        particle_dist->emit_particles(particles + current_particle);
        current_particle += particle_dist->particles_per_timestep;
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
        printf("\tRunning function solve_spray_equations.\n");
    }

    template<class T> 
    void ParticleSolver<T>::update_particle_positions()
    {
        printf("\tRunning function update_particle_positions.\n");
        for (int p = 0; p < current_particle; p++)
        {
            particles[p].timestep();
        }
    }

    template<class T> 
    void ParticleSolver<T>::update_spray_source_terms()
    {
        printf("\tRunning function update_spray_source_terms.\n");
    }

    template<class T> 
    void ParticleSolver<T>::map_source_terms_to_grid()
    {
        printf("\tRunning function map_source_terms_to_grid.\n");
    }

    template<class T> 
    void ParticleSolver<T>::interpolate_data()
    {
        printf("\tRunning function interpolate_data.\n");
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