#include <stdio.h>

#include "particles/ParticleSolver.hpp"
#include "visit/VisitWriter.hpp"


namespace minicombust::particles 
{
    using namespace minicombust::visit;

    template<class T>
    void ParticleSolver<T>::output_data()
    {
        static int count = 0;
        global_mesh->clear_particles_per_point_array();
        // Assign each particles to one of the vertexes of the bounding cell.
        for(int p = 0; p < current_particle; p++)
        {
            Particle<T> *particle     = particles + p;
            double closest_dist    = __DBL_MAX__;
            vec<T> *closest_vertex = nullptr;
            for (int i = 0; i < global_mesh->cell_size; i++)
            {
                const double dist = magnitude(particle->x1 - *global_mesh->cells[particle->cell][i]);
                if ( dist < closest_dist )
                {
                    closest_dist   = dist;
                    closest_vertex = global_mesh->cells[particle->cell][i];
                }
            }
            global_mesh->particles_per_point[closest_vertex - global_mesh->mesh_points] += 1;
        }

        VisitWriter<double> *vtk_writer = new VisitWriter<double>(global_mesh);
        vtk_writer->write_file("minicombust", count++);
    }

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
        
        output_data();

        // Update particle positions
        for (int p = 0; p < current_particle; p++)
        {
            particles[p].timestep(global_mesh);
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