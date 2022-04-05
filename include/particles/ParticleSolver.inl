#include <stdio.h>

#include "particles/ParticleSolver.hpp"
#include "visit/VisitWriter.hpp"

#define PARTICLE_SOLVER_DEBUG 0


namespace minicombust::particles 
{
    using namespace minicombust::visit;

    template<class T>
    void ParticleSolver<T>::output_data(int timestep)
    {
        mesh->clear_particles_per_point_array();

        // Assign each particles to one of the vertexes of the bounding cell.
        for(int p = 0; p < current_particle; p++)  // Iterate through each particle
        {
            Particle<T> *particle     = particles + p;
            if ( particle->decayed )  continue;

            double closest_dist       = __DBL_MAX__;
            uint64_t closest_vertex   = UINT64_MAX;
            for (int i = 0; i < mesh->cell_size; i++)  // Iterate through the points of the cell that the particle is in
            {
                const uint64_t point_index = mesh->cells[particle->cell*mesh->cell_size + i];
                const double dist = magnitude(particle->x1 - mesh->points[point_index]);
                if ( dist < closest_dist )
                {
                    closest_dist   = dist;
                    closest_vertex = point_index;
                }
            }
            mesh->particles_per_point[closest_vertex] += 1;
        }

        VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh);
        vtk_writer->write_particles("minicombust", timestep);
    }

    template<class T>
    void ParticleSolver<T>::print_logger_stats(int timesteps)
    {
        cout << "Particle Solver Stats:                         " << endl;
        cout << "\tParticles:                                   " << logger.num_particles                                                         << endl;
        cout << "\tParticles (per iter):                        " << particle_dist->particles_per_timestep                                        << endl;
        cout << endl;
        cout << "\tCell checks:                                 " << logger.cell_checks                                                           << endl;
        cout << "\tCell checks (per iter):                      " << logger.cell_checks / timesteps                                               << endl;
        cout << "\tCell checks (per particle, per iter):        " << logger.cell_checks / (logger.num_particles*timesteps)                        << endl;
        cout << endl;
        cout << "\tStart adjustments:                           " << logger.position_adjustments                                                           << endl;
        cout << "\tStart adjustments (per iter):                " << logger.position_adjustments / timesteps                                               << endl;
        cout << "\tStart adjustments (per particle, per iter):  " << logger.position_adjustments / (logger.num_particles*timesteps)                        << endl;
        cout << endl;
        cout << "\tBoundary Intersections:                      " << logger.boundary_intersections                                                << endl;
        cout << "\tDecayed Particles:                           " << round(10000.*(logger.decayed_particles / logger.num_particles))/100. << "% " << endl;
    }


    template<class T> 
    void ParticleSolver<T>::update_flow_field()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_flow_field.\n");

    }
            
    template<class T> 
    void ParticleSolver<T>::particle_release()
    {
        // TODO: Reuse decaying particle space
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: particle_release.\n");
        particle_dist->emit_particles(particles + current_particle);
        current_particle += particle_dist->particles_per_timestep;
        logger.num_particles += particle_dist->particles_per_timestep;
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: solve_spray_equations.\n");

        // Solve spray equations
        for (int p = 0; p < current_particle; p++)
        {
            particles[p].timestep(mesh, 0.1, &logger);
        }
    }

    template<class T> 
    void ParticleSolver<T>::update_particle_positions()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_particle_positions.\n");
        
        // Update particle positions
        for (int p = 0; p < current_particle; p++)
        {
            particles[p].solve_spray(mesh, 0.1);
        }

    }

    template<class T> 
    void ParticleSolver<T>::update_spray_source_terms()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_spray_source_terms.\n");
    }

    template<class T> 
    void ParticleSolver<T>::map_source_terms_to_grid()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: map_source_terms_to_grid.\n");
    }

    template<class T> 
    void ParticleSolver<T>::interpolate_data()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: interpolate_data.\n");
    }

    template<class T> 
    void ParticleSolver<T>::timestep()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("Start particle timestep\n");
        // update_flow_field();
        particle_release();
        solve_spray_equations();
        update_particle_positions();
        // update_spray_source_terms();
        // map_source_terms_to_grid();
        if (PARTICLE_SOLVER_DEBUG)  printf("Stop particle timestep\n");
    }

}   // namespace minicombust::particles 