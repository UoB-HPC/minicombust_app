#include <stdio.h>

#include "particles/ParticleSolver.hpp"
#include "visit/VisitWriter.hpp"



namespace minicombust::particles 
{
    using namespace minicombust::visit;

    template<class T>
    void ParticleSolver<T>::output_data(uint64_t timestep)
    {
        mesh->clear_particles_per_point_array();

        // Assign each particles to one of the vertexes of the bounding cell.
        for(uint64_t p = 0; p < current_particle1; p++)  // Iterate through each particle
        {
            Particle<T> *particle     = particles + p;
            if ( particle->decayed )  continue;

            double closest_dist       = __DBL_MAX__;
            uint64_t closest_vertex   = UINT64_MAX;
            for (uint64_t i = 0; i < mesh->cell_size; i++)  // Iterate through the points of the cell that the particle is in
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

        VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh, current_particle1, particles);
        vtk_writer->write_particles("minicombust", timestep);
    }

    template<class T>
    void ParticleSolver<T>::print_logger_stats(uint64_t timesteps, double runtime)
    {
        cout << "Particle Solver Stats:                         " << endl;
        cout << "\tParticles:                                   " << ((double)logger.num_particles)                                                                   << endl;
        cout << "\tParticles (per iter):                        " << particle_dist->particles_per_timestep                                                            << endl;
        cout << "\tEmitted Particles:                           " << logger.emitted_particles                                                            << endl;
        cout << endl;
        cout << "\tCell checks:                                 " << ((double)logger.cell_checks)                                                                     << endl;
        cout << "\tCell checks (per iter):                      " << ((double)logger.cell_checks) / timesteps                                                         << endl;
        cout << "\tCell checks (per particle, per iter):        " << ((double)logger.cell_checks) / (((double)logger.num_particles)*timesteps)                        << endl;
        cout << endl;
        cout << "\tEdge adjustments:                            " << ((double)logger.position_adjustments)                                                            << endl;
        cout << "\tEdge adjustments (per iter):                 " << ((double)logger.position_adjustments) / timesteps                                                << endl;
        cout << "\tEdge adjustments (per particle, per iter):   " << ((double)logger.position_adjustments) / (((double)logger.num_particles)*timesteps)               << endl;
        cout << endl;
        cout << "\tBoundary Intersections:                      " << ((double)logger.boundary_intersections)                                                          << endl;
        cout << "\tDecayed Particles:                           " << round(10000.*(((double)logger.decayed_particles) / ((double)logger.num_particles)))/100. << "% " << endl;
        cout << "\tBurnt Particles:                             " << ((double)logger.burnt_particles) << " " << endl;
        cout << "\tBreakups:                                    " << ((double)logger.breakups) << " " << endl;
        cout << "\tUnallowed breakups:                          " << ((double)logger.unsplit_particles) << " " << endl;

        cout << endl;
        cout << "\tGFLOPS:                                      " << ((double)logger.flops)  / 1000000000.0                                                          << endl;
        cout << "\tPerformance (GFLOPS/s):                      " << (((double)logger.flops) / 1000000000.0) / runtime                                               << endl;
        cout << "\tMemory Loads (GB):                           " << ((double)logger.loads)  / 1000000000.0                                                          << endl;    
        cout << "\tMemory Stores (GB):                          " << ((double)logger.stores) / 1000000000.0                                                          << endl;    
        cout << "\tMemory Bandwidth (GB):                       " << (((double)logger.loads + (double)logger.stores) / 1000000000.0) /runtime                        << endl;    
        cout << "\tOperational Intensity:                       " << (double)logger.flops / ((double)logger.loads + (double)logger.stores)                           << endl;    
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
        if ((current_particle1 + particle_dist->particles_per_timestep) < (mesh->max_cell_particles * mesh->mesh_size))
        {
            if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: particle_release.\n");
            particle_dist->emit_particles(particles, current_particle1);
            current_particle1         += particle_dist->particles_per_timestep;
            logger.num_particles      += particle_dist->particles_per_timestep;
            logger.emitted_particles  += particle_dist->particles_per_timestep;
            current_particle0          = current_particle1;
            return;
        }
        
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: solve_spray_equations.\n");

        for (uint64_t c = 0; c < mesh->mesh_size; c++)
        {
            mesh->evaporated_fuel_mass_rate[c] = 0.0;
            mesh->particle_energy_rate[c]      = 0.0;
            mesh->particle_momentum_rate[c]    = {0.0, 0.0, 0.0};
        }

        if (LOGGER)
        {
            logger.stores += mesh->mesh_size * (sizeof(vec<T>) + 2 * sizeof(T)); 
        }

        // Solve spray equations
        for (uint64_t p = 0; p < current_particle0; p++)
        {
            if (particles[p].decayed)  continue;;
            vec<T> total_vector_weight   = {0.0, 0.0, 0.0};
            T total_scalar_weight        = 0.0;

            vec<T> interp_gas_vel = {0.0, 0.0, 0.0};
            T interp_gas_pre      = 0.0;
            T interp_gas_tem      = 0.0;
            for (uint64_t n = 0; n < mesh->cell_size; n++)
            {
                const uint64_t node           = mesh->cells[particles[p].cell*mesh->cell_size + n]; 
                const vec<T> node_to_particle = particles[p].x1 - mesh->points[mesh->cells[particles[p].cell*mesh->cell_size + n]];

                vec<T> weight      = 1.0 / (node_to_particle * node_to_particle);
                T weight_magnitude = magnitude(weight);
                
                total_vector_weight   += weight;
                total_scalar_weight   += weight_magnitude;
                interp_gas_vel        += weight           * nodal_gas_velocity[node];
                interp_gas_pre        += weight_magnitude * nodal_gas_pressure[node];
                interp_gas_tem        += weight_magnitude * nodal_gas_temperature[node];
            }

            interp_gas_vel /= total_vector_weight;
            interp_gas_pre /= total_scalar_weight;
            interp_gas_tem /= total_scalar_weight;
            // if (p == 0)    cout << p << " " << print_vec(particles[p].a1) << " " << print_vec(particles[p].x1) << " " << particles[p].decayed << endl;
            Particle<T> *daughter_droplet = particles[p].solve_spray(mesh, delta, &logger, interp_gas_vel, interp_gas_pre, interp_gas_tem, current_particle1 < (mesh->max_cell_particles * mesh->mesh_size));
            // if (p == 0)    cout << p << " " << print_vec(particles[p].a1) << " " << print_vec(particles[p].x1) << " " << particles[p].decayed << endl << endl;
            
            if (daughter_droplet != nullptr )
            {
                particles[current_particle1] = *daughter_droplet;
                current_particle1++;
                logger.num_particles++;
            }
            // else if (current_particle1 >= mesh->max_cell_particles * mesh->mesh_size)
            // {
            //     cout << "Splitting a particle but will cause overflow. Not breaking up." << endl;
            // }
        }
    }

    template<class T> 
    void ParticleSolver<T>::update_particle_positions()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_particle_positions.\n");
        
        // Update particle positions
        for (uint64_t p = 0; p < current_particle0; p++)
        {   
            particles[p].timestep(mesh, delta, &logger);
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
    void ParticleSolver<T>::interpolate_nodal_data()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: interpolate_data.\n");

        memset(nodal_counter, 0, mesh->points_size * sizeof(uint8_t));
        for (uint64_t n = 0; n < mesh->points_size; n++)
        {
            nodal_gas_velocity[n]     = {0.0, 0.0, 0.0};
            nodal_gas_pressure[n]     = 0.0;
            nodal_gas_temperature[n]  = 0.0;
        }

        
        for (uint64_t c = 0; c < mesh->mesh_size; c++)
        {
            // Can we mark all neighbours that we need for average
            vec<T> vel           = mesh->gas_velocity[c];
            vec<T> vel_grad      = mesh->gas_velocity_gradient[c];
            T pressure           = mesh->gas_pressure[c];
            T pressure_grad      = mesh->gas_pressure_gradient[c];
            T temp               = mesh->gas_temperature[c];
            T temp_grad          = mesh->gas_temperature_gradient[c];
            
            for (uint64_t n = 0; n < mesh->cell_size; n++)
            {
                const uint64_t point_id = mesh->cells[c*mesh->cell_size + n];
                nodal_counter[point_id]++;


                vec<T> direction                  = mesh->points[point_id] - mesh->cell_centres[c];
                nodal_gas_velocity[point_id]     += vel      + dot_product(vel_grad, direction);
                nodal_gas_pressure[point_id]     += pressure + dot_product(pressure_grad, direction);
                nodal_gas_temperature[point_id]  += temp     + dot_product(temp_grad, direction);
            }
        }



        
        for (uint64_t n = 0; n < mesh->points_size; n++)
        {
            const T node_neighbours       = 8; // Cube specific
            const T boundary_neighbours   = node_neighbours - nodal_counter[n]; // If nodal counter is not 4, we are on a boundary

            nodal_gas_velocity[n]     += boundary_neighbours * (vec<T>){0.1, 0.1, 0.1};
            nodal_gas_pressure[n]     += boundary_neighbours * 960;
            nodal_gas_temperature[n]  += boundary_neighbours * 560.;
            
            nodal_gas_velocity[n]     /= node_neighbours;
            nodal_gas_pressure[n]     /= node_neighbours;
            nodal_gas_temperature[n]  /= node_neighbours;
        }
    }

    

    template<class T> 
    void ParticleSolver<T>::timestep()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("Start particle timestep\n");
        // update_flow_field();
        interpolate_nodal_data();
        particle_release();
        solve_spray_equations();
        update_particle_positions();
        // update_spray_source_terms();
        // map_source_terms_to_grid();
        if (PARTICLE_SOLVER_DEBUG)  printf("Stop particle timestep\n");
    }

}   // namespace minicombust::particles 
