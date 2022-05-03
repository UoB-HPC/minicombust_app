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

        // cout << endl;
        // cout << "\tGFLOPS:                                      " << ((double)logger.flops)  / 1000000000.0                                                          << endl;
        // cout << "\tPerformance (GFLOPS/s):                      " << (((double)logger.flops) / 1000000000.0) / runtime                                               << endl;
        // cout << "\tMemory Loads (GB):                           " << ((double)logger.loads)  / 1000000000.0                                                          << endl;    
        // cout << "\tMemory Stores (GB):                          " << ((double)logger.stores) / 1000000000.0                                                          << endl;    
        // cout << "\tMemory Bandwidth (GB):                       " << (((double)logger.loads + (double)logger.stores) / 1000000000.0) /runtime                        << endl;    
        // cout << "\tOperational Intensity:                       " << (double)logger.flops / ((double)logger.loads + (double)logger.stores)                           << endl;  

        #ifdef PAPI
        cout << endl;
        performance_logger.print_counters();
        #endif  
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
        #ifdef PAPI
        performance_logger.my_papi_start();
        #endif

        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: solve_spray_equations.\n");

        for (uint64_t c = 0; c < mesh->mesh_size; c++)
        {
            mesh->evaporated_fuel_mass_rate[c] = 0.0;
            mesh->particle_energy_rate[c]      = 0.0;
            mesh->particle_momentum_rate[c]    = {0.0, 0.0, 0.0};
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

        #ifdef PAPI
        performance_logger.my_papi_stop(performance_logger.spray_kernel_event_counts, &performance_logger.spray_ticks);
        #endif
    }

    template<class T> 
    void ParticleSolver<T>::update_particle_positions()
    {
        #ifdef PAPI
        performance_logger.my_papi_start();
        #endif

        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_particle_positions.\n");
        
        // Update particle positions
        for (uint64_t p = 0; p < current_particle0; p++)
        {   
            particles[p].timestep(mesh, delta, &logger);
        }

        #ifdef PAPI
        performance_logger.my_papi_stop(performance_logger.position_kernel_event_counts, &performance_logger.position_ticks);
        #endif
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
        #ifdef PAPI
        performance_logger.my_papi_start();
        #endif

        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: interpolate_data.\n");

        const uint64_t point_size = mesh->points_size; 
        const uint64_t mesh_size  = mesh->mesh_size; 
        const uint64_t cell_size  = mesh->cell_size; 

        memset(nodal_gas_velocity_soa.x,    0, point_size * sizeof(T));
        memset(nodal_gas_velocity_soa.y,    0, point_size * sizeof(T));
        memset(nodal_gas_velocity_soa.z,    0, point_size * sizeof(T));
        memset(nodal_gas_pressure,          0, point_size * sizeof(T));
        memset(nodal_gas_temperature,       0, point_size * sizeof(T));

        #pragma ivdep
        for (uint64_t c = 0; c < mesh_size; c++)
        {
            const uint64_t *cell       = mesh->cells + c*cell_size;

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t point_id = cell[n];

                const T direction_x = mesh->points_soa.x[point_id] - mesh->cell_centres_soa.x[c];
                const T direction_y = mesh->points_soa.y[point_id] - mesh->cell_centres_soa.y[c];
                const T direction_z = mesh->points_soa.z[point_id] - mesh->cell_centres_soa.z[c];

                nodal_gas_velocity_soa.x[point_id] += mesh->gas_velocity_soa.x[c] + mesh->gas_velocity_gradient_soa.x[c] * (direction_x + direction_y + direction_z);
                nodal_gas_velocity_soa.y[point_id] += mesh->gas_velocity_soa.y[c] + mesh->gas_velocity_gradient_soa.y[c] * (direction_x + direction_y + direction_z);
                nodal_gas_velocity_soa.z[point_id] += mesh->gas_velocity_soa.z[c] + mesh->gas_velocity_gradient_soa.z[c] * (direction_x + direction_y + direction_z);
                nodal_gas_pressure[point_id]       += mesh->gas_pressure[c]       + mesh->gas_pressure_gradient[c]       * (direction_x + direction_y + direction_z);
                nodal_gas_temperature[point_id]    += mesh->gas_temperature[c]    + mesh->gas_temperature_gradient[c]    * (direction_x + direction_y + direction_z);
            }
        }



        #pragma ivdep
        for (uint64_t n = 0; n < point_size; n++)
        {
            const T node_neighbours       = 8;                                          // Cube specific
            const T boundary_neighbours   = node_neighbours - mesh->cells_per_point[n]; // If nodal counter is not 8, we are on a boundary

            nodal_gas_velocity_soa.x[n]    = (nodal_gas_velocity_soa.x[n]  + boundary_neighbours * mesh->dummy_gas_vel.x) / node_neighbours;
            nodal_gas_velocity_soa.y[n]    = (nodal_gas_velocity_soa.y[n]  + boundary_neighbours * mesh->dummy_gas_vel.y) / node_neighbours;
            nodal_gas_velocity_soa.z[n]    = (nodal_gas_velocity_soa.z[n]  + boundary_neighbours * mesh->dummy_gas_vel.z) / node_neighbours;
            nodal_gas_pressure[n]          = (nodal_gas_pressure[n]        + boundary_neighbours * mesh->dummy_gas_pre)   / node_neighbours;
            nodal_gas_temperature[n]       = (nodal_gas_temperature[n]     + boundary_neighbours * mesh->dummy_gas_tem)   / node_neighbours;
        }

        #ifdef PAPI
        performance_logger.my_papi_stop(performance_logger.interpolation_kernel_event_counts, &performance_logger.interpolation_ticks);
        #endif
        
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
