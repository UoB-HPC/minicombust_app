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

        const uint64_t mesh_size  = mesh->mesh_size; 
        const uint64_t cell_size  = mesh->cell_size; 


        memset(mesh->particle_momentum_rate,    0,    mesh_size*sizeof(vec<T>));
        memset(mesh->particle_energy_rate,      0,    mesh_size*sizeof(T));
        memset(mesh->evaporated_fuel_mass_rate, 0,    mesh_size*sizeof(T));

        // Solve spray equations
        #pragma ivdep
        for (uint64_t p = 0; p < current_particle0; p++)
        {
            if (particles[p].decayed)  continue;
            vec<T> total_vector_weight   = {0.0, 0.0, 0.0};
            T total_scalar_weight        = 0.0;

            vec<T> interp_gas_vel = {0.0, 0.0, 0.0};
            T interp_gas_pre      = 0.0;
            T interp_gas_tem      = 0.0;
            
            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node           = mesh->cells[particles[p].cell*cell_size + n]; 
                const vec<T> node_to_particle = particles[p].x1 - mesh->points[mesh->cells[particles[p].cell*cell_size + n]];

                vec<T> weight      = 1.0 / (node_to_particle * node_to_particle);
                T weight_magnitude = magnitude(weight);
                
                total_vector_weight   += weight;
                total_scalar_weight   += weight_magnitude;
                interp_gas_vel        += weight           * nodal_flow_aos[node].vel;
                interp_gas_pre        += weight_magnitude * nodal_flow_aos[node].pressure;
                interp_gas_tem        += weight_magnitude * nodal_flow_aos[node].temp;
            }

            interp_gas_vel /= total_vector_weight;
            interp_gas_pre /= total_scalar_weight;
            interp_gas_tem /= total_scalar_weight;


            // if (p == 0)    cout << p << " " << print_vec(particles[p].a1) << " " << print_vec(particles[p].x1) << " " << particles[p].decayed << endl;
            particles[p].solve_spray(mesh, delta, &logger, interp_gas_vel, interp_gas_pre, interp_gas_tem, &current_particle1, particles);
            // if (p == 0)    cout << p << " " << print_vec(particles[p].a1) << " " << print_vec(particles[p].x1) << " " << particles[p].decayed << endl << endl;

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
            if ( particles[current_particle0].decayed)  continue;
            // Check if particle is in the current cell. Tetras = Volume/Area comparison method. https://www.peertechzpublications.com/articles/TCSIT-6-132.php.
            particles[current_particle0].update_cell(mesh, &logger);
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

        const T node_neighbours       = 8; // Cube specific

        memset(nodal_flow_aos, 0, point_size * sizeof(flow_aos<T>));


        // memset(nodal_gas_velocity_soa.x,    0, point_size * sizeof(vec<T>));
        // memset(nodal_gas_velocity_soa.y,    0, point_size * sizeof(vec<T>));
        // memset(nodal_gas_velocity_soa.z,    0, point_size * sizeof(vec<T>));
        // memset(nodal_gas_pressure,    0, point_size * sizeof(double));
        // memset(nodal_gas_temperature, 0, point_size * sizeof(double));
        

        #pragma ivdep
        for (uint64_t c = 0; c < mesh_size; c++)
        {
            const uint64_t *cell       = mesh->cells + c*cell_size;
            const vec<T> cell_centre         = mesh->cell_centres[c];

            const flow_aos<T> flow_term      = mesh->flow_terms[c];      
            const flow_aos<T> flow_grad_term = mesh->flow_grad_terms[c]; 
            
            // const T gas_vel_x          = mesh->gas_velocity_soa.x[c];
            // const T gas_vel_y          = mesh->gas_velocity_soa.y[c];
            // const T gas_vel_z          = mesh->gas_velocity_soa.z[c];
            // const T pressure           = mesh->gas_pressure[c];
            // const T pressure_grad      = mesh->gas_pressure_gradient[c];
            // const T temp               = mesh->gas_temperature[c];
            // const T temp_grad          = mesh->gas_temperature_gradient[c];

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t point_id = cell[n];

                const vec<T> direction             = mesh->points[point_id] - cell_centre;
                nodal_flow_aos[point_id].vel      += flow_term.vel      + dot_product(flow_grad_term.vel,      direction);
                nodal_flow_aos[point_id].pressure += flow_term.pressure + dot_product(flow_grad_term.pressure, direction);
                nodal_flow_aos[point_id].temp     += flow_term.temp     + dot_product(flow_grad_term.temp,     direction);

                // const T direction_x = mesh->points_soa.x[point_id] - mesh->cell_centres_soa.x[c];
                // const T direction_y = mesh->points_soa.y[point_id] - mesh->cell_centres_soa.y[c];
                // const T direction_z = mesh->points_soa.z[point_id] - mesh->cell_centres_soa.z[c];

                // nodal_gas_velocity_soa.x[point_id] += gas_vel_x +     gas_vel_x * (direction_x + direction_y + direction_z);
                // nodal_gas_velocity_soa.y[point_id] += gas_vel_y +     gas_vel_y * (direction_x + direction_y + direction_z);
                // nodal_gas_velocity_soa.z[point_id] += gas_vel_z +     gas_vel_z * (direction_x + direction_y + direction_z);
                // nodal_gas_pressure[point_id]       += pressure  + pressure_grad * (direction_x + direction_y + direction_z);
                // nodal_gas_temperature[point_id]    += temp      +     temp_grad * (direction_x + direction_y + direction_z);
            }
        }

        #pragma ivdep
        for (uint64_t n = 0; n < point_size; n++)
        {
            const T boundary_neighbours   = node_neighbours - mesh->cells_per_point[n]; // If nodal counter is not 8, we are on a boundary

            nodal_flow_aos[n].vel        = (nodal_flow_aos[n].vel       + boundary_neighbours * mesh->dummy_gas_vel) / node_neighbours;
            nodal_flow_aos[n].pressure   = (nodal_flow_aos[n].pressure  + boundary_neighbours * mesh->dummy_gas_pre) / node_neighbours;
            nodal_flow_aos[n].temp       = (nodal_flow_aos[n].temp      + boundary_neighbours * mesh->dummy_gas_tem) / node_neighbours;
        }

        


        #ifdef PAPI
        performance_logger.my_papi_stop(performance_logger.interpolation_kernel_event_counts, &performance_logger.interpolation_ticks);
        #endif

        // for (uint64_t n = 0; n < point_size; n++)
        // {
        //     nodal_gas_velocity[n]     = nodal_flow_aos[n].vel;
        //     nodal_gas_pressure[n]     = nodal_flow_aos[n].pressure;
        //     nodal_gas_temperature[n]  = nodal_flow_aos[n].temp;
        // }
        
    }

    

    template<class T> 
    void ParticleSolver<T>::timestep()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("Start particle timestep\n");
        cout << "Particles in simulation: " << logger.num_particles - logger.decayed_particles << endl;
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
