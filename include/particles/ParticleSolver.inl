#include <stdio.h>

#include "particles/ParticleSolver.hpp"
#include "visit/VisitWriter.hpp"



namespace minicombust::particles 
{
    using namespace minicombust::visit;

    template<class T>
    void ParticleSolver<T>::output_data(uint64_t timestep)
    {
        // COMMENTED SECTION FOR CLUSTERING PARTICLES TO MESH POINTS
        // mesh->clear_particles_per_point_array();

        // // Assign each particles to one of the vertexes of the bounding cell.
        // for(uint64_t p = 0; p < particles.size(); p++)  // Iterate through each particle
        // {
        //     Particle<T> *particle     = &particles[p];
        //     if ( particle->decayed )  continue;

        //     double closest_dist       = __DBL_MAX__;
        //     uint64_t closest_vertex   = UINT64_MAX;
        //     for (uint64_t i = 0; i < mesh->cell_size; i++)  // Iterate through the points of the cell that the particle is in
        //     {
        //         const uint64_t point_index = mesh->cells[particle->cell*mesh->cell_size + i];
        //         const double dist = magnitude(particle->x1 - mesh->points[point_index]);
        //         if ( dist < closest_dist )
        //         {
        //             closest_dist   = dist;
        //             closest_vertex = point_index;
        //         }
        //     }
        //     mesh->particles_per_point[closest_vertex] += 1;
        // }

        VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh);
        vtk_writer->write_particles("minicombust", timestep, particles);
    }

    template<class T>
    void ParticleSolver<T>::print_logger_stats(uint64_t timesteps, double runtime)
    {
        cout << "Particle Solver Stats:                         " << endl;
        cout << "\tParticles:                                   " << ((double)logger.num_particles)                                                                   << endl;
        cout << "\tParticles (per iter):                        " << particle_dist->particles_per_timestep                                                            << endl;
        cout << "\tEmitted Particles:                           " << logger.emitted_particles                                                                         << endl;
        cout << "\tAvg Particles (per iter):                    " << logger.avg_particles                                                                             << endl;
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
        #ifdef PAPI
        performance_logger.my_papi_start();
        #endif

        // TODO: Reuse decaying particle space
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: particle_release.\n");
        particle_dist->emit_particles(particles, cell_set, &logger);

        #ifdef PAPI
        performance_logger.my_papi_stop(performance_logger.emit_event_counts, &performance_logger.emit_ticks);
        #endif
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
       

        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: solve_spray_equations.\n");

        const uint64_t mesh_size       = mesh->mesh_size; 
        const uint64_t cell_size       = mesh->cell_size; 

        const uint64_t particles_size  = particles.size(); 

        memset(mesh->particle_momentum_rate,    0,    mesh_size*sizeof(vec<T>));
        memset(mesh->particle_energy_rate,      0,    mesh_size*sizeof(T));
        memset(mesh->evaporated_fuel_mass_rate, 0,    mesh_size*sizeof(T));


        #ifdef PAPI
        performance_logger.my_papi_start();
        #endif

        // Solve spray equations
        #pragma ivdep
        for (uint64_t p = 0; p < particles_size; p++)
        {
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

            particles[p].gas_vel           = interp_gas_vel / total_vector_weight;
            particles[p].gas_pressure      = interp_gas_pre / total_scalar_weight;
            particles[p].gas_temperature   = interp_gas_tem / total_scalar_weight;
        }

        #ifdef PAPI
        performance_logger.my_papi_stop(performance_logger.particle_interpolation_event_counts, &performance_logger.particle_interpolation_ticks);
        performance_logger.my_papi_start();
        #endif


        // #pragma ivdep
        // for (uint64_t p = 0; p < particles_size; p++)
        // {
        //     particles[p].solve_spray(mesh, delta, &logger, particles);
        // }

        uint64_t particle_index = 0;
        for (uint64_t p = 0; p < particles_size; p++)
        {
            particles[particle_index].solve_spray(mesh, delta, &logger, particles);

            if (particles[particle_index].decayed)  particles.erase(particles.begin() + particle_index--);
            particle_index++;
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

        cell_set.clear();

        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_particle_positions.\n");
        const uint64_t particles_size  = particles.size(); 
        uint64_t particle_index = 0;

        // Update particle positions
        for (uint64_t p = 0; p < particles_size; p++)
        {   
            // Check if particle is in the current cell. Tetras = Volume/Area comparison method. https://www.peertechzpublications.com/articles/TCSIT-6-132.php.
            particles[particle_index].update_cell(mesh, &logger);


            if ( particles[particle_index].decayed)
            {
                particles.erase(particles.begin() + particle_index--);
            }
            else
            {
                cell_set.insert(particles[particle_index].cell);
            }
            particle_index++;
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
        const uint64_t cell_size  = mesh->cell_size; 

        const T node_neighbours       = 8; // Cube specific

        memset(nodal_flow_aos, 0, point_size * sizeof(flow_aos<T>));


        // FASTER IF particle in most cells

        // memset(nodal_gas_velocity_soa.x,    0, point_size * sizeof(vec<T>));
        // memset(nodal_gas_velocity_soa.y,    0, point_size * sizeof(vec<T>));
        // memset(nodal_gas_velocity_soa.z,    0, point_size * sizeof(vec<T>));
        // memset(nodal_gas_pressure,    0, point_size * sizeof(double));
        // memset(nodal_gas_temperature, 0, point_size * sizeof(double));

        // #pragma ivdep
        // for (uint64_t c = 0; c < mesh_size; c++)
        // {

        //     const uint64_t *cell             = mesh->cells + c*cell_size;
        //     const vec<T> cell_centre         = mesh->cell_centres[c];

        //     const flow_aos<T> flow_term      = mesh->flow_terms[c];      
        //     const flow_aos<T> flow_grad_term = mesh->flow_grad_terms[c]; 
            
            // const T gas_vel_x          = mesh->gas_velocity_soa.x[c];
            // const T gas_vel_y          = mesh->gas_velocity_soa.y[c];
            // const T gas_vel_z          = mesh->gas_velocity_soa.z[c];
            // const T pressure           = mesh->gas_pressure[c];
            // const T pressure_grad      = mesh->gas_pressure_gradient[c];
            // const T temp               = mesh->gas_temperature[c];
            // const T temp_grad          = mesh->gas_temperature_gradient[c];

            // #pragma ivdep
            // for (uint64_t n = 0; n < cell_size; n++)
            // {
            //     const uint64_t point_id = cell[n];

            //     const vec<T> direction             = mesh->points[point_id] - cell_centre;
            //     nodal_flow_aos[point_id].vel      += flow_term.vel      + dot_product(flow_grad_term.vel,      direction);
            //     nodal_flow_aos[point_id].pressure += flow_term.pressure + dot_product(flow_grad_term.pressure, direction);
            //     nodal_flow_aos[point_id].temp     += flow_term.temp     + dot_product(flow_grad_term.temp,     direction);

                // const T direction_x = mesh->points_soa.x[point_id] - mesh->cell_centres_soa.x[c];
                // const T direction_y = mesh->points_soa.y[point_id] - mesh->cell_centres_soa.y[c];
                // const T direction_z = mesh->points_soa.z[point_id] - mesh->cell_centres_soa.z[c];

                // nodal_gas_velocity_soa.x[point_id] += gas_vel_x +     gas_vel_x * (direction_x + direction_y + direction_z);
                // nodal_gas_velocity_soa.y[point_id] += gas_vel_y +     gas_vel_y * (direction_x + direction_y + direction_z);
                // nodal_gas_velocity_soa.z[point_id] += gas_vel_z +     gas_vel_z * (direction_x + direction_y + direction_z);
                // nodal_gas_pressure[point_id]       += pressure  + pressure_grad * (direction_x + direction_y + direction_z);
                // nodal_gas_temperature[point_id]    += temp      +     temp_grad * (direction_x + direction_y + direction_z);
        //     }
        // }

        // #pragma ivdep
        // for (uint64_t n = 0; n < point_size; n++)
        // {
        //     const T boundary_neighbours   = node_neighbours - mesh->cells_per_point[n]; // If nodal counter is not 8, we are on a boundary

        //     nodal_flow_aos[n].vel        = (nodal_flow_aos[n].vel       + boundary_neighbours * mesh->dummy_gas_vel) / node_neighbours;
        //     nodal_flow_aos[n].pressure   = (nodal_flow_aos[n].pressure  + boundary_neighbours * mesh->dummy_gas_pre) / node_neighbours;
        //     nodal_flow_aos[n].temp       = (nodal_flow_aos[n].temp      + boundary_neighbours * mesh->dummy_gas_tem) / node_neighbours;
        // }


        // Faster if particles in some of the cells

        unordered_set<uint64_t> neighbours;
        unordered_set<uint64_t> points;

        #pragma ivdep
        for (unordered_set<uint64_t>::iterator cell_it = cell_set.begin(); cell_it != cell_set.end(); ++cell_it)
        {
            const uint64_t cell = *cell_it;

            for (uint64_t face = 0; face < mesh->faces_per_cell; face++)
            {
                const uint64_t neighbour_id = mesh->cell_neighbours[cell*mesh->faces_per_cell + face];
                if (neighbour_id == MESH_BOUNDARY)  continue;

                neighbours.insert(neighbour_id);
                for (uint64_t face2 = 0; face2 < mesh->faces_per_cell; face2++)
                {
                    const uint64_t neighbour_id2 = mesh->cell_neighbours[neighbour_id*mesh->faces_per_cell + face2];
                    if (neighbour_id2 == MESH_BOUNDARY)  continue;

                    neighbours.insert(neighbour_id2);
                }
            }
        }

        neighbours.insert(cell_set.begin(), cell_set.end());

        #pragma ivdep
        for (unordered_set<uint64_t>::iterator cell_it = neighbours.begin(); cell_it != neighbours.end(); ++cell_it)
        {
            const uint64_t c = *cell_it;

            const uint64_t *cell             = mesh->cells + c*cell_size;
            const vec<T> cell_centre         = mesh->cell_centres[c];

            const flow_aos<T> flow_term      = mesh->flow_terms[c];      
            const flow_aos<T> flow_grad_term = mesh->flow_grad_terms[c]; 

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t point_id = cell[n];

                const vec<T> direction             = mesh->points[point_id] - cell_centre;
                nodal_flow_aos[point_id].vel      += flow_term.vel      + dot_product(flow_grad_term.vel,      direction);
                nodal_flow_aos[point_id].pressure += flow_term.pressure + dot_product(flow_grad_term.pressure, direction);
                nodal_flow_aos[point_id].temp     += flow_term.temp     + dot_product(flow_grad_term.temp,     direction);

                points.insert(point_id);
            }
        }

        // cout << cell_set.size() << " cells size " << neighbours.size() << " cells+neighbours size " << points.size() << " points size" << endl;

        for (unordered_set<uint64_t>::iterator point_it = points.begin(); point_it != points.end(); ++point_it)
        {
            const uint64_t n = *point_it;
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

        static int  count = 0;
        if ((count++ % 20) == 0)  cout << "\tTimestep " << count-1 << ". Particles in simulation: " << particles.size() << " reserved_size " << mesh->max_cell_particles << endl;

        logger.avg_particles += (double)particles.size() / (double)num_timesteps;

        // update_flow_field();
        particle_release();
        interpolate_nodal_data(); 
        solve_spray_equations();
        update_particle_positions();
        // update_spray_source_terms();
        // map_source_terms_to_grid();
        if (PARTICLE_SOLVER_DEBUG)  printf("Stop particle timestep\n");
    }

}   // namespace minicombust::particles 
