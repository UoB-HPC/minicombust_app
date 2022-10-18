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
        Particle_Logger loggers[mpi_config->particle_flow_world_size];
        MPI_Gather(&logger, sizeof(Particle_Logger), MPI_BYTE, &loggers, sizeof(Particle_Logger), MPI_BYTE, 0, mpi_config->particle_flow_world);
        
        memset(&logger,           0, sizeof(Particle_Logger));
        for (int rank = 0; rank < mpi_config->particle_flow_world_size; rank++)
        {
            logger.num_particles            += loggers[rank].num_particles;
            logger.avg_particles            += loggers[rank].avg_particles;
            logger.emitted_particles        += loggers[rank].emitted_particles;
            logger.cell_checks              += loggers[rank].cell_checks;
            logger.position_adjustments     += loggers[rank].position_adjustments;
            logger.lost_particles           += loggers[rank].lost_particles;
            logger.boundary_intersections   += loggers[rank].boundary_intersections;
            logger.decayed_particles        += loggers[rank].decayed_particles;
            logger.burnt_particles          += loggers[rank].burnt_particles;
            logger.breakups                 += loggers[rank].breakups;
            logger.interpolated_cells       += loggers[rank].interpolated_cells / (double)mpi_config->particle_flow_world_size;
        }

        if (mpi_config->rank == 0)
        {
            cout << "Particle Solver Stats:                         " << endl;
            cout << "\tParticles:                                   " << ((double)logger.num_particles)                                                                   << endl;
            cout << "\tParticles (per iter):                        " << particle_dist->particles_per_timestep*mpi_config->particle_flow_world_size                       << endl;
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
            cout << "\tLost Particles:                              " << ((double)logger.lost_particles      )                                                            << endl;
            cout << endl;
            cout << "\tBoundary Intersections:                      " << ((double)logger.boundary_intersections)                                                          << endl;
            cout << "\tDecayed Particles:                           " << round(10000.*(((double)logger.decayed_particles) / ((double)logger.num_particles)))/100. << "% " << endl;
            cout << "\tBurnt Particles:                             " << ((double)logger.burnt_particles)                                                                 << endl;
            cout << "\tBreakups:                                    " << ((double)logger.breakups)                                                                        << endl;
            cout << "\tBreakup Age:                                 " << ((double)logger.breakup_age)                                                                     << endl;
            cout << endl;
            cout << "\tInterpolated Cells (per rank):               " << ((double)logger.interpolated_cells)                                                              << endl;
            cout << "\tInterpolated Cells Percentage (per rank):    " << round(10000.*(((double)logger.interpolated_cells) / ((double)mesh->mesh_size)))/100. << "% "     << endl;

            cout << endl;
        }
        performance_logger.print_counters(mpi_config->rank, runtime);
    }


    template<class T> 
    void ParticleSolver<T>::update_flow_field(bool send_particle)
    {
        performance_logger.my_papi_start();

        const uint64_t cell_size = cell_particle_field_map.size();
        uint64_t cells[cell_size];
        particle_aos<T> cell_particle_fields[cell_size];
        
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_flow_field.\n");
        int flow_rank = mpi_config->particle_flow_world_size;

        uint64_t count = 0;
        for (auto& cell_it: cell_particle_field_map)
        {
            cells[count]                = cell_it.first;
            cell_particle_fields[count] = cell_it.second;
            count++;
        }
        MPI_Gather(&cell_size,         1, MPI_UINT64_T, nullptr, 0,    MPI_UINT64_T, flow_rank, mpi_config->world);
        MPI_Gatherv(cells,     cell_size, MPI_UINT64_T, nullptr, 0, 0, MPI_UINT64_T, flow_rank, mpi_config->world);

        if (send_particle)
        {
            MPI_Gatherv(cell_particle_fields,  cell_size, mpi_config->MPI_PARTICLE_STRUCTURE, nullptr, 0, 0, mpi_config->MPI_PARTICLE_STRUCTURE, flow_rank, mpi_config->world);
        }
        MPI_Bcast(&neighbours_size,                1, MPI_UINT64_T, flow_rank, mpi_config->world);
        
        MPI_Bcast(neighbour_indexes, neighbours_size, MPI_UINT64_T, flow_rank, mpi_config->world);
        logger.interpolated_cells += ((float) neighbours_size) / ((float)num_timesteps);

        MPI_Bcast(cell_flow_aos,      (int)neighbours_size, mpi_config->MPI_FLOW_STRUCTURE, flow_rank, mpi_config->world);
        MPI_Bcast(cell_flow_grad_aos, (int)neighbours_size, mpi_config->MPI_FLOW_STRUCTURE, flow_rank, mpi_config->world);

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
    }
            
    template<class T> 
    void ParticleSolver<T>::particle_release()
    {
        performance_logger.my_papi_start();

        // TODO: Reuse decaying particle space
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: particle_release.\n");
        particle_dist->emit_particles(particles, cell_particle_field_map, &logger);

        performance_logger.my_papi_stop(performance_logger.emit_event_counts, &performance_logger.emit_time);
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
       

        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: solve_spray_equations.\n");

        const uint64_t cell_size       = mesh->cell_size; 

        const uint64_t particles_size  = particles.size(); 

        performance_logger.my_papi_start();

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

        performance_logger.my_papi_stop(performance_logger.particle_interpolation_event_counts, &performance_logger.particle_interpolation_time);
        performance_logger.my_papi_start();

        vector<uint64_t> decayed_particles;
        #pragma ivdep
        for (uint64_t p = 0; p < particles_size; p++)
        {
            particles[p].solve_spray( delta, &logger, particles );

            if (particles[p].decayed)  decayed_particles.push_back(p);
        }

        const uint64_t decayed_particles_size = decayed_particles.size();
        #pragma ivdep
        for (int128_t i = decayed_particles_size - 1; i >= 0; i--)
        {
            particles[decayed_particles[i]] = particles.back();
            particles.pop_back();
        }


        performance_logger.my_papi_stop(performance_logger.spray_kernel_event_counts, &performance_logger.spray_time);
    }

    template<class T> 
    void ParticleSolver<T>::update_particle_positions()
    {
        performance_logger.my_papi_start();

        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_particle_positions.\n");
        const uint64_t particles_size  = particles.size();

        // Update particle positions
        vector<uint64_t> decayed_particles;
        #pragma ivdep
        for (uint64_t p = 0; p < particles_size; p++)
        {   
            // Check if particle is in the current cell. Tetras = Volume/Area comparison method. https://www.peertechzpublications.com/articles/TCSIT-6-132.php.
            particles[p].update_cell(mesh, &logger);

            if (particles[p].decayed)  decayed_particles.push_back(p);
            else
            {
                cell_particle_field_map[particles[p].cell].momentum += particles[p].particle_cell_fields.momentum;
                cell_particle_field_map[particles[p].cell].energy   += particles[p].particle_cell_fields.energy;
                cell_particle_field_map[particles[p].cell].fuel     += particles[p].particle_cell_fields.fuel;
            }
        }


        const uint64_t decayed_particles_size = decayed_particles.size();
        #pragma ivdep
        for (int128_t i = decayed_particles_size - 1; i >= 0; i--)
        {
            particles[decayed_particles[i]] = particles.back();
            particles.pop_back();
        }

        performance_logger.my_papi_stop(performance_logger.position_kernel_event_counts, &performance_logger.position_time);
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
        performance_logger.my_papi_start();

        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: interpolate_data.\n");

        const uint64_t point_size = mesh->points_size; 
        const uint64_t mesh_size  = mesh->mesh_size; 
        const uint64_t cell_size  = mesh->cell_size; 

        const T node_neighbours       = 8; // Cube specific

        memset(nodal_flow_aos, 0, point_size * sizeof(flow_aos<T>));

        const bool WHOLE_MESH = false;

        if (WHOLE_MESH)
        {
            // FASTER IF particle in most cells

            // memset(nodal_gas_velocity_soa.x,    0, point_size * sizeof(vec<T>));
            // memset(nodal_gas_velocity_soa.y,    0, point_size * sizeof(vec<T>));
            // memset(nodal_gas_velocity_soa.z,    0, point_size * sizeof(vec<T>));
            // memset(nodal_gas_pressure,    0, point_size * sizeof(double));
            // memset(nodal_gas_temperature, 0, point_size * sizeof(double));

            #pragma ivdep
            for (uint64_t c = 0; c < mesh_size; c++)
            {

                const uint64_t *cell             = mesh->cells + c*cell_size;
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

        }
        else
        {
            // Faster if particles in some of the cells
            static int time = 0;
            time++;
            unordered_set<uint64_t> points;

            #pragma ivdep
            for (uint64_t i = 0; i < neighbours_size; i++)
            {

                const uint64_t c = neighbour_indexes[i];

                const uint64_t *cell             = mesh->cells + c*cell_size;
                const vec<T> cell_centre         = mesh->cell_centres[c];

                const flow_aos<T> flow_term      = cell_flow_aos[i];      
                const flow_aos<T> flow_grad_term = cell_flow_grad_aos[i]; 

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


            for (unordered_set<uint64_t>::iterator point_it = points.begin(); point_it != points.end(); ++point_it)
            {
                const uint64_t n = *point_it;
                const T boundary_neighbours   = node_neighbours - mesh->cells_per_point[n]; // If nodal counter is not 8, we are on a boundary

                nodal_flow_aos[n].vel        = (nodal_flow_aos[n].vel       + boundary_neighbours * mesh->dummy_gas_vel) / node_neighbours;
                nodal_flow_aos[n].pressure   = (nodal_flow_aos[n].pressure  + boundary_neighbours * mesh->dummy_gas_pre) / node_neighbours;
                nodal_flow_aos[n].temp       = (nodal_flow_aos[n].temp      + boundary_neighbours * mesh->dummy_gas_tem) / node_neighbours;
            }
            cell_particle_field_map.clear();
        }


        performance_logger.my_papi_stop(performance_logger.interpolation_kernel_event_counts, &performance_logger.interpolation_time);
        
    }


    template<class T> 
    void ParticleSolver<T>::timestep()
    {
        static int count = 0;
        const int  comms_timestep = 1;

        if (PARTICLE_SOLVER_DEBUG)  printf("Start particle timestep\n");
        if ( (count % 100) == 0 && mpi_config->particle_flow_rank == 0 )  
            cout << "\tTimestep " << count << ". Particles in simulation estimate (rank0 * num_ranks): " << particles.size() * mpi_config->particle_flow_world_size << " reserved_size " << reserve_particles_size << endl;

        particle_release();
        if (mpi_config->world_size != 1 && (count % comms_timestep) == 0)
        {
            update_flow_field(count > 0);
            interpolate_nodal_data(); 
        }
        else if (mpi_config->world_size == 1)
        {
            interpolate_nodal_data(); 
        }
        solve_spray_equations();
        update_particle_positions();
        // update_spray_source_terms();
        // map_source_terms_to_grid();

        logger.avg_particles += (double)particles.size() / (double)num_timesteps;

        count++;

        if (PARTICLE_SOLVER_DEBUG)  printf("Stop particle timestep\n");
    }

}   // namespace minicombust::particles 