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

        VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh, mpi_config);
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
            logger.sent_cells               += loggers[rank].sent_cells              / (double)  mpi_config->particle_flow_world_size;
            logger.sent_cells_per_block     += loggers[rank].sent_cells_per_block    / (double)  mpi_config->particle_flow_world_size;
            logger.nodes_recieved           += loggers[rank].nodes_recieved          / (double)  mpi_config->particle_flow_world_size;
            logger.useful_nodes_proportion  += loggers[rank].useful_nodes_proportion / (double)  mpi_config->particle_flow_world_size;
        }

        MPI_Barrier(mpi_config->world);

        if (mpi_config->rank == 0)
        {
            cout << "Particle Solver Stats:                         " << endl;
            cout << "\tParticles:                                   " << ((double)logger.num_particles)                                                                   << endl;
            cout << "\tParticles (per iter):                        " << particle_dist->even_particles_per_timestep*mpi_config->particle_flow_world_size                  << endl;
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
            cout << "\tDecayed Particles:                           " << ((double)logger.decayed_particles)                                                               << endl;
            cout << "\tDecayed Particles:                           " << round(10000.*(((double)logger.decayed_particles) / ((double)logger.num_particles)))/100. << "% " << endl;
            cout << "\tBurnt Particles:                             " << ((double)logger.burnt_particles)                                                                 << endl;
            cout << "\tBreakups:                                    " << ((double)logger.breakups)                                                                        << endl;
            cout << "\tBreakup Age:                                 " << ((double)logger.breakup_age)                                                                     << endl;
            cout << endl; 
            cout << "\tAvg Sent Cells       (avg per rank, block):  " << round(logger.sent_cells_per_block / timesteps)                                                   << endl;
            cout << "\tTotal Sent Cells     (avg per rank):         " << round(logger.sent_cells / timesteps)                                                             << endl;
            cout << "\tTotal Recieved Nodes (avg per rank):         " << round(logger.nodes_recieved / timesteps)                                                         << endl;
            cout << "\tUseful Nodes         (avg per rank):         " << round(logger.useful_nodes_proportion / timesteps)                                                << endl;
            cout << "\tUseful Nodes (%)     (avg per rank):         " << round(10000.*((logger.useful_nodes_proportion) / (logger.nodes_recieved))) / 100. << "% "        << endl;

            cout << endl;

            cout <<"NOTE: REDUCING RELATIVE GAS VEL by 50\% in Particle.hpp while flow isn't implemented!!!" << endl;
            cout << endl;
        }

        performance_logger.print_counters(mpi_config->rank, mpi_config->world_size, runtime);
    }


    template<class T> 
    void ParticleSolver<T>::update_flow_field()
    {
        performance_logger.my_papi_start();

        if (PARTICLE_SOLVER_DEBUG )  printf("\tRank %d: Running fn: update_flow_field.\n", mpi_config->rank);

        active_blocks.clear();

        double avg_sent_cells  = 0.;
        double non_zero_blocks = 0.;

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            cell_particle_field_map[b].erase(MESH_BOUNDARY);
            logger.sent_cells     += cell_particle_field_map[b].size();
            avg_sent_cells        += cell_particle_field_map[b].size();
            non_zero_blocks       += cell_particle_field_map[b].size() > 0;

            neighbours_size[b]   = cell_particle_field_map[b].size();
            
            if ( cell_particle_field_map[b].size() )
                active_blocks.push_back(b);
        }
        avg_sent_cells /= non_zero_blocks;
        logger.sent_cells_per_block += avg_sent_cells;


        for (uint64_t b : active_blocks)
        {
            neighbours_size[b] = cell_particle_field_map[b].size();

            MPI_Issend(&neighbours_size[b], 1, MPI_UINT64_T, mpi_config->particle_flow_world_size + b, 0, mpi_config->world, &requests[b] );
        }

        for (uint64_t b : active_blocks)
        {
            MPI_Wait(&requests[b], MPI_STATUS_IGNORE);
        }

        MPI_Barrier(mpi_config->particle_flow_world);

        bool sends_done = 1;
        MPI_Ibcast(&sends_done, 1, MPI_INT, 0, mpi_config->world, &requests[0]);

        MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
        MPI_Barrier(mpi_config->world);

        for (uint64_t b : active_blocks)
        {
            MPI_Isend(cell_particle_indexes[b],  neighbours_size[b], MPI_UINT64_T,                       mpi_config->particle_flow_world_size + b, 1, mpi_config->world, &requests[3*b + 1] );
            MPI_Isend(cell_particle_aos[b],      neighbours_size[b], mpi_config->MPI_PARTICLE_STRUCTURE, mpi_config->particle_flow_world_size + b, 2, mpi_config->world, &requests[3*b + 2] );
        }

        for (uint64_t b : active_blocks)
        {
            MPI_Waitall( 2, &requests[3*b + 1], MPI_STATUSES_IGNORE );
        }

        MPI_Barrier(mpi_config->world);

        for (uint64_t b : active_blocks)
            cell_particle_field_map[b].clear();

        for (uint64_t b : active_blocks)
        {
            // printf("Rank %d Waiting %d\n", mpi_config->rank, b);

            MPI_Probe (MPI_ANY_SOURCE, 0, mpi_config->world, &statuses[b] );

            const uint64_t send_rank = statuses[b].MPI_SOURCE;
            const uint64_t block_id  = statuses[b].MPI_SOURCE - mpi_config->particle_flow_world_size;
            MPI_Get_count( &statuses[b], MPI_UINT64_T, (int*)&neighbours_size[block_id] );
            resize_nodes_arrays(neighbours_size);

            logger.nodes_recieved += neighbours_size[block_id];
            // printf("Rank %d Recieving data from %d\n", mpi_config->rank ,b);
            MPI_Irecv ( all_interp_node_indexes[block_id],     neighbours_size[block_id], MPI_UINT64_T,                   send_rank, 0, mpi_config->world, &requests[3 * block_id + 0] );
            MPI_Irecv ( all_interp_node_flow_fields[block_id], neighbours_size[block_id], mpi_config->MPI_FLOW_STRUCTURE, send_rank, 1, mpi_config->world, &requests[3 * block_id + 1] );
        }

        uint64_t ba = active_blocks.size() - 1;
        uint64_t bi;
        bool     all_processed   = true;
        bool    *processed_block = async_locks;
        for (uint64_t b : active_blocks)  
        {
            processed_block[b]  = 0;
            all_processed      &= processed_block[b];
        }

        while (!all_processed)
        {
            ba = (ba + 1) % active_blocks.size();
            bi = active_blocks[ba];

            int recieve_done = 0;
            MPI_Test(&requests[3*bi + 0], &recieve_done, MPI_STATUS_IGNORE);

            if (recieve_done && !processed_block[bi])
            {
                for (uint64_t i = 0; i < neighbours_size[bi]; i++)
                {
                    // if (node_to_field_address_map.count(all_interp_node_indexes[bi][i]))
                    // {
                        node_to_field_address_map[all_interp_node_indexes[bi][i]] = &all_interp_node_flow_fields[bi][i];
                    // }
                }
                processed_block[bi] = true;
            }

            all_processed = true;
            for (uint64_t b : active_blocks)  all_processed &= processed_block[b];
        }

        

        for (uint64_t b : active_blocks)
        {
            MPI_Wait(&requests[3*b + 1], MPI_STATUS_IGNORE);
        }


        
        logger.useful_nodes_proportion += node_to_field_address_map.size();

        
        MPI_Barrier(mpi_config->world);

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
    }
            
    template<class T> 
    void ParticleSolver<T>::particle_release()
    {
        performance_logger.my_papi_start();

        // TODO: Reuse decaying particle space
        if (PARTICLE_SOLVER_DEBUG )  printf("\tRank %d: Running fn: particle_release.\n", mpi_config->rank);
        function<void(uint64_t *, uint64_t ***, particle_aos<T> ***)> resize_cell_particles_fn = [this] (uint64_t *elements, uint64_t ***indexes, particle_aos<T> ***cell_particle_fields) { return resize_cell_particle(elements, indexes, cell_particle_fields); };

        particle_dist->emit_particles_evenly(particles, cell_particle_field_map, node_to_field_address_map, cell_particle_indexes, cell_particle_aos, resize_cell_particles_fn, &logger);
        // particle_dist->emit_particles_waves(particles, cell_particle_field_map, cell_particle_indexes, cell_particle_aos,  &logger);

        performance_logger.my_papi_stop(performance_logger.emit_event_counts, &performance_logger.emit_time);
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
        if (PARTICLE_SOLVER_DEBUG )  printf("\tRank %d: Running fn: solve_spray_equations.\n", mpi_config->rank);

        // printf("Rank %d particles size %lu\n", mpi_config->rank, particles.size());


        const uint64_t cell_size       = mesh->cell_size; 

        const uint64_t particles_size  = particles.size(); 

        performance_logger.my_papi_start();

        // unordered_set<uint64_t> REMOVE_TEST_num_points;

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
                const int64_t node = mesh->cells[(particles[p].cell - mesh->shmem_cell_disp) * cell_size + n];

                const vec<T> node_to_particle = particles[p].x1 - mesh->points[node - mesh->shmem_point_disp];

                vec<T> weight      = 1.0 / ((node_to_particle * node_to_particle) + vec<T> {__DBL_MIN__, __DBL_MIN__, __DBL_MIN__});
                T weight_magnitude = magnitude(weight);
                // cout << "Weight mag " << print_vec(weight) << " particles[p].x1 " << print_vec(particles[p].x1)  << "mesh->points[node - mesh->shmem_point_disp]" << print_vec(mesh->points[node - mesh->shmem_point_disp]) << endl;

                total_vector_weight   += weight;
                total_scalar_weight   += weight_magnitude;


                // if (node_to_field_address_map[node]->temp     != mesh->dummy_gas_tem)              
                //     {printf("ERROR SOLVE SPRAY : Wrong temp value %f at %lu (cell %lu)\n",          node_to_field_address_map[node]->temp,     node, particles[p].cell); exit(1);}
                // if (node_to_field_address_map[node]->pressure != mesh->dummy_gas_pre)              
                //     {printf("ERROR SOLVE SPRAY : Wrong pres value %f at %lu (cell %lu)\n",          node_to_field_address_map[node]->pressure, node, particles[p].cell); exit(1);}
                // if (node_to_field_address_map[node]->vel.x != mesh->dummy_gas_vel.x) 
                //     {printf("ERROR SOLVE SPRAY : Wrong velo value {%.10f y z} at %lu (cell %lu)\n", node_to_field_address_map[node]->vel.x,    node, particles[p].cell); exit(1);}

                interp_gas_vel        += weight           * node_to_field_address_map[node]->vel;
                interp_gas_pre        += weight_magnitude * node_to_field_address_map[node]->pressure;
                interp_gas_tem        += weight_magnitude * node_to_field_address_map[node]->temp;
            }

            particles[p].gas_vel           = interp_gas_vel / total_vector_weight;
            particles[p].gas_pressure      = interp_gas_pre / total_scalar_weight;
            particles[p].gas_temperature   = interp_gas_tem / total_scalar_weight;

            // if (particles[p].gas_temperature     != mesh->dummy_gas_tem)              
            //     {printf("ERROR SOLVE SPRAY PARTICLE : Wrong gas_temperature value %f at %lu (cell %lu)\n",          particles[p].gas_temperature,     p, particles[p].cell); exit(1);}
            // if (particles[p].gas_pressure != mesh->dummy_gas_pre)              
            //     {printf("ERROR SOLVE SPRAY PARTICLE : Wrong pres value %f at %lu (cell %lu)\n",          particles[p].gas_pressure, p, particles[p].cell); exit(1);}
            // if (particles[p].gas_vel.x != mesh->dummy_gas_vel.x) 
            //     {printf("ERROR SOLVE SPRAY PARTICLE : Wrong velo value {%.10f y z} at %lu (cell %lu)\n", particles[p].gas_vel.x,    p, particles[p].cell); exit(1);}

        }

        node_to_field_address_map.clear(); // TODO move this? 


        // static uint64_t node_avg = 0;
        static uint64_t timestep_counter = 0;
        timestep_counter++;

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

        if (PARTICLE_SOLVER_DEBUG )  printf("\tRank %d: Running fn: update_particle_positions.\n", mpi_config->rank);
        const uint64_t particles_size  = particles.size();

        uint64_t elements [mesh->num_blocks];
        for (uint64_t i = 0; i < mesh->num_blocks; i++)
            elements[i] = 0;

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
                const uint64_t cell     = particles[p].cell;
                const uint64_t block_id = mesh->get_block_id(particles[p].cell);

                if ( cell_particle_field_map[block_id].count(cell) ) // TODO Resize arrays.
                {
                    const uint64_t index = cell_particle_field_map[block_id][cell];

                    cell_particle_aos[block_id][index].momentum += particles[p].particle_cell_fields.momentum;
                    cell_particle_aos[block_id][index].energy   += particles[p].particle_cell_fields.energy;
                    cell_particle_aos[block_id][index].fuel     += particles[p].particle_cell_fields.fuel;
                }
                else
                {

                    const uint64_t index = cell_particle_field_map[block_id].size();
                    elements[block_id]   = cell_particle_field_map[block_id].size() + 1;

                    resize_cell_particle(elements, NULL, NULL);

                    cell_particle_indexes[block_id][index]   = cell;
                    cell_particle_aos[block_id][index]       = particles[p].particle_cell_fields;

                    cell_particle_field_map[block_id][cell]  = index;

                    #pragma ivdep
                    for (uint64_t n = 0; n < mesh->cell_size; n++)
                    {
                        const uint64_t node_id = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];

                        if (!node_to_field_address_map.count(node_id))
                            node_to_field_address_map[node_id] = nullptr;
                    }
                }
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
        if (PARTICLE_SOLVER_DEBUG )  printf("\tRank %d: Running fn: update_spray_source_terms.\n", mpi_config->rank);
    }

    template<class T> 
    void ParticleSolver<T>::map_source_terms_to_grid()
    {
        if (PARTICLE_SOLVER_DEBUG )  printf("\tRank %d: Running fn: map_source_terms_to_grid.\n", mpi_config->rank);
    }

    template<class T> 
    void ParticleSolver<T>::interpolate_nodal_data()
    {
        performance_logger.my_papi_start();

        if (PARTICLE_SOLVER_DEBUG )  printf("\tRank %d: Running fn: interpolate_data.\n", mpi_config->rank);

        performance_logger.my_papi_stop(performance_logger.interpolation_kernel_event_counts, &performance_logger.interpolation_time);
    }

    template<class T> 
    void ParticleSolver<T>::timestep()
    {
        static int count = 0;
        const int  comms_timestep = 1;

        if (PARTICLE_SOLVER_DEBUG )  printf("Start particle timestep\n");
        if ( (count % 100) == 0 )
        {
            uint64_t particles_in_simulation = particles.size();
            uint64_t total_particles_in_simulation;

            double arr_usage  = ((double)get_array_memory_usage())   / 1.e9;
            double stl_usage  = ((double)get_stl_memory_usage())     / 1.e9 ;
            double mesh_usage = ((double)mesh->get_memory_usage())   / 1.e9 ;
            double arr_usage_total, stl_usage_total, mesh_usage_total;


            MPI_Reduce(&particles_in_simulation, &total_particles_in_simulation, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&arr_usage,               &arr_usage_total,               1, MPI_DOUBLE,   MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&stl_usage,               &stl_usage_total,               1, MPI_DOUBLE,   MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&mesh_usage,              &mesh_usage_total,              1, MPI_DOUBLE,   MPI_SUM, 0, mpi_config->particle_flow_world);


            if ( mpi_config->particle_flow_rank == 0 )
            {
                // printf("Timestep %6d Particle array mem (TOTAL %8.3f GB) (AVG %8.3f GB) STL mem (TOTAL %8.3f GB) (AVG %8.3f GB) Particles (TOTAL %lu) (AVG %lu) \n", count, arr_usage_total,               arr_usage_total               / mpi_config->particle_flow_world_size, 
                //                                                                                                                                                             stl_usage_total,               stl_usage_total               / mpi_config->particle_flow_world_size, 
                //                                                                                                                                                             total_particles_in_simulation, total_particles_in_simulation / mpi_config->particle_flow_world_size);
                printf("Timestep %6d Particle mem (TOTAL %8.3f GB) (AVG %8.3f GB) Particles (TOTAL %lu) (AVG %lu) \n", count, (arr_usage_total + stl_usage_total + mesh_usage_total), (arr_usage_total + stl_usage_total + mesh_usage_total) / mpi_config->particle_flow_world_size, 
                                                                                                                              total_particles_in_simulation,                           total_particles_in_simulation                         / mpi_config->particle_flow_world_size);

            }
        }


        particle_release();

        if (mpi_config->world_size != 1 && (count % comms_timestep) == 0)
            update_flow_field();
        
        solve_spray_equations();

        update_particle_positions();

        logger.avg_particles += (double)particles.size() / (double)num_timesteps;

        count++;

        if (PARTICLE_SOLVER_DEBUG )  printf("Stop particle timestep\n");
    }
}   // namespace minicombust::particles 