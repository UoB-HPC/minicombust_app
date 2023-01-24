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
        double total_cells_recieved;
        if (mpi_config->particle_flow_rank == 0)
            MPI_Recv(&total_cells_recieved, 1, MPI_DOUBLE, mpi_config->particle_flow_world_size, 0, mpi_config->world, MPI_STATUS_IGNORE);

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
            cout << "\tDuplicated cells across particle ranks:      " << round(10000.*(1 - total_cells_recieved / (mpi_config->particle_flow_world_size * logger.sent_cells))) / 100. << "% " << endl;
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

        uint64_t elements[mesh->num_blocks];
        uint64_t block_world_size[mesh->num_blocks];
        double avg_sent_cells  = 0.;
        double non_zero_blocks = 0.;
        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            cell_particle_field_map[b].erase(MESH_BOUNDARY);
            logger.sent_cells     += cell_particle_field_map[b].size();
            avg_sent_cells        += cell_particle_field_map[b].size();
            non_zero_blocks       += cell_particle_field_map[b].size() > 0;

            neighbours_size[b]         = cell_particle_field_map[b].size();
            block_world_size[b] = cell_particle_field_map[b].size() ? 1 : 0;
            // MPI_Iallreduce(MPI_IN_PLACE, &block_world_size[b], 1, MPI_INT, MPI_SUM, mpi_config->every_one_flow_world[b], &requests[3 * b]);
        }
        avg_sent_cells /= non_zero_blocks;
        logger.sent_cells_per_block += avg_sent_cells;

        

        MPI_Barrier(mpi_config->world);

        // for (uint64_t b = 0; b < mesh->num_blocks; b++)
        // {
        //     MPI_Wait(&requests[3 * b], MPI_STATUS_IGNORE);

        //     if ( block_world_size[b] > 1 )
        //     {
        //         if ( neighbours_size[b] )
        //         {
        //             // MPI_Comm_split(mpi_config->every_one_flow_world[b], 1, mpi_config->rank, &mpi_config->one_flow_world[b]); 
        //             // MPI_Comm_rank(mpi_config->one_flow_world[b], &mpi_config->one_flow_rank[b]);
        //             // MPI_Comm_size(mpi_config->one_flow_world[b], &mpi_config->one_flow_world_size[b]);
        //         }
        //         else
        //         {
        //             // MPI_Comm_split(mpi_config->every_one_flow_world[b], MPI_UNDEFINED, mpi_config->rank, &mpi_config->one_flow_world[b]); 
        //             mpi_config->one_flow_world_size[b] = 0; 
        //         }
        //     }
        // }

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            neighbours_size[b] = cell_particle_field_map[b].size();

            if ( neighbours_size[b] )
            {
                MPI_Issend(&neighbours_size[b], 1, MPI_UINT64_T, mpi_config->particle_flow_world_size + b, 0, mpi_config->world, &requests[b] );
                // printf("Particle Rank %d sending to flow block %d\n", mpi_config->rank, mpi_config->particle_flow_world_size + (int)b);
            }
        }

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            if ( neighbours_size[b] )
                MPI_Wait(&requests[b], MPI_STATUS_IGNORE);
        }

        // printf("Particle Rank %d hit barrier \n", mpi_config->rank);
        MPI_Barrier(mpi_config->particle_flow_world);

        bool sends_done = 1;
        MPI_Ibcast(&sends_done, 1, MPI_INT, 0, mpi_config->world, &requests[0]);

        MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
        MPI_Barrier(mpi_config->world);

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            if ( neighbours_size[b] )
            {
                MPI_Isend(cell_particle_indexes[b],  neighbours_size[b], MPI_UINT64_T, mpi_config->particle_flow_world_size + b, 1, mpi_config->world, &requests[3*b + 1] );

                // for (uint64_t i = 0; i < neighbours_size[b]; i++)
                // {
                //     const uint64_t cell = cell_particle_indexes[b][i];
                //     printf("Particle rank %d is sending cell %lu to flow block %d\n", mpi_config->rank, cell, mpi_config->particle_flow_world_size + b);
                // }

                MPI_Isend(cell_particle_aos[b],     neighbours_size[b], mpi_config->MPI_PARTICLE_STRUCTURE, mpi_config->particle_flow_world_size + b, 2, mpi_config->world, &requests[3*b + 2] );
            }
        }

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            if ( neighbours_size[b] )
                MPI_Waitall( 2, &requests[3*b + 1], MPI_STATUSES_IGNORE );
        }

        // function<void(uint64_t *, uint64_t ***, particle_aos<T> ***)> resize_cell_particles_fn = [this] (uint64_t *elements, uint64_t ***indexes, particle_aos<T> ***cell_particle_fields) { return resize_cell_particle(elements, indexes, cell_particle_fields); };
        // MPI_GatherMap (mpi_config, mesh->num_blocks, cell_particle_field_map, cell_particle_indexes, cell_particle_aos, elements, async_locks, send_counts, recv_indexes, recv_indexed_fields, requests, resize_cell_particles_fn);
        MPI_Barrier(mpi_config->world);


        // Get reduced neighbours size
        // for (uint64_t b = 0; b < mesh->num_blocks; b++) // TODO: Wait async
        // {
        //     // if (neighbours_size[b] != 0) MPI_Bcast(&neighbours_size[b], 1, MPI_UINT64_T, mpi_config->one_flow_world_size[b] - 1, mpi_config->one_flow_world[b]);
        //     MPI_Ibcast(&neighbours_size[b], 1, MPI_UINT64_T, mpi_config->every_one_flow_world_size[b] - 1, mpi_config->every_one_flow_world[b], &requests[b]);
        //     // printf("Rank %d block %lu bcast size %lu\n", mpi_config->rank, b, neighbours_size[b]);
        // }


        // MPI_Waitall( mesh->num_blocks, requests, MPI_STATUSES_IGNORE );


        // resize_nodes_arrays(neighbours_size); 
        // for (uint64_t b = 0; b < mesh->num_blocks; b++)
        // {
        //     // if ( neighbours_size[b] != 0 )  
        //     // {
        //     //     MPI_Ibcast(all_interp_node_indexes[b],     neighbours_size[b], MPI_UINT64_T,                   mpi_config->one_flow_world_size[b] - 1, mpi_config->one_flow_world[b], &requests[3 * b + 0]);
        //     //     MPI_Ibcast(all_interp_node_flow_fields[b], neighbours_size[b], mpi_config->MPI_FLOW_STRUCTURE, mpi_config->one_flow_world_size[b] - 1, mpi_config->one_flow_world[b], &requests[3 * b + 1]);
        //     // }
        //     MPI_Ibcast(all_interp_node_indexes[b],     neighbours_size[b], MPI_UINT64_T,                   mpi_config->every_one_flow_world_size[b] - 1, mpi_config->every_one_flow_world[b], &requests[3 * b + 0]);
        //     MPI_Ibcast(all_interp_node_flow_fields[b], neighbours_size[b], mpi_config->MPI_FLOW_STRUCTURE, mpi_config->every_one_flow_world_size[b] - 1, mpi_config->every_one_flow_world[b], &requests[3 * b + 1]);
        // }


        uint64_t bi = mesh->num_blocks - 1;
        bool     all_processed   = true;
        bool    *processed_block = async_locks;
        for (uint64_t bi = 0; bi < mesh->num_blocks; bi++)  
        {
            processed_block[bi] = (neighbours_size[bi] == 0);
            all_processed      &= processed_block[bi];
        }
        while (!all_processed)
        {
            bi = (bi + 1) % mesh->num_blocks;

            if (!processed_block[bi])
            {
                int message_waiting = 0;
                MPI_Iprobe(MPI_ANY_SOURCE, 3 * bi, mpi_config->world, &message_waiting, &statuses[bi]);

                int *local_block_ranks = &block_ranks[bi * mpi_config->particle_flow_world_size];

                if ( message_waiting )
                {
                    int block_tree_rank = -1, block_tree_size = -1;
                    MPI_Get_count( &statuses[bi], MPI_INT, &block_tree_size );
                    MPI_Recv( local_block_ranks, block_tree_size, MPI_INT, statuses[bi].MPI_SOURCE, statuses[bi].MPI_TAG, mpi_config->world, MPI_STATUS_IGNORE );
                    
                    for ( int r = 0; r < block_tree_size; r++ )
                    {
                        if ( local_block_ranks[r] == mpi_config->rank )  
                        {
                            block_tree_rank = r + 1;
                            break;
                        }
                    }
                    // printf("BLOCK %lu : Rank %d block_tree_rank %d \n", bi, mpi_config->rank, block_tree_rank);


                    int start_level = (int)pow(2., ceil(log((double) (block_tree_rank + 1)) / log(2.))); 
                    int max_level   = (int)pow(2., ceil(log((double) (block_tree_size + 1)) / log(2.)));


                    MPI_Probe ( statuses[bi].MPI_SOURCE, 3*bi + 1, mpi_config->world, &statuses[bi] );
                    MPI_Get_count( &statuses[bi], MPI_UINT64_T, (int*)&neighbours_size[bi] );
                    resize_nodes_arrays(neighbours_size);
                    int level_count = 0;

                    MPI_Irecv ( all_interp_node_indexes[bi],     neighbours_size[bi], MPI_UINT64_T,                   statuses[bi].MPI_SOURCE, 3*bi + 1, mpi_config->world, &requests[0] );
                    MPI_Irecv ( all_interp_node_flow_fields[bi], neighbours_size[bi], mpi_config->MPI_FLOW_STRUCTURE, statuses[bi].MPI_SOURCE, 3*bi + 2, mpi_config->world, &requests[1] );

                    cell_particle_field_map[bi].clear();

                    bool recv_field = 0;
                    MPI_Wait(requests, MPI_STATUS_IGNORE);

                    level_count++;
                    
                    // printf("BLOCK %lu : Level %3d rank %3d (%3d) recieved %lu data from rank %3d\n",  bi, start_level/2, mpi_config->rank, block_tree_rank, neighbours_size[bi], statuses[bi].MPI_SOURCE);
                    for ( int level = start_level; level < max_level; level *= 2 )
                    {
                        int next_tree_index = block_tree_rank + level;
                        if ( next_tree_index > block_tree_size)  continue;
                        int next_tree_rank = local_block_ranks[next_tree_index - 1];
                        // printf("BLOCK %lu : Level %3d rank %3d (%3d) is sending data to rank %3d (%3d)\n", bi,  level, mpi_config->rank, block_tree_rank, next_tree_rank, next_tree_index);

                        MPI_Isend ( local_block_ranks,               block_tree_size,     MPI_INT,                        next_tree_rank, 3*bi + 0, mpi_config->world, &requests[3*level_count + 0] );
                        MPI_Isend ( all_interp_node_indexes[bi],     neighbours_size[bi], MPI_UINT64_T,                   next_tree_rank, 3*bi + 1, mpi_config->world, &requests[3*level_count + 1] );
                        
                        if (!recv_field)
                        {
                            MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
                            recv_field = 1;
                        }

                        MPI_Isend ( all_interp_node_flow_fields[bi], neighbours_size[bi], mpi_config->MPI_FLOW_STRUCTURE, next_tree_rank, 3*bi + 2, mpi_config->world, &requests[3*level_count + 2] );
                        level_count++;
                    }

                    for (uint64_t i = 0; i < neighbours_size[bi]; i++)
                    {
                        if (node_to_field_address_map.count(all_interp_node_indexes[bi][i]))
                        {
                            node_to_field_address_map[all_interp_node_indexes[bi][i]] = &all_interp_node_flow_fields[bi][i];
                        }
                    }
                    
                    if (!recv_field)
                    {
                        MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
                        recv_field = 1;
                    }

                    processed_block[bi] = true;
                }
            }
            all_processed = true;
            for (uint64_t bi = 0; bi < mesh->num_blocks; bi++)  
                all_processed &= processed_block[bi];
        }


        // uint64_t b = 0;
        // bool all_true = false;
        // for (uint64_t bi = 0; bi < mesh->num_blocks; bi++)  
        //     processed_block[bi] = (neighbours_size[bi] == 0);

        // while (!all_true) // TODO: If we know block data is here, we can start operating on particles within that block. Potentially even store particles in blocks? 
        // {
        //     int ready = 1;
        //     // if (neighbours_size[b] != 0)
        //     //     MPI_Testall(2, &requests[3 * b + 1], &ready, MPI_STATUSES_IGNORE);

        //     if ( ready && !processed_block[b] )
        //     {
        //         logger.nodes_recieved += neighbours_size[b];
        //         // printf("Rank %3d is processing data\n", mpi_config->rank );
        //         for (uint64_t i = 0; i < neighbours_size[b]; i++)
        //         {
        //             // if ( all_interp_node_indexes[b][i] >= mesh->points_size )
        //             // {
        //             //     printf("Rank %d out of bounds %lu\n", mpi_config->rank, all_interp_node_indexes[b][i]);
        //             //     exit(0);
        //             // }

        //             if (node_to_field_address_map.count(all_interp_node_indexes[b][i]))
        //             {
        //                 node_to_field_address_map[all_interp_node_indexes[b][i]] = &all_interp_node_flow_fields[b][i];
        //             }

        //             // if (all_interp_node_flow_fields[b][i].temp     != mesh->dummy_gas_tem)             
        //             //     {printf("ERROR RECV VALS : Wrong temp value %f at %lu \n",          all_interp_node_flow_fields[b][i].temp,     all_interp_node_indexes[b][i]); exit(1);}
        //             // if (all_interp_node_flow_fields[b][i].pressure != mesh->dummy_gas_pre)              
        //             //     {printf("ERROR RECV VALS : Wrong pres value %f at %lu \n",          all_interp_node_flow_fields[b][i].pressure, all_interp_node_indexes[b][i]); exit(1);}
        //             // if (all_interp_node_flow_fields[b][i].vel.x != mesh->dummy_gas_vel.x) 
        //             //     {printf("ERROR RECV VALS : Wrong velo value {%.10f y z} at %lu \n", all_interp_node_flow_fields[b][i].vel.x,    all_interp_node_indexes[b][i]); exit(1);}
        //         }
        //         processed_block[b] = true;
        //     }

        //     all_true = true;
        //     for (uint64_t bi = 0; bi < mesh->num_blocks; bi++)  
        //         all_true &= processed_block[bi];
        //     b = (b + 1) % mesh->num_blocks;
        // }

        // printf("Rank %3d is done!\n", mpi_config->rank );

        
        logger.useful_nodes_proportion += node_to_field_address_map.size();

        
        MPI_Barrier(mpi_config->world);
        // for (uint64_t b = 0; b < mesh->num_blocks; b++)
        // {
        //     if ( neighbours_size[b] ) MPI_Comm_free(&mpi_config->one_flow_world[b]);
        // }

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