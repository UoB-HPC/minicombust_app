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
        performance_logger.print_counters(mpi_config->rank, mpi_config->world_size, runtime);
    }


    template<class T> 
    void ParticleSolver<T>::update_flow_field(bool send_particle)
    {
        performance_logger.my_papi_start();

        if (PARTICLE_SOLVER_DEBUG )  printf("\tRunning fn: update_flow_field.\n");
        // printf("Rank %d Update flow\n", mpi_config->rank);

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            for (auto& cell_it: cell_particle_field_map[b])
            {
                uint64_t cell = cell_it.first;

                const uint64_t block_id = mesh->get_block_id(cell);

                neighbours_sets[mesh->get_block_id(cell)].insert(cell); 

                // Get 6 immediate neighbours
                const uint64_t below_neighbour                = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + DOWN_FACE];
                const uint64_t above_neighbour                = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + UP_FACE];
                const uint64_t around_left_neighbour          = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + LEFT_FACE];
                const uint64_t around_right_neighbour         = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + RIGHT_FACE];
                const uint64_t around_front_neighbour         = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + FRONT_FACE];
                const uint64_t around_back_neighbour          = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + BACK_FACE];

                neighbours_sets[mesh->get_block_id(below_neighbour)].insert(below_neighbour);                
                neighbours_sets[mesh->get_block_id(above_neighbour)].insert(above_neighbour);               
                neighbours_sets[mesh->get_block_id(around_left_neighbour)].insert(around_left_neighbour);          
                neighbours_sets[mesh->get_block_id(around_right_neighbour)].insert(around_right_neighbour);         
                neighbours_sets[mesh->get_block_id(around_front_neighbour)].insert(around_front_neighbour);         
                neighbours_sets[mesh->get_block_id(around_back_neighbour)].insert(around_back_neighbour);          

                // Get 8 cells neighbours around
                if (around_left_neighbour != MESH_BOUNDARY)   
                {
                    const uint64_t around_left_front_neighbour    = mesh->cell_neighbours[around_left_neighbour*mesh->faces_per_cell  + FRONT_FACE];
                    const uint64_t around_left_back_neighbour     = mesh->cell_neighbours[around_left_neighbour*mesh->faces_per_cell  + BACK_FACE];
                    neighbours_sets[mesh->get_block_id(around_left_front_neighbour)].insert(around_left_front_neighbour);    
                    neighbours_sets[mesh->get_block_id(around_left_back_neighbour)].insert(around_left_back_neighbour);     
                }
                if (around_right_neighbour != MESH_BOUNDARY)
                {
                    const uint64_t around_right_front_neighbour   = mesh->cell_neighbours[around_right_neighbour*mesh->faces_per_cell + FRONT_FACE];
                    const uint64_t around_right_back_neighbour    = mesh->cell_neighbours[around_right_neighbour*mesh->faces_per_cell + BACK_FACE];
                    neighbours_sets[mesh->get_block_id(around_right_front_neighbour)].insert(around_right_front_neighbour);   
                    neighbours_sets[mesh->get_block_id(around_right_back_neighbour)].insert(around_right_back_neighbour); 
                }
                if (below_neighbour != MESH_BOUNDARY)
                {
                    // Get 8 cells around below cell
                    const uint64_t below_left_neighbour           = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + LEFT_FACE];
                    const uint64_t below_right_neighbour          = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + RIGHT_FACE];
                    const uint64_t below_front_neighbour          = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + FRONT_FACE];
                    const uint64_t below_back_neighbour           = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + BACK_FACE];
                    neighbours_sets[mesh->get_block_id(below_left_neighbour)].insert(below_left_neighbour);           
                    neighbours_sets[mesh->get_block_id(below_right_neighbour)].insert(below_right_neighbour);          
                    neighbours_sets[mesh->get_block_id(below_front_neighbour)].insert(below_front_neighbour);          
                    neighbours_sets[mesh->get_block_id(below_back_neighbour)].insert(below_back_neighbour);           
                    if (below_left_neighbour != MESH_BOUNDARY)
                    {
                        const uint64_t below_left_front_neighbour     = mesh->cell_neighbours[below_left_neighbour*mesh->faces_per_cell   + FRONT_FACE];
                        const uint64_t below_left_back_neighbour      = mesh->cell_neighbours[below_left_neighbour*mesh->faces_per_cell   + BACK_FACE];
                        neighbours_sets[mesh->get_block_id(below_left_front_neighbour)].insert(below_left_front_neighbour);     
                        neighbours_sets[mesh->get_block_id(below_left_back_neighbour)].insert(below_left_back_neighbour);      
                    }
                    if (below_right_neighbour != MESH_BOUNDARY)
                    {
                        const uint64_t below_right_front_neighbour    = mesh->cell_neighbours[below_right_neighbour*mesh->faces_per_cell  + FRONT_FACE];
                        const uint64_t below_right_back_neighbour     = mesh->cell_neighbours[below_right_neighbour*mesh->faces_per_cell  + BACK_FACE];
                        neighbours_sets[mesh->get_block_id(below_right_front_neighbour)].insert(below_right_front_neighbour);    
                        neighbours_sets[mesh->get_block_id(below_right_back_neighbour)].insert(below_right_back_neighbour); 
                    }
                }
                if (above_neighbour != MESH_BOUNDARY)
                {
                    // Get 8 cells neighbours above
                    const uint64_t above_left_neighbour           = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + LEFT_FACE];
                    const uint64_t above_right_neighbour          = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + RIGHT_FACE];
                    const uint64_t above_front_neighbour          = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + FRONT_FACE];
                    const uint64_t above_back_neighbour           = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + BACK_FACE];
                    neighbours_sets[mesh->get_block_id(above_left_neighbour)].insert(above_left_neighbour);           
                    neighbours_sets[mesh->get_block_id(above_right_neighbour)].insert(above_right_neighbour);          
                    neighbours_sets[mesh->get_block_id(above_front_neighbour)].insert(above_front_neighbour);          
                    neighbours_sets[mesh->get_block_id(above_back_neighbour)].insert(above_back_neighbour);           
                    if (above_left_neighbour != MESH_BOUNDARY)
                    {
                        const uint64_t above_left_front_neighbour     = mesh->cell_neighbours[above_left_neighbour*mesh->faces_per_cell   + FRONT_FACE];
                        const uint64_t above_left_back_neighbour      = mesh->cell_neighbours[above_left_neighbour*mesh->faces_per_cell   + BACK_FACE];
                        neighbours_sets[mesh->get_block_id(above_left_front_neighbour)].insert(above_left_front_neighbour);     
                        neighbours_sets[mesh->get_block_id(above_left_back_neighbour)].insert(above_left_back_neighbour);      
                    }
                    if (above_right_neighbour != MESH_BOUNDARY)
                    {
                        const uint64_t above_right_front_neighbour    = mesh->cell_neighbours[above_right_neighbour*mesh->faces_per_cell  + FRONT_FACE];
                        const uint64_t above_right_back_neighbour     = mesh->cell_neighbours[above_right_neighbour*mesh->faces_per_cell  + BACK_FACE];
                        neighbours_sets[mesh->get_block_id(above_right_front_neighbour)].insert(above_right_front_neighbour);    
                        neighbours_sets[mesh->get_block_id(above_right_back_neighbour)].insert(above_right_back_neighbour);     
                    }
                }
            }
        }


        uint64_t elements[mesh->num_blocks];
        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            neighbours_sets[b].erase(MESH_BOUNDARY);
            elements[b] = neighbours_sets[b].size();
        }

        resize_cell_indexes (elements, NULL);

        // printf("Rank %d Resized\n", mpi_config->rank);
        uint64_t count = 0;
        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            count = 0;
            for (auto& cell: neighbours_sets[b])
            {
                cell_indexes[b][count++] = cell;
            }
        }

        // printf("Rank %d compiled neighbours\n", mpi_config->rank);
        MPI_Barrier(mpi_config->world);

        uint64_t block_world_size[mesh->num_blocks];
        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            block_world_size[b] = neighbours_sets[b].size() ? 1 : 0;
            MPI_Iallreduce(MPI_IN_PLACE, &block_world_size[b], 1, MPI_INT, MPI_SUM, mpi_config->every_one_flow_world[b], &scatter_requests[2*b]);
        }

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            neighbours_size[b] = neighbours_sets[b].size();
            MPI_Wait(&scatter_requests[2*b], MPI_STATUS_IGNORE);

            if ( block_world_size[b] > 1 )
            {
                if ( neighbours_size[b] )
                {
                    MPI_Comm_split(mpi_config->every_one_flow_world[b], 1, mpi_config->rank, &mpi_config->one_flow_world[b]); 
                    MPI_Comm_rank(mpi_config->one_flow_world[b], &mpi_config->one_flow_rank[b]);
                    MPI_Comm_size(mpi_config->one_flow_world[b], &mpi_config->one_flow_world_size[b]);
                }
                else
                {
                    MPI_Comm_split(mpi_config->every_one_flow_world[b], MPI_UNDEFINED, mpi_config->rank, &mpi_config->one_flow_world[b]); 
                    mpi_config->one_flow_world_size[b] = 0; 
                }
            }
        }


        MPI_Barrier(mpi_config->world);
        // printf("Rank %d created world\n", mpi_config->rank);

        // Send local neighbours size
        auto resize_cell_indexes_fn = [this] (uint64_t *elements, uint64_t ***indexes) { return resize_cell_indexes(elements, indexes); };

        MPI_GatherSet ( mpi_config, mesh->num_blocks, neighbours_sets, cell_indexes, elements, resize_cell_indexes_fn);

        // Get reduced neighbours size
        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            if (neighbours_size[b] != 0) MPI_Bcast(&neighbours_size[b], 1, MPI_UINT64_T, mpi_config->one_flow_world_size[b] - 1, mpi_config->one_flow_world[b]);
            // printf("Rank %d block %lu bcast size %lu\n", mpi_config->rank, b, neighbours_size[b]);
        }

        // rank_neighbours_size = neighbours_size / mpi_config->particle_flow_world_size;
        // if ((uint64_t)mpi_config->particle_flow_rank < (neighbours_size % (uint64_t)mpi_config->particle_flow_world_size))
        //     rank_neighbours_size++;
        // resize_cell_flow (rank_neighbours_size);

        //Get local portion of neighbour cells and cell fields
        // MPI_Iscatterv(NULL, NULL, NULL, MPI_UINT64_T,                   cell_indexes,       rank_neighbours_size, MPI_UINT64_T,                   one_flow_root_rank, mpi_config->one_flow_world, &scatter_requests[0]);
        // MPI_Iscatterv(NULL, NULL, NULL, mpi_config->MPI_FLOW_STRUCTURE, cell_flow_aos,      rank_neighbours_size, mpi_config->MPI_FLOW_STRUCTURE, one_flow_root_rank, mpi_config->one_flow_world, &scatter_requests[1]);
        // MPI_Iscatterv(NULL, NULL, NULL, mpi_config->MPI_FLOW_STRUCTURE, cell_flow_grad_aos, rank_neighbours_size, mpi_config->MPI_FLOW_STRUCTURE, one_flow_root_rank, mpi_config->one_flow_world, &scatter_requests[2]);

        resize_nodes_arrays(neighbours_size); 
        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            if (neighbours_size[b] != 0)  
            {
                MPI_Ibcast(all_interp_node_indexes[b],     neighbours_size[b], MPI_UINT64_T,                   mpi_config->one_flow_world_size[b] - 1, mpi_config->one_flow_world[b], &scatter_requests[2 * b + 0]);
                MPI_Ibcast(all_interp_node_flow_fields[b], neighbours_size[b], mpi_config->MPI_FLOW_STRUCTURE, mpi_config->one_flow_world_size[b] - 1, mpi_config->one_flow_world[b], &scatter_requests[2 * b + 1]);
            }
        }

        node_to_field_address_map.clear(); // OVERLAPS
        for (uint64_t b = 0; b < mesh->num_blocks; b++)
            neighbours_sets[b].clear();

        // logger.interpolated_cells += ((float) neighbours_size) / ((float)num_timesteps);

        if (send_particle)
        {
            // Write local particle fields to array
            for (uint64_t b = 0; b < mesh->num_blocks; b++)
            {
                cell_particle_field_map[b].erase(MESH_BOUNDARY);
                if (cell_particle_field_map[b].size() == 0 && neighbours_size[b] != 0)
                    cell_particle_field_map[b][MESH_BOUNDARY] = (particle_aos<T>){(vec<T>){0.0, 0.0, 0.0}, 0.0, 0.0};

                elements[b] = cell_particle_field_map[b].size();

            }

            resize_cell_particle(elements, NULL, NULL);
            for (uint64_t b = 0; b < mesh->num_blocks; b++)
            {
                count = 0;
                for (auto& cell_it: cell_particle_field_map[b])
                {
                    cell_particle_indexes[b][count] = cell_it.first;
                    cell_particle_aos[b][count++]   = cell_it.second;
                }
                // printf("Rank %d Gathermap block %lu elements %lu\n", mpi_config->rank, b, elements[b]);
            }
                

            function<void(uint64_t *, uint64_t ***, particle_aos<T> ***)> resize_cell_particles_fn = [this] (uint64_t *elements, uint64_t ***indexes, particle_aos<T> ***cell_particle_fields) { return resize_cell_particle(elements, indexes, cell_particle_fields); };
            MPI_GatherMap (mpi_config, mesh->num_blocks, cell_particle_field_map, cell_particle_indexes, cell_particle_aos, elements, resize_cell_particles_fn);
        }

        

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
            cell_particle_field_map[b].clear();

        uint64_t b = 0;
        bool all_true = true;
        bool processed_block[mesh->num_blocks] = {false};
        for (uint64_t bi = 0; bi < mesh->num_blocks; bi++)  all_true &= processed_block[bi];
        while (!all_true)
        {
            int ready = 1;
            if (neighbours_size[b] != 0)
                MPI_Testall(2, &scatter_requests[2 * b], &ready, MPI_STATUSES_IGNORE);

            if (ready && !processed_block[b])
            {
                for (uint64_t i = 0; i < neighbours_size[b]; i++)
                {
                    node_to_field_address_map[all_interp_node_indexes[b][i]] = &all_interp_node_flow_fields[b][i];
                }
                processed_block[b] = true;
            }

            all_true = true;
            for (uint64_t bi = 0; bi < mesh->num_blocks; bi++)  all_true &= processed_block[bi];
            b = (b + 1) % mesh->num_blocks;
        }

        // for (auto& node_it: node_to_field_address_map)
        // {
        //     if (node_it.second->temp     != mesh->dummy_gas_tem)              
        //         {printf("ERROR UPDATE FLOW: Wrong temp value %f at %lu\n", node_it.second->temp,           node_it.first); exit(1);}
        //     if (node_it.second->pressure != mesh->dummy_gas_pre)              
        //         {printf("ERROR UPDATE FLOW: Wrong pres value %f at %lu\n", node_it.second->pressure,       node_it.first); exit(1);}
        //     if (node_it.second->vel.x != mesh->dummy_gas_vel.x) 
        //         {printf("ERROR UPDATE FLOW: Wrong velo value {%.10f y z} at %lu\n", node_it.second->vel.x, node_it.first); exit(1);}
        // }

        
        MPI_Barrier(mpi_config->world);
        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            if ( neighbours_size[b] ) MPI_Comm_free(&mpi_config->one_flow_world[b]);
        }

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
    }
            
    template<class T> 
    void ParticleSolver<T>::particle_release()
    {
        performance_logger.my_papi_start();

        // TODO: Reuse decaying particle space
        if (PARTICLE_SOLVER_DEBUG )  printf("\tRunning fn: particle_release.\n");
        particle_dist->emit_particles(particles, cell_particle_field_map, &logger);

        performance_logger.my_papi_stop(performance_logger.emit_event_counts, &performance_logger.emit_time);
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
        if (PARTICLE_SOLVER_DEBUG )  printf("\tRunning fn: solve_spray_equations.\n");

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
                const uint64_t node           = mesh->cells[particles[p].cell*cell_size + n]; 
                const vec<T> node_to_particle = particles[p].x1 - mesh->points[mesh->cells[particles[p].cell*cell_size + n]];

                // REMOVE_TEST_num_points.insert(node);

                vec<T> weight      = 1.0 / (node_to_particle * node_to_particle);
                T weight_magnitude = magnitude(weight);

                total_vector_weight   += weight;
                total_scalar_weight   += weight_magnitude;

                // if (node_to_field_address_map[node]->temp     != mesh->dummy_gas_tem)              
                //     {printf("ERROR SOLVE SPRAY: Wrong temp value %f at %lu (cell %lu)\n",          node_to_field_address_map[node]->temp,     node, particles[p].cell); exit(1);}
                // if (node_to_field_address_map[node]->pressure != mesh->dummy_gas_pre)              
                //     {printf("ERROR SOLVE SPRAY: Wrong pres value %f at %lu (cell %lu)\n",          node_to_field_address_map[node]->pressure, node, particles[p].cell); exit(1);}
                // if (node_to_field_address_map[node]->vel.x != mesh->dummy_gas_vel.x) 
                //     {printf("ERROR SOLVE SPRAY: Wrong velo value {%.10f y z} at %lu (cell %lu)\n", node_to_field_address_map[node]->vel.x,    node, particles[p].cell); exit(1);}
                interp_gas_vel        += weight           * node_to_field_address_map[node]->vel;
                interp_gas_pre        += weight_magnitude * node_to_field_address_map[node]->pressure;
                interp_gas_tem        += weight_magnitude * node_to_field_address_map[node]->temp;
            }

            particles[p].gas_vel           = interp_gas_vel / total_vector_weight;
            particles[p].gas_pressure      = interp_gas_pre / total_scalar_weight;
            particles[p].gas_temperature   = interp_gas_tem / total_scalar_weight;
        }

        // static uint64_t node_avg = 0;
        static uint64_t timestep_counter = 0;
        timestep_counter++;
        // node_avg += REMOVE_TEST_num_points.size();
        // if (mpi_config->particle_flow_rank == 0)  printf("INTERP_PART Time %d, nodes_fields_used %d\n", timestep_counter-1, REMOVE_TEST_num_points.size());

        // if (timestep_counter == 1500 && mpi_config->rank == 0)
        // {
        //     printf("NODES INTERPOLATED %f\n", ((double)node_avg) / 1500.);
        // }

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

        if (PARTICLE_SOLVER_DEBUG )  printf("\tRunning fn: update_particle_positions.\n");
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
                const uint64_t block_id = mesh->get_block_id(particles[p].cell);

                cell_particle_field_map[block_id][particles[p].cell].momentum += particles[p].particle_cell_fields.momentum;
                cell_particle_field_map[block_id][particles[p].cell].energy   += particles[p].particle_cell_fields.energy;
                cell_particle_field_map[block_id][particles[p].cell].fuel     += particles[p].particle_cell_fields.fuel;
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
        if (PARTICLE_SOLVER_DEBUG )  printf("\tRunning fn: update_spray_source_terms.\n");
    }

    template<class T> 
    void ParticleSolver<T>::map_source_terms_to_grid()
    {
        if (PARTICLE_SOLVER_DEBUG )  printf("\tRunning fn: map_source_terms_to_grid.\n");
    }

    template<class T> 
    void ParticleSolver<T>::interpolate_nodal_data()
    {
        performance_logger.my_papi_start();

        if (PARTICLE_SOLVER_DEBUG )  printf("\tRunning fn: interpolate_data.\n");

    //     const uint64_t cell_size  = mesh->cell_size; 

    //     const T node_neighbours       = 8; // Cube specific

    //     // Faster if particles in some of the cells
    //     static int time = 0;
    //     int time_count = 0;

    //     time++;

    //     time_stats[time_count]   -= MPI_Wtime(); // 0

    //     uint64_t local_nodes_size = 0;

    //     // Process the allocation of cell fields (NOTE: Imperfect solution near edges. Fix by doing interpolation on flow side.)
    //     #pragma ivdep
    //     for (uint64_t i = 0; i < rank_neighbours_size; i++)
    //     {
    //         const uint64_t c = cell_indexes[i];


    //         const uint64_t *cell             = mesh->cells + c*cell_size;
    //         const vec<T> cell_centre         = mesh->cell_centres[c];

    //         const flow_aos<T> flow_term      = cell_flow_aos[i];      
    //         const flow_aos<T> flow_grad_term = cell_flow_grad_aos[i]; 

    //         // USEFUL ERROR CHECKING!
    //         // if (flow_term.temp     != mesh->dummy_gas_tem) {printf("INTERP NODAL ERROR: Wrong temp value\n"); exit(1);}
    //         // if (flow_term.pressure != mesh->dummy_gas_pre) {printf("INTERP NODAL ERROR: Wrong pres value\n"); exit(1);}
    //         // if (flow_term.vel.x      != mesh->dummy_gas_vel.x) {printf("INTERP NODAL ERROR: Wrong velo value\n"); exit(1);}

    //         // if (flow_grad_term.temp     != 0.)                 {printf("INTERP NODAL ERROR: Wrong temp grad value\n"); exit(1);}
    //         // if (flow_grad_term.pressure != 0.)                 {printf("INTERP NODAL ERROR: Wrong pres grad value\n"); exit(1);}
    //         // if (flow_grad_term.vel.x    != 0.) {printf("INTERP NODAL ERROR: Wrong velo grad value\n"); exit(1);}

    //         #pragma ivdep
    //         for (uint64_t n = 0; n < cell_size; n++)
    //         {
    //             const uint64_t node_id = cell[n];
    //             const vec<T> direction             = mesh->points[node_id] - cell_centre;

    //             if (node_to_position_map.contains(node_id))
    //             {
    //                 all_interp_node_flow_fields[node_to_position_map[node_id]].vel      += (flow_term.vel      + dot_product(flow_grad_term.vel,      direction)) / node_neighbours;
    //                 all_interp_node_flow_fields[node_to_position_map[node_id]].pressure += (flow_term.pressure + dot_product(flow_grad_term.pressure, direction)) / node_neighbours;
    //                 all_interp_node_flow_fields[node_to_position_map[node_id]].temp     += (flow_term.temp     + dot_product(flow_grad_term.temp,     direction)) / node_neighbours;
    //                 // if (node_id == 16)
    //                 //     printf("rank %d node cell %d flow size %f temp %f\n", mpi_config->rank, c, all_interp_node_flow_fields[node_to_position_map[16]].temp , flow_term.temp);
    //             }
    //             else
    //             {
    //                 const T boundary_neighbours = node_neighbours - mesh->cells_per_point[node_id];

    //                 node_to_position_map[node_id] = local_nodes_size;
                    
    //                 flow_aos<T> temp_term;
    //                 temp_term.vel      = ((mesh->dummy_gas_vel * boundary_neighbours) + flow_term.vel      + dot_product(flow_grad_term.vel,      direction)) / node_neighbours;
    //                 temp_term.pressure = ((mesh->dummy_gas_pre * boundary_neighbours) + flow_term.pressure + dot_product(flow_grad_term.pressure, direction)) / node_neighbours;
    //                 temp_term.temp     = ((mesh->dummy_gas_tem * boundary_neighbours) + flow_term.temp     + dot_product(flow_grad_term.temp,     direction)) / node_neighbours;
                    
    //                 all_interp_node_indexes[local_nodes_size]     = node_id;
    //                 all_interp_node_flow_fields[local_nodes_size] = temp_term;

    //                 // if (node_id == 16)
    //                 //     printf("rank %d node cell %d flow size %f boundary_neighbours %f temp %f\n", mpi_config->rank, c, all_interp_node_flow_fields[node_to_position_map[16]].temp , boundary_neighbours, temp_term.temp);

    //                 local_nodes_size++;
    //             }
    //         }
    //     }

    //     // USEFUL FOR ERROR CHECKING
    //     // for (uint64_t i = 0; i < mesh->mesh_size * cell_size; i++ )
    //     // {
    //     //     if (mpi_config->rank == 0 && mesh->cells[i] == 16)
    //     //     {
    //     //         printf("cell found %lu for point %d \n", i / 8, mesh->cells_per_point[16] );
    //     //     }
    //     // }
    //     // if (node_to_position_map.contains(16)) printf("rank %d node flow size %f\n", mpi_config->rank, all_interp_node_flow_fields[node_to_position_map[16]].temp );

    //     time_stats[time_count++] += MPI_Wtime();
    //     time_stats[time_count]   -= MPI_Wtime(); //1
        
    //     static uint64_t node_avg = 0;
    //     node_avg += node_to_position_map.size();


    //     const uint64_t rank = mpi_config->particle_flow_rank;

    //     int max_levels = 1;
    //     while (max_levels < mpi_config->particle_flow_world_size)
    //         max_levels *= 2;

    //     bool have_data = true;
    //     for ( int level = 2; level <= max_levels ; level *= 2)
    //     {

    //         if (have_data)
    //         {
    //             bool reciever = ((rank % level) == 0) ? true : false;
    //             if ( reciever )
    //             {
    //                 uint64_t send_rank = rank + (level / 2);
    //                 if (send_rank >= (uint64_t) mpi_config->particle_flow_world_size) {
    //                     continue;
    //                 }

                
    //                 uint64_t send_count;

    //                 MPI_Recv (&send_count,     1,          MPI_UINT64_T,                   send_rank, level, mpi_config->particle_flow_world, MPI_STATUS_IGNORE);
    //                 // printf("LEVEL %d: Rank %lu recv from %lu size %lu local size %lu\n", level, rank, send_rank, send_count, local_nodes_size);
    //                 resize_nodes_arrays(local_nodes_size + send_count); // RESIZE within comms loop
    //                 uint64_t    *recv_indexes    = all_interp_node_indexes     + local_nodes_size;
    //                 flow_aos<T> *recv_flow_terms = all_interp_node_flow_fields + local_nodes_size;

    //                 MPI_Recv (recv_indexes,    send_count, MPI_UINT64_T,                   send_rank, level, mpi_config->particle_flow_world, MPI_STATUS_IGNORE);
    //                 MPI_Recv (recv_flow_terms, send_count, mpi_config->MPI_FLOW_STRUCTURE, send_rank, level, mpi_config->particle_flow_world, MPI_STATUS_IGNORE);

    //                 for (uint64_t i = 0; i < send_count; i++)
    //                 {
    //                     if ( node_to_position_map.contains(recv_indexes[i]) ) // Aggregate terms
    //                     {
    //                         all_interp_node_flow_fields[node_to_position_map[recv_indexes[i]]].vel      += recv_flow_terms[i].vel;
    //                         all_interp_node_flow_fields[node_to_position_map[recv_indexes[i]]].temp     += recv_flow_terms[i].temp;
    //                         all_interp_node_flow_fields[node_to_position_map[recv_indexes[i]]].pressure += recv_flow_terms[i].pressure;

    //                     }
    //                     else // Create new entry
    //                     { 
    //                         all_interp_node_indexes[local_nodes_size]     = recv_indexes[i];
    //                         all_interp_node_flow_fields[local_nodes_size] = recv_flow_terms[i];
                    
    //                         node_to_position_map[recv_indexes[i]] = local_nodes_size;
    //                         local_nodes_size++;
    //                     }

    //                 }
    //                 // if (node_to_position_map.contains(516076)) printf("LEVEL %d: RANK %d RECIEVED THE PAYLOAD %f %p\n", level, mpi_config->rank, all_interp_node_flow_fields[node_to_position_map[516076]].temp, &all_interp_node_flow_fields[node_to_position_map[516076]]);
    //             }
    //             else
    //             {
    //                 uint64_t recv_rank  = rank - (level / 2);
    //                 // printf("LEVEL %d: Rank %lu send to %lu size %lu\n", level, rank, recv_rank, local_nodes_size);

    //                 MPI_Ssend (&local_nodes_size,           1,                MPI_UINT64_T,                   recv_rank, level, mpi_config->particle_flow_world);
    //                 MPI_Ssend (all_interp_node_indexes,     local_nodes_size, MPI_UINT64_T,                   recv_rank, level, mpi_config->particle_flow_world);
    //                 MPI_Ssend (all_interp_node_flow_fields, local_nodes_size, mpi_config->MPI_FLOW_STRUCTURE, recv_rank, level, mpi_config->particle_flow_world);
                    
    //                 have_data = false;
    //             }
    //         }
    //     }

    //     time_stats[time_count++] += MPI_Wtime();
    //     time_stats[time_count]   -= MPI_Wtime(); //2

    //     MPI_Request requests[2];


    //     MPI_Bcast (&local_nodes_size,           1,                 MPI_UINT64_T,                    0, mpi_config->particle_flow_world);
    //     resize_nodes_arrays(local_nodes_size); 
    //     MPI_Ibcast (all_interp_node_indexes,     local_nodes_size,  MPI_UINT64_T,                   0, mpi_config->particle_flow_world, &requests[0]);
    //     MPI_Ibcast (all_interp_node_flow_fields, local_nodes_size,  mpi_config->MPI_FLOW_STRUCTURE, 0, mpi_config->particle_flow_world, &requests[1]);
    //     MPI_Wait(&requests[0], MPI_STATUS_IGNORE);

    //     for (uint64_t i = 0; i < local_nodes_size; i++)
    //     {
    //         node_to_position_map[all_interp_node_indexes[i]] = i;
    //     }

    //     MPI_Wait(&requests[1], MPI_STATUS_IGNORE);

    //     time_stats[time_count++] += MPI_Wtime();

    //     if ( time == 1500 )
    //     {
    //         if (mpi_config->rank == 0)
    //         {
    //             double total_time = 0.0;
    //             printf("\nInterpolate Nodal Timings:\n");

    //             for (int i = 0; i < time_count; i++)
    //                 total_time += time_stats[i];
    //             for (int i = 0; i < time_count; i++)
    //                 printf("Time stats %d: %f %.2f\n", i, time_stats[i], 100 * time_stats[i] / total_time);
    //             printf("Total time %f\n", total_time);
    //         }
    //     }
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

            double arr_usage = ((double)get_array_memory_usage()) / 1.e9;
            double stl_usage = ((double)get_stl_memory_usage())   / 1.e9 ;
            double arr_usage_total, stl_usage_total;


            MPI_Reduce(&particles_in_simulation, &total_particles_in_simulation, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&arr_usage,               &arr_usage_total,               1, MPI_DOUBLE,   MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&stl_usage,               &stl_usage_total,               1, MPI_DOUBLE,   MPI_SUM, 0, mpi_config->particle_flow_world);


            if ( mpi_config->particle_flow_rank == 0 )
            {
                printf("Timestep %6d Particle Array mem (GB) %8.3f Array mem total (GB) %8.3f STL mem (GB) %8.3f STL mem total (GB) %8.3f Particles %lu\n", count, arr_usage, arr_usage_total, stl_usage, stl_usage_total, total_particles_in_simulation);
            }
        }

        particle_release();
        if (mpi_config->world_size != 1 && (count % comms_timestep) == 0)
        {
            update_flow_field(count > 0);
            // interpolate_nodal_data(); 
        }
        else if (mpi_config->world_size == 1)
        {
            // interpolate_nodal_data(); 
        }
        solve_spray_equations();
        update_particle_positions();

        logger.avg_particles += (double)particles.size() / (double)num_timesteps;

        count++;

        if (PARTICLE_SOLVER_DEBUG )  printf("Stop particle timestep\n");
    }
}   // namespace minicombust::particles 