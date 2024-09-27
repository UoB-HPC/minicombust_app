#include <stdio.h>
#include "particles/ParticleSolver.hpp"
#include "visit/VisitWriter.hpp"

#ifdef have_gpu
    #include <nvToolsExt.h>
    void nvtxpop()
    {
        nvtxRangePop();
    }
    void nvtxpush(const char *message)
    {
        nvtxRangePush(message);
    }
#else
    #define UNUSED(x) (void)(x)
    void nvtxpop()
    {
    }
    void nvtxpush(const char *message)
    {
        UNUSED(message);
    }
#endif

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
        vtk_writer->write_particles("out/minicombust", timestep, particles);
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

        if (mpi_config->particle_flow_rank == 0 && output_file == stdout)
        {
            fprintf(output_file, "Particle Solver Stats:                         \n");
            fprintf(output_file, "\tParticles:                                   %e\n", ((double)logger.num_particles));
            fprintf(output_file, "\tParticles (per iter):                        %lu\n", particle_dist->even_particles_per_timestep*mpi_config->particle_flow_world_size);
            fprintf(output_file, "\tEmitted Particles:                           %lu\n", logger.emitted_particles);
            fprintf(output_file, "\tAvg Particles (per iter):                    %e\n", logger.avg_particles);
            fprintf(output_file, "\n");
            fprintf(output_file, "\tCell checks:                                 %e\n", ((double)logger.cell_checks));
            fprintf(output_file, "\tCell checks (per iter):                      %e\n", ((double)logger.cell_checks) / timesteps);
            fprintf(output_file, "\tCell checks (per particle, per iter):        %.2f\n", ((double)logger.cell_checks) / (((double)logger.num_particles)*timesteps));
            fprintf(output_file, "\n");
            fprintf(output_file, "\tEdge adjustments:                            %.2f\n", ((double)logger.position_adjustments));
            fprintf(output_file, "\tEdge adjustments (per iter):                 %.2f\n", ((double)logger.position_adjustments) / timesteps);
            fprintf(output_file, "\tEdge adjustments (per particle, per iter):   %.2f\n", ((double)logger.position_adjustments) / (((double)logger.num_particles)*timesteps));
            fprintf(output_file, "\tLost Particles:                              %.2f\n", ((double)logger.lost_particles      ));
            fprintf(output_file, "\n");
            fprintf(output_file, "\tBoundary Intersections:                      %.2f\n", ((double)logger.boundary_intersections));
            fprintf(output_file, "\tDecayed Particles:                           %e\n", ((double)logger.decayed_particles));
            fprintf(output_file, "\tDecayed Particles:                           %.2f%%\n", round(10000.*(((double)logger.decayed_particles) / ((double)logger.num_particles)))/100.);
            fprintf(output_file, "\tBurnt Particles:                             %e\n", ((double)logger.burnt_particles));
            fprintf(output_file, "\tBreakups:                                    %.2f\n", ((double)logger.breakups));
            fprintf(output_file, "\tBreakup Age:                                 %.2f\n", ((double)logger.breakup_age));
            fprintf(output_file, "\n");
            fprintf(output_file, "\tAvg Sent Cells       (avg per rank, block):  %.0f\n", round(logger.sent_cells_per_block / timesteps));
            fprintf(output_file, "\tTotal Sent Cells     (avg per rank):         %.0f\n", round(logger.sent_cells / timesteps));
            fprintf(output_file, "\tTotal Recieved Nodes (avg per rank):         %.0f\n", round(logger.nodes_recieved / timesteps));
            fprintf(output_file, "\tUseful Nodes         (avg per rank):         %.0f\n", round(logger.useful_nodes_proportion / timesteps));
            fprintf(output_file, "\tUseful Nodes (%%)     (avg per rank):         %.2f%%\n", round(10000.*((logger.useful_nodes_proportion) / (logger.nodes_recieved))) / 100.);
        }

        performance_logger.print_counters(mpi_config->rank, mpi_config->world_size, runtime);
    }


    template<class T> 
    void ParticleSolver<T>::update_flow_field()
    {
        performance_logger.my_papi_start();

        nvtxpush("Particle update_flow_field");

        active_blocks.clear();

        double avg_sent_cells  = 0.;
        double non_zero_blocks = 0.;

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            cell_particle_field_map[b].erase(MESH_BOUNDARY);
            uint64_t cell_size = cell_particle_field_map[b].size();

            logger.sent_cells += cell_size;
            avg_sent_cells    += cell_size;
            non_zero_blocks   += cell_size > 0;

            neighbours_size[b]   = cell_size;
            
            if ( cell_size )
            {
                active_blocks.push_back(b);
                if (statuses.size() < active_blocks.size())
                {
                    MPI_Status mpi_status = { 0, 0, 0, 0, 0};
                    statuses.push_back( mpi_status );

                    send_requests.push_back( MPI_REQUEST_NULL );
                    send_requests.push_back( MPI_REQUEST_NULL );
                    recv_requests.push_back( MPI_REQUEST_NULL );
                    recv_requests.push_back( MPI_REQUEST_NULL );
                }
            }
        }
        avg_sent_cells              /= non_zero_blocks;
        logger.sent_cells_per_block += avg_sent_cells;


        if ( PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Running fn: update_flow_field.\n", mpi_config->rank);
        if ( PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Sending index sizes.\n", mpi_config->rank);

		// MPI_Barrier(mpi_config->world);

        uint64_t count = 0;
        for (uint64_t b : active_blocks)
        {
            // b = active_blocks[(mpi_config->rank + count) % active_blocks.size()];
            neighbours_size[b] = cell_particle_field_map[b].size();
            if ( PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )
			{  
				fprintf(output_file, "\tRank %d: Sending %d indexes to block %lu.\n", mpi_config->rank, neighbours_size[b], b);
			}
            MPI_Issend(cell_particle_indexes[b], neighbours_size[b], MPI_UINT64_T,                        b, 0, mpi_config->world, &send_requests[count++  + 0*active_blocks.size()] );
	
            // MPI_Isend(cell_particle_aos[b],     neighbours_size[b], mpi_config->MPI_PARTICLE_STRUCTURE,  mpi_config->particle_flow_world_size + b, 2, mpi_config->world, &send_requests[count++ + 1*active_blocks.size()] );
        }
        MPI_Waitall(active_blocks.size(), send_requests.data(), MPI_STATUSES_IGNORE);


        count = 0;
        for (uint64_t b : active_blocks)
        {
            neighbours_size[b] = cell_particle_field_map[b].size();
            if ( PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )
			{  
				fprintf(output_file, "\tRank %d: Sending %d indexes to block %lu.\n", mpi_config->rank, neighbours_size[b], b);
			}
	
            // MPI_Issend(cell_particle_indexes[b], neighbours_size[b], MPI_UINT64_T,                        mpi_config->particle_flow_world_size + b, 0, mpi_config->world, &send_requests[count  + 0*active_blocks.size()] );
            MPI_Isend(cell_particle_aos[b],     neighbours_size[b], mpi_config->MPI_PARTICLE_STRUCTURE,  b, 2, mpi_config->world, &send_requests[count++ + 1*active_blocks.size()] );
        }


        // MPI_Waitall(active_blocks.size(), send_requests.data(), MPI_STATUSES_IGNORE);

        if ( PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Wait has returned successfully\n", mpi_config->rank);
        MPI_Barrier(mpi_config->particle_flow_world);

        int sends_done = 1;
        if ( PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: All index sizes sent.\n", mpi_config->rank);
        
        MPI_Ibcast(&sends_done, 1, MPI_INT, (mpi_config->world_size - mpi_config->particle_flow_world_size), mpi_config->world, &bcast_request);

        for (uint64_t b : active_blocks)
            cell_particle_field_map[b].clear();

        uint64_t posted_recvs = 0;
        uint64_t ba = active_blocks.size() - 1;
        uint64_t bi;
        bool     all_processed      = true;
        bool    *posted_block_recvs = async_locks;
        bool    *processed_block    = async_locks + active_blocks.size();
        for ( uint64_t ba = 0; ba < active_blocks.size(); ba++ )  
        {
            posted_block_recvs[ba]  = false;
            processed_block[ba]     = false;
            all_processed       &= processed_block[ba];
        }
        
        while (!all_processed)
        {
            ba = (ba + 1) % active_blocks.size();
            bi = active_blocks[ba];

            int message_waiting = 0;
            MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[posted_recvs] );

            if ( message_waiting && (posted_recvs < active_blocks.size()) )
            {
                const uint64_t send_rank = statuses[posted_recvs].MPI_SOURCE;
                const uint64_t block_id  = statuses[posted_recvs].MPI_SOURCE;

                MPI_Get_count( &statuses[posted_recvs], MPI_UINT64_T, &neighbours_size[block_id] );
                resize_nodes_arrays(neighbours_size[block_id] + 1, block_id);

                logger.nodes_recieved += neighbours_size[block_id];
                int active_block_index = find(active_blocks.begin(), active_blocks.end(), block_id) - active_blocks.begin(); 

                // if (mpi_config->rank == 0) printf( "Recving %d node values from %lu. Particles size %lu\n", neighbours_size[block_id], send_rank, particles.size() );
                if ( PARTICLE_SOLVER_DEBUG )  fprintf(output_file, "\tRank %d: Posted %d recieves (ptr %p) for flow block %lu (slots %d %ld max %ld) .\n", mpi_config->rank, neighbours_size[block_id], all_interp_node_indexes[block_id], block_id, active_block_index, active_block_index + active_blocks.size(), recv_requests.size() );
                MPI_Irecv ( all_interp_node_indexes[block_id],     neighbours_size[block_id], MPI_UINT64_T,                   send_rank, 0, mpi_config->world, &recv_requests[active_block_index] );
                MPI_Irecv ( all_interp_node_flow_fields[block_id], neighbours_size[block_id], mpi_config->MPI_FLOW_STRUCTURE, send_rank, 1, mpi_config->world, &recv_requests[active_block_index + active_blocks.size()] );

                posted_block_recvs[active_block_index] = true;
                posted_recvs++;
                continue;
            }

            int recieve_done = 0;
            int recieve_fld_done = 0;
            MPI_Test(&recv_requests[ba],                        &recieve_done,     MPI_STATUS_IGNORE);
            MPI_Test(&recv_requests[ba + active_blocks.size()], &recieve_fld_done, MPI_STATUS_IGNORE);

            if ( recieve_done && recieve_fld_done && !processed_block[ba] && posted_block_recvs[ba] )
            {
                #pragma ivdep
                for (int i = 0; i < neighbours_size[bi]; i++)
                {
                    node_to_field_address_map[all_interp_node_indexes[bi][i]] = &all_interp_node_flow_fields[bi][i];
                }
                    
                processed_block[ba] = true;
            }

            all_processed = true;
            for (uint64_t b = 0; b < active_blocks.size(); b++ )  all_processed &= processed_block[b];
        }

        MPI_Waitall( recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);

        nvtxpop();


        logger.useful_nodes_proportion += node_to_field_address_map.size();
        
        // MPI_Barrier(mpi_config->world);
        if ( PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )
		{
			fprintf(output_file, "\tRank %d: Completed comms.\n", mpi_config->rank);
        }

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
    }
            
    template<class T> 
    void ParticleSolver<T>::particle_release()
    {
        performance_logger.my_papi_start();

        // TODO: Reuse decaying particle space
        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  
		{
			fprintf(output_file, "\tRank %d: Running fn: particle_release.\n", mpi_config->rank);
        }
		function<void(uint64_t *, uint64_t ***, particle_aos<T> ***)> resize_cell_particles_fn = [this] (uint64_t *elements, uint64_t ***indexes, particle_aos<T> ***cell_particle_fields) { return resize_cell_particle(elements, indexes, cell_particle_fields); };

        particle_dist->emit_particles_evenly(particles, cell_particle_field_map, node_to_field_address_map, cell_particle_indexes, cell_particle_aos, resize_cell_particles_fn, &logger);
        // particle_dist->emit_particles_waves(particles, cell_particle_field_map, cell_particle_indexes, cell_particle_aos,  &logger);

        performance_logger.my_papi_stop(performance_logger.emit_event_counts, &performance_logger.emit_time);
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Running fn: solve_spray_equations.\n", mpi_config->rank);

        const uint64_t cell_size       = mesh->cell_size; 
        const uint64_t particles_size  = particles.size(); 

        performance_logger.my_papi_start();
        nvtxpush("solve_spray::interpolation");
        
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
                if (PARTICLE_SOLVER_DEBUG && (particles[p].cell >= mesh->mesh_size))
                    {fprintf(output_file, "ERROR::: RANK %d Cell %lu out of range\n", mpi_config->rank, particles[p].cell); exit(1);}

                
                
                uint64_t node = mesh->cells[(particles[p].cell - mesh->shmem_cell_disp) * cell_size + n];
                const uint64_t block_id = mesh->get_block_id(particles[p].cell);

                if (PARTICLE_SOLVER_DEBUG && (node >= mesh->points_size))
                    {fprintf(output_file, "ERROR::: RANK %d Node %lu out of range\n", mpi_config->rank, node); exit(1);}
                if (PARTICLE_SOLVER_DEBUG && (node_to_field_address_map[node] < (flow_aos<T> *)5))
                    {fprintf(output_file, "Rank %d Block %lu cell %lu node %lu flow_pointer %p block_flow_pointer %p size %lu\n", mpi_config->rank, block_id, particles[p].cell, node, node_to_field_address_map[node], all_interp_node_flow_fields[block_id], node_flow_array_sizes[block_id] ); exit(1);};


                const vec<T> node_to_particle = particles[p].x1 - mesh->points[node - mesh->shmem_point_disp];

                vec<T> weight      = 1.0 / ((node_to_particle * node_to_particle) + vec<T> {__DBL_MIN__, __DBL_MIN__, __DBL_MIN__});
                T weight_magnitude = magnitude(weight);


                total_vector_weight   += weight;
                total_scalar_weight   += weight_magnitude;

				// if (PARTICLE_SOLVER_DEBUG) check_flow_field_exit ( "SOLVE SPRAY: Node value", node_to_field_address_map[node], &mesh->dummy_flow_field, node );
                interp_gas_vel        += weight           * node_to_field_address_map[node]->vel;
                interp_gas_pre        += weight_magnitude * node_to_field_address_map[node]->pressure;
                interp_gas_tem        += weight_magnitude * node_to_field_address_map[node]->temp;
            }

            particles[p].local_flow_value.vel           = interp_gas_vel / total_vector_weight;
            particles[p].local_flow_value.pressure      = interp_gas_pre / total_scalar_weight;
            particles[p].local_flow_value.temp          = interp_gas_tem / total_scalar_weight;

			// if (PARTICLE_SOLVER_DEBUG) check_flow_field_exit ( "SOLVE SPRAY: Interpolated particle value ", &particles[p].local_flow_value, &mesh->dummy_flow_field, p );
        }

        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Finished interpolation. Starting spray computation.\n", mpi_config->rank);

        node_to_field_address_map.clear(); // TODO move this? 

        // static uint64_t node_avg = 0;
        static uint64_t timestep_counter = 0;
        timestep_counter++;

        performance_logger.my_papi_stop(performance_logger.particle_interpolation_event_counts, &performance_logger.particle_interpolation_time);
        performance_logger.my_papi_start();

        nvtxpop();
        nvtxpush("solve_spray::main");


        vector<uint64_t> decayed_particles;
        #pragma ivdep
        for (uint64_t p = 0; p < particles_size; p++)
        {
            particles[p].solve_spray( delta, &logger, particles );

            if (particles[p].decayed){
				decayed_particles.push_back(p);
			}
        }

        nvtxpop();
        nvtxpush("solve_spray::decayed");


        const uint64_t decayed_particles_size = decayed_particles.size();
		#pragma ivdep
        for (int128_t i = decayed_particles_size - 1; i >= 0; i--)
        {
            particles[decayed_particles[i]] = particles.back();
            particles.pop_back();
        }
        nvtxpop();


        performance_logger.my_papi_stop(performance_logger.spray_kernel_event_counts, &performance_logger.spray_time);
    }

    template<class T> 
    void ParticleSolver<T>::update_particle_positions()
    {
        performance_logger.my_papi_start();

        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Running fn: update_particle_positions.\n", mpi_config->rank);
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

            if (particles[p].decayed)
			{
				decayed_particles.push_back(p);
            }
			else
            {
                const uint64_t cell     = particles[p].cell;
                const uint64_t block_id = mesh->get_block_id(particles[p].cell);

                if ( cell_particle_field_map[block_id].count(cell) ) 
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


                    // #pragma ivdep
                    // for (uint64_t n = 0; n < mesh->cell_size; n++)
                    // {
                    //     const uint64_t node_id = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];

                    //     if (!node_to_field_address_map.count(node_id))
                    //     {
                    //         node_to_field_address_map[node_id] = (flow_aos<T> *)2;
                    //     }
                    // }

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
        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Running fn: update_spray_source_terms.\n", mpi_config->rank);
    }

    template<class T> 
    void ParticleSolver<T>::map_source_terms_to_grid()
    {
        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Running fn: map_source_terms_to_grid.\n", mpi_config->rank);
    }

    template<class T> 
    void ParticleSolver<T>::interpolate_nodal_data()
    {
        performance_logger.my_papi_start();

        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "\tRank %d: Running fn: interpolate_data.\n", mpi_config->rank);

        performance_logger.my_papi_stop(performance_logger.interpolation_kernel_event_counts, &performance_logger.interpolation_time);
    }

    template<class T> 
    void ParticleSolver<T>::timestep()
    {
		nvtxpush(__FUNCTION__);

        static int count = 0;
        const int  comms_timestep = 1;

        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "Rank %d: Start particle timestep\n", mpi_config->rank);
        if ( ((count + 1) % 100) == 0 )
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
                fprintf(output_file, "Timestep %6d Particle mem (TOTAL %8.3f GB) (AVG %8.3f GB) Particles (TOTAL %lu) (AVG %lu) \n", count + 1, (arr_usage_total + stl_usage_total + mesh_usage_total), (arr_usage_total + stl_usage_total + mesh_usage_total) / mpi_config->particle_flow_world_size, 
                                                                                                                              total_particles_in_simulation,                           total_particles_in_simulation                         / mpi_config->particle_flow_world_size);
            }
        }

		particle_timing[0] -= MPI_Wtime();
		compute_time -= MPI_Wtime();
		nvtxpush("particle_release");
        particle_release();
		nvtxpop();
		compute_time += MPI_Wtime();
		particle_timing[0] += MPI_Wtime();

		particle_timing[1] -= MPI_Wtime();
        if (mpi_config->world_size != 1 && (count % comms_timestep) == 0)
        {
            update_flow_field();
        }
		particle_timing[1] += MPI_Wtime();        

		particle_timing[2] -= MPI_Wtime();
		compute_time -= MPI_Wtime();
        nvtxpush("solve_spray_equations");
        solve_spray_equations();
        nvtxpop();
		particle_timing[2] += MPI_Wtime();

		particle_timing[3] -= MPI_Wtime();
        nvtxpush("update_particle_positions");
        update_particle_positions();
        nvtxpop();
		compute_time += MPI_Wtime();
		particle_timing[3] += MPI_Wtime();
        logger.avg_particles += (double)particles.size() / (double)num_timesteps;

		if(count + 1 == 100)
        {
            if(mpi_config->particle_flow_rank == 0)
            {
                MPI_Reduce(MPI_IN_PLACE, particle_timing, 4, MPI_DOUBLE, MPI_SUM,
                           0, mpi_config->particle_flow_world);
                for(int i = 0; i < 4; i++)
                {
                    particle_timing[i] /= mpi_config->particle_flow_world_size;
                }
                fprintf(output_file, "\nParticle Timing: \nRelease: %f\nCalc update particles: %f\nSolve Spray: %f\nUpdate: %f\n",particle_timing[0],particle_timing[1],particle_timing[2],particle_timing[3]);
            }
            else
            {
                MPI_Reduce(particle_timing, nullptr, 4, MPI_DOUBLE, MPI_SUM,
                           0, mpi_config->particle_flow_world);
            }
        }

        count++;
		nvtxpop();

        if (PARTICLE_SOLVER_DEBUG && mpi_config->rank == mpi_config->particle_flow_rank )  fprintf(output_file, "Rank %d: Stop particle timestep\n", mpi_config->rank);
    }
}   // namespace minicombust::particles 
