#include <stdio.h>
#include <limits.h>

#include "flow/FlowSolver.hpp"


using namespace std;

namespace minicombust::flow 
{
    template<typename T> bool FlowSolver<T>::is_halo ( uint64_t cell )
    {
        return ( cell < mesh->block_element_disp[mpi_config->particle_flow_rank] || cell >= mesh->block_element_disp[mpi_config->particle_flow_rank + 1] );
    }

    template<typename T> void FlowSolver<T>::get_neighbour_cells ( const uint64_t recv_id )
    {
        double node_neighbours   = 8;
        const uint64_t cell_size = mesh->cell_size;

        resize_nodes_arrays(node_to_position_map.size() + elements[recv_id] * cell_size + 1 ); // TODO: Move outside loop

        #pragma ivdep
        for (int i = 0; i < elements[recv_id]; i++)
        {
            uint64_t cell = neighbour_indexes[recv_id][i];


            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id      = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];
                local_particle_node_sets[recv_id].insert(node_id);
            }

            if ( new_cells_set.contains(cell) )  continue;

            new_cells_set.insert(cell);
            unordered_neighbours_set[0].insert(cell);   
            

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id      = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];

                if (!node_to_position_map.count(node_id))
                {
                    const T boundary_neighbours = node_neighbours - mesh->cells_per_point[node_id - mesh->shmem_point_disp];

                    flow_aos<T> temp_term;
                    temp_term.vel      = mesh->dummy_gas_vel * (boundary_neighbours / node_neighbours);
                    temp_term.pressure = mesh->dummy_gas_pre * (boundary_neighbours / node_neighbours);
                    temp_term.temp     = mesh->dummy_gas_tem * (boundary_neighbours / node_neighbours);

                    const uint64_t position = node_to_position_map.size();
                    interp_node_indexes[position]     = node_id;
                    interp_node_flow_fields[position] = temp_term; 
                    node_to_position_map[node_id]     = position;
                }
            }

            // Get 6 immediate neighbours
            const uint64_t below_neighbour                = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + DOWN_FACE];
            const uint64_t above_neighbour                = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + UP_FACE];
            const uint64_t around_left_neighbour          = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + LEFT_FACE];
            const uint64_t around_right_neighbour         = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + RIGHT_FACE];
            const uint64_t around_front_neighbour         = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + FRONT_FACE];
            const uint64_t around_back_neighbour          = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + BACK_FACE];

            unordered_neighbours_set[0].insert(below_neighbour);             // Immediate neighbour cell indexes are correct   
            unordered_neighbours_set[0].insert(above_neighbour);             // Immediate neighbour cell indexes are correct  
            unordered_neighbours_set[0].insert(around_left_neighbour);       // Immediate neighbour cell indexes are correct   
            unordered_neighbours_set[0].insert(around_right_neighbour);      // Immediate neighbour cell indexes are correct   
            unordered_neighbours_set[0].insert(around_front_neighbour);      // Immediate neighbour cell indexes are correct   
            unordered_neighbours_set[0].insert(around_back_neighbour);       // Immediate neighbour cell indexes are correct   

            // Get 8 cells neighbours around
            if ( around_left_neighbour != MESH_BOUNDARY  )   // If neighbour isn't edge of mesh and isn't a halo cell
            {
                const uint64_t around_left_front_neighbour    = mesh->cell_neighbours[ (around_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                const uint64_t around_left_back_neighbour     = mesh->cell_neighbours[ (around_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(around_left_front_neighbour);    
                unordered_neighbours_set[0].insert(around_left_back_neighbour);     
            }
            if ( around_right_neighbour != MESH_BOUNDARY )
            {
                const uint64_t around_right_front_neighbour   = mesh->cell_neighbours[ (around_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell + FRONT_FACE] ;
                const uint64_t around_right_back_neighbour    = mesh->cell_neighbours[ (around_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(around_right_front_neighbour);   
                unordered_neighbours_set[0].insert(around_right_back_neighbour); 
            }
            if ( below_neighbour != MESH_BOUNDARY )
            {
                // Get 8 cells around below cell
                const uint64_t below_left_neighbour           = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + LEFT_FACE]  ;
                const uint64_t below_right_neighbour          = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + RIGHT_FACE] ;
                const uint64_t below_front_neighbour          = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + FRONT_FACE] ;
                const uint64_t below_back_neighbour           = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(below_left_neighbour);           
                unordered_neighbours_set[0].insert(below_right_neighbour);          
                unordered_neighbours_set[0].insert(below_front_neighbour);          
                unordered_neighbours_set[0].insert(below_back_neighbour);           
                if ( below_left_neighbour != MESH_BOUNDARY )
                {
                    const uint64_t below_left_front_neighbour     = mesh->cell_neighbours[ (below_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + FRONT_FACE] ;
                    const uint64_t below_left_back_neighbour      = mesh->cell_neighbours[ (below_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(below_left_front_neighbour);     
                    unordered_neighbours_set[0].insert(below_left_back_neighbour);      
                }
                if ( below_right_neighbour != MESH_BOUNDARY )
                {
                    const uint64_t below_right_front_neighbour    = mesh->cell_neighbours[ (below_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                    const uint64_t below_right_back_neighbour     = mesh->cell_neighbours[ (below_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(below_right_front_neighbour);    
                    unordered_neighbours_set[0].insert(below_right_back_neighbour); 
                }
            }
            if ( above_neighbour != MESH_BOUNDARY )
            {
                // Get 8 cells neighbours above
                const uint64_t above_left_neighbour           = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + LEFT_FACE]  ;
                const uint64_t above_right_neighbour          = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + RIGHT_FACE] ;
                const uint64_t above_front_neighbour          = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + FRONT_FACE] ;
                const uint64_t above_back_neighbour           = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(above_left_neighbour);           
                unordered_neighbours_set[0].insert(above_right_neighbour);          
                unordered_neighbours_set[0].insert(above_front_neighbour);          
                unordered_neighbours_set[0].insert(above_back_neighbour);           
                if ( above_left_neighbour != MESH_BOUNDARY )
                {
                    const uint64_t above_left_front_neighbour     = mesh->cell_neighbours[ (above_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + FRONT_FACE] ;
                    const uint64_t above_left_back_neighbour      = mesh->cell_neighbours[ (above_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(above_left_front_neighbour);     
                    unordered_neighbours_set[0].insert(above_left_back_neighbour);      
                }
                if ( above_right_neighbour != MESH_BOUNDARY )
                {
                    const uint64_t above_right_front_neighbour    = mesh->cell_neighbours[ (above_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                    const uint64_t above_right_back_neighbour     = mesh->cell_neighbours[ (above_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(above_right_front_neighbour);    
                    unordered_neighbours_set[0].insert(above_right_back_neighbour);     
                }
            }
        }

        unordered_neighbours_set[0].erase(MESH_BOUNDARY);
    }

    template<typename T> void FlowSolver<T>::interpolate_to_nodes ()
    {
        const uint64_t cell_size = mesh->cell_size;
        double node_neighbours   = 8;

        // Process the allocation of cell fields (NOTE: Imperfect solution near edges. Fix by doing interpolation on flow side.)
        // #pragma ivdep
        for ( auto& cell_it : unordered_neighbours_set[0] )
        {
            const uint64_t block_cell_disp  = mesh->local_cells_disp;
            const uint64_t c                = cell_it;
            const uint64_t displaced_c      = cell_it - block_cell_disp;

            flow_aos<T> flow_term;      
            flow_aos<T> flow_grad_term; 

            if (is_halo(cell_it)) 
            {
                flow_term.temp          = mesh->dummy_gas_tem;      
                flow_grad_term.temp     = 0.0;  

                flow_term.pressure      = mesh->dummy_gas_pre;      
                flow_grad_term.pressure = 0.0; 
                
                flow_term.vel           = mesh->dummy_gas_vel; 
                flow_grad_term.vel      = {0.0, 0.0, 0.0}; 
            }
            else
            {
                flow_term      = mesh->flow_terms[displaced_c];

                flow_term.vel.x = phi.U[displaced_c];
                flow_term.vel.y = phi.V[displaced_c];
                flow_term.vel.z = phi.W[displaced_c];

                printf("flow_term %f %f %f\n", flow_term.vel.x, flow_term.vel.y, flow_term.vel.z);

                flow_grad_term = mesh->flow_grad_terms[displaced_c]; 
            }

            const vec<T> cell_centre         = mesh->cell_centers[c - mesh->shmem_cell_disp];


            if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL ERROR: Flow value",      &flow_term,      &mesh->dummy_flow_field,      c );
            if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL ERROR: Flow grad value", &flow_grad_term, &mesh->dummy_flow_field_grad, c );

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id = mesh->cells[(c - mesh->shmem_cell_disp)*mesh->cell_size + n];

                if (node_to_position_map.count(node_id))
                {
                    const vec<T> direction      = mesh->points[node_id - mesh->shmem_point_disp] - cell_centre;

                    interp_node_flow_fields[node_to_position_map[node_id]].temp     += (flow_term.temp     + dot_product(flow_grad_term.temp,     direction)) / node_neighbours;
                    interp_node_flow_fields[node_to_position_map[node_id]].pressure += (flow_term.pressure + dot_product(flow_grad_term.pressure, direction)) / node_neighbours;
                    interp_node_flow_fields[node_to_position_map[node_id]].vel      += (flow_term.vel      + dot_product(flow_grad_term.vel,      direction)) / node_neighbours;
                }
            }
        }

        // TODO: Can comment this with properly implemented halo exchange. Need cell neighbours for halo and nodes!
        uint64_t const nsize = node_to_position_map.size();

        if (FLOW_SOLVER_DEBUG)
        {
            #pragma ivdep
            for ( uint64_t i = 0;  i < nsize; i++ ) 
            {
                if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL FINAL ERROR: ", &interp_node_flow_fields[i], &mesh->dummy_flow_field, i );
            }
        }
    }

    template<typename T> void FlowSolver<T>::update_flow_field()
    {
        
        int time_count = 0;
        time_stats[time_count]  -= MPI_Wtime(); //0
        unordered_neighbours_set[0].clear();
        node_to_position_map.clear();
        new_cells_set.clear();
        ranks.clear();

        for (uint64_t i = 0; i < local_particle_node_sets.size(); i++)
        {
            local_particle_node_sets[i].clear();
        }
        
        performance_logger.my_papi_start();

        // MPI_Barrier(mpi_config->world);
        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Completed Barrrier.\n", mpi_config->rank);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //1
        static double time0=0., time1=0., time2=0.;
        static double recv_time1=0., recv_time2=0., recv_time3=0.;

        int recvs_complete = 0;

        MPI_Ibcast(&recvs_complete, 1, MPI_INT, 0, mpi_config->world, &bcast_request);
        
        int message_waiting = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);

        bool  first_time           = true ;
        bool  all_processed        = false;
        bool *processed_neighbours = async_locks;
        while(!all_processed)
        {
            time0 -= MPI_Wtime(); //1

            if (first_time)
            {
                // printf("\tFlow Rank %d: Waiting message_waiting %d\n", mpi_config->rank, message_waiting );
                // first_time = false;
            }

            // printf("\tFlow Rank %d: Waiting message_waiting %d\n", mpi_config->rank, message_waiting );
            if ( message_waiting )
            {
                uint64_t rank_slot = ranks.size();
                ranks.push_back(statuses[rank_slot].MPI_SOURCE);
                MPI_Get_count( &statuses[rank_slot], MPI_UINT64_T, &elements[rank_slot] );

                resize_cell_particle(elements[rank_slot], rank_slot);
                if ( FLOW_SOLVER_DEBUG )  printf("\tFlow block %d: Recieving %d indexes from %d (slot %lu). Max element size %lu. neighbour index rank size %ld array_pointer %p \n", mpi_config->particle_flow_rank, elements[rank_slot], ranks.back(), rank_slot, cell_index_array_size[rank_slot] / sizeof(uint64_t), neighbour_indexes.size(), neighbour_indexes[rank_slot]);

                logger.recieved_cells += elements[rank_slot];

                MPI_Irecv(neighbour_indexes[rank_slot], elements[rank_slot], MPI_UINT64_T,                       ranks[rank_slot], 0, mpi_config->world, &recv_requests[2*rank_slot]     );
                MPI_Irecv(cell_particle_aos[rank_slot], elements[rank_slot], mpi_config->MPI_PARTICLE_STRUCTURE, ranks[rank_slot], 2, mpi_config->world, &recv_requests[2*rank_slot + 1] );

                processed_neighbours[rank_slot] = false;

                if ( statuses.size() <= ranks.size() )
                {
                    statuses.push_back (empty_mpi_status);
                    recv_requests.push_back ( MPI_REQUEST_NULL );
                    recv_requests.push_back ( MPI_REQUEST_NULL );
                    send_requests.push_back ( MPI_REQUEST_NULL );
                    send_requests.push_back ( MPI_REQUEST_NULL );

                    cell_index_array_size.push_back(max_storage    * sizeof(uint64_t));
                    cell_particle_array_size.push_back(max_storage * sizeof(particle_aos<T>));

                    neighbour_indexes.push_back((uint64_t*)         malloc(cell_index_array_size.back()));
                    cell_particle_aos.push_back((particle_aos<T> * )malloc(cell_particle_array_size.back()));

                    local_particle_node_sets.push_back(unordered_set<uint64_t>());
                }
                message_waiting = 0;
                MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);
                continue;
             }


            time0 += MPI_Wtime(); //1
            time1 -= MPI_Wtime(); //1
            
            all_processed = true;
            for ( uint64_t p = 0; p < ranks.size(); p++ )
            {
                int recieved_indexes = 0;
                MPI_Test(&recv_requests[2*p], &recieved_indexes, MPI_STATUS_IGNORE);

                if ( recieved_indexes && !processed_neighbours[p] )
                {
                    if ( FLOW_SOLVER_DEBUG )  printf("\tFlow block %d: Processing %d indexes from %d. Local set size %lu (%lu of %lu sets)\n", mpi_config->particle_flow_rank, elements[p], ranks[p], local_particle_node_sets[p].size(), p, local_particle_node_sets.size());
                    
                    get_neighbour_cells (p);
                    processed_neighbours[p] = true;

                }

                all_processed &= processed_neighbours[p];
            }

            time1 += MPI_Wtime(); //1
            time2 -= MPI_Wtime(); //1

            MPI_Test ( &bcast_request, &recvs_complete, MPI_STATUS_IGNORE );
            MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);

            if ( FLOW_SOLVER_DEBUG && recvs_complete )  printf("\tFlow block %d: Recieved broadcast signal. message_waiting %d recvs_complete %d all_processed %d\n", mpi_config->particle_flow_rank, message_waiting, recvs_complete, all_processed);
            // printf("\tFlow block %d: message_waiting %d recvs_complete %d all_processed %d\n", mpi_config->particle_flow_rank, message_waiting, recvs_complete, all_processed);
            
            all_processed = all_processed & !message_waiting & recvs_complete;
            time2 += MPI_Wtime(); //1
        }

        logger.reduced_recieved_cells += new_cells_set.size();

        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Recieved index sizes.\n", mpi_config->rank);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //2


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //3
        

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //4
        
        interpolate_to_nodes ();

        // Send size of reduced neighbours of cells back to ranks.
        uint64_t neighbour_point_size = node_to_position_map.size();

        logger.sent_nodes += neighbour_point_size;

        uint64_t max_send_buffer_size = 0;
        for (uint64_t p = 0; p < ranks.size(); p++)
        {
            max_send_buffer_size += local_particle_node_sets[p].size();
        }
        resize_send_buffers_nodes_arrays (max_send_buffer_size);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //5
        
        
        uint64_t ptr_disp = 0;
        bool *processed_cell_fields = async_locks;
        for (uint64_t p = 0; p < ranks.size(); p++)
        {
            recv_time1  -= MPI_Wtime();
            uint64_t local_disp = 0;
            #pragma ivdep
            for ( uint64_t node : local_particle_node_sets[p] )
            {
                send_buffers_interp_node_indexes[ptr_disp     + local_disp] = interp_node_indexes[node_to_position_map[node]];
                send_buffers_interp_node_flow_fields[ptr_disp + local_disp] = interp_node_flow_fields[node_to_position_map[node]];
                local_disp++;
                
                // if ( send_buffers_interp_node_indexes[ptr_disp + local_disp] > mesh->points_size )
                //     {printf("ERROR SEND VALS : Flow Rank %d Particle %lu Value %lu out of range at %lu\n", mpi_config->rank, ranks[p], send_buffers_interp_node_indexes[ptr_disp + local_disp], local_disp); exit(1);}
            }
            recv_time1  += MPI_Wtime();
            recv_time2  -= MPI_Wtime();
            // printf("Flow Rank %3d is sending %lu data to %d\n", mpi_config->particle_flow_rank, local_disp, ranks[p]);

            MPI_Isend ( send_buffers_interp_node_indexes + ptr_disp,     local_disp, MPI_UINT64_T,                   ranks[p], 0, mpi_config->world, &send_requests[p] );
            MPI_Isend ( send_buffers_interp_node_flow_fields + ptr_disp, local_disp, mpi_config->MPI_FLOW_STRUCTURE, ranks[p], 1, mpi_config->world, &send_requests[p + ranks.size()] );
            ptr_disp += local_disp;

            processed_cell_fields[p] = false;

            recv_time2  += MPI_Wtime();
        }
        
        recv_time3  -= MPI_Wtime();

        if ( FLOW_SOLVER_DEBUG && mpi_config->particle_flow_rank == 0 )  printf("\tFlow Rank %d: Posted sends.\n", mpi_config->rank);

        all_processed = false;
        while ( !all_processed )
        {
            all_processed = true;

            for ( uint64_t p = 0; p < ranks.size(); p++ )
            {
                int recieved_indexes = 0;
                MPI_Test(&recv_requests[2*p + 1], &recieved_indexes, MPI_STATUS_IGNORE);

                if ( recieved_indexes && !processed_neighbours[p] )
                {
                    if ( FLOW_SOLVER_DEBUG )  printf("\tFlow block %d: Processing %d cell fields from %d .\n", mpi_config->particle_flow_rank, elements[p], ranks[p]);

                    for (int i = 0; i < elements[p]; i++)
                    {
                        mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].momentum += cell_particle_aos[p][i].momentum;
                        mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].energy   += cell_particle_aos[p][i].energy;
                        mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].fuel     += cell_particle_aos[p][i].fuel;
                    }

                    processed_cell_fields[p] = true;
                }
                all_processed &= processed_cell_fields[p];
            }
            if ( FLOW_SOLVER_DEBUG && all_processed )  printf("\tFlow block %d: all_processed %d\n", mpi_config->particle_flow_rank, all_processed);
        }
        recv_time3 += MPI_Wtime();

        MPI_Waitall(send_requests.size() - 2, send_requests.data(), MPI_STATUSES_IGNORE); // Check field values later on!

        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Processed cell particle fields .\n", mpi_config->rank);

        // MPI_Barrier(mpi_config->world);
        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //6

        
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //7

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //8

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
        
        time_stats[time_count++] += MPI_Wtime();

        static int timestep_count = 0;
        if (timestep_count++ == 1499)
        {

            // printf("Rank %d Time 0: %.2f\n", mpi_config->rank, time0);
            // printf("Rank %d Time 1: %.2f\n", mpi_config->rank, time1);
            // printf("Rank %d Time 2: %.2f\n", mpi_config->rank, time2);

            // printf("Rank %d RECV Time 1: %.2f\n", mpi_config->rank, recv_time1);
            // printf("Rank %d RECV Time 2: %.2f\n", mpi_config->rank, recv_time2);
            // printf("Rank %d RECV Time 3: %.2f\n", mpi_config->rank, recv_time3);

            if ( mpi_config->particle_flow_rank == 0 )
            {
                for (int i = 0; i < time_count; i++)
                    MPI_Reduce(MPI_IN_PLACE, &time_stats[i], 1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);

                double total_time = 0.0;
                printf("\nUpdate Flow Field Communuication Timings\n");

                for (int i = 0; i < time_count; i++)
                    total_time += time_stats[i];
                for (int i = 0; i < time_count; i++)
                    printf("Time stats %d: %.3f %.2f\n", i, time_stats[i]  / mpi_config->particle_flow_world_size, 100 * time_stats[i] / total_time);
                printf("Total time %f\n", total_time / mpi_config->particle_flow_world_size);

            }
            else{
                for (int i = 0; i < time_count; i++)
                    MPI_Reduce(&time_stats[i], nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);
            }
        }
    } 

    template<typename T> void FlowSolver<T>::get_phi_gradient ( T *phi_component, vec<T> *phi_grad_component )
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function get_phi_gradient.\n", mpi_config->rank);
        // NOTE: Currently Least squares is the only method supported

        Eigen::Matrix3f A;
        Eigen::Vector3f b;
        Eigen::Vector3f x;

        for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
        {
            const uint64_t cell = block_cell + mesh->local_cells_disp;
            // vec<T> center      = mesh->cell_centers[real_cell - mesh->shmem_cell_disp];

            A = Eigen::Matrix3f::Zero();
            b = Eigen::Vector3f::Zero();

            for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
            {
                const uint64_t face_id = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];
                const uint64_t cell0   = mesh->faces[face_id].cell0;
                const uint64_t cell1   = mesh->faces[face_id].cell1;

                if ( cell1 == MESH_BOUNDARY ) continue;

                T dPhi;
                vec<T> dX;
                if ( cell0 == cell )
                {
                    if ( cell1 - mesh->local_cells_disp > mesh->local_mesh_size ) continue;

                    dPhi = phi_component[cell1      - mesh->local_cells_disp] - phi_component[cell0      - mesh->local_cells_disp];
                    dX   = mesh->cell_centers[cell1 - mesh->shmem_cell_disp]  - mesh->cell_centers[cell0 - mesh->shmem_cell_disp];
                }
                else
                {
                    if ( cell0 - mesh->local_cells_disp > mesh->local_mesh_size ) continue;
                    dPhi = phi_component[cell0      - mesh->local_cells_disp ] - phi_component[cell1      - mesh->local_cells_disp];
                    dX   = mesh->cell_centers[cell0 - mesh->shmem_cell_disp]   - mesh->cell_centers[cell1 - mesh->shmem_cell_disp];
                }
                
                // 
                // ADD code for porous cells here!!
                // 

                A(0,0) = A(0,0) + dX.x * dX.x;
                A(1,0) = A(1,0) + dX.x * dX.y;
                A(2,0) = A(2,0) + dX.x * dX.z;

                A(0,1) = A(1,0);
                A(1,1) = A(1,1) + dX.y * dX.y;
                A(2,1) = A(2,1) + dX.y * dX.z;

                A(0,2) = A(2,0);
                A(1,2) = A(2,1);
                A(2,2) = A(2,2) + dX.x * dX.z;

                b(0) = b(0) + dX.x * dPhi;
                b(1) = b(1) + dX.y * dPhi;
                b(2) = b(2) + dX.z * dPhi;
            }

            // Swap out eigen solve for LAPACK SGESV?
            // call SGESV( 3, 1, A, 3, IPIV, RHS_A, 3, INFO )

            x = A.partialPivLu().solve(b);

            phi_grad_component[block_cell].x = x(0);
            phi_grad_component[block_cell].x = x(1);
            phi_grad_component[block_cell].x = x(2);
        }
    }


    template<typename T> void FlowSolver<T>::setup_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component )
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function setup_sparse_matrix.\n", mpi_config->rank);

        T RURF = 1. / URFactor;

        static int timestep = 0;
        // if ((timestep % 100) == 0)
        //     printf("Flow Rank %d SpmatrixA before (rows %lu cols %lu nnz %lu) \n", mpi_config->rank, A_spmatrix.rows(), A_spmatrix.cols(), A_spmatrix.nonZeros() );

        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
            residual[i] = 0.0;

        uint64_t face_count = 0;
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
            T app = A_phi_component[i];

            for (uint64_t f = 0; f < mesh->faces_per_cell; f++)
            {
                T face_value = 0.0;
                uint64_t face_id  = mesh->cell_faces[i * mesh->faces_per_cell + f];

                uint64_t cell0    = mesh->faces[face_id].cell0 - mesh->local_cells_disp;
                // uint64_t cell1    = mesh->faces[face_id].cell1 - mesh->local_cells_disp;
                uint64_t cell1    = ( mesh->faces[face_id].cell1 == MESH_BOUNDARY ) ? mesh->local_mesh_size : mesh->faces[face_id].cell1 - mesh->local_cells_disp;
                
                // if ( cell1 != MESH_BOUNDARY )
                // {
                    if ( cell0 == i )
                    {
                        if ( cell1 >= mesh->local_mesh_size ) continue;

                        face_value = face_fields[face_id].cell1;

                        A_spmatrix.coeffRef(i, cell1) = face_value;

                        residual[i] = residual[i] - face_value * phi_component[cell1];
                        face_count++;
                    }
                    else if ( cell1 == i )
                    {
                        if ( cell0 >= mesh->local_mesh_size ) continue;

                        face_value = face_fields[face_id].cell0;

                        A_spmatrix.coeffRef(i, cell0) = face_value;

                        residual[i] = residual[i] - face_value * phi_component[cell0];
                        face_count++;
                    }
                    app = app - face_value;
                // }
            }

            A_phi_component[i]  = app * RURF;
            S_phi_component[i]  = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];
            
            A_spmatrix.coeffRef(i, i) = A_phi_component[i];
            face_count++;

            residual[i] = residual[i] + S_phi_component[i] - A_phi_component[i] * phi_component[i];
        }

        A_spmatrix.makeCompressed();

        // Questions
        // 1. What do the diagonal values represent? A cell doesn't have a face with itself with itself?]
        //      - A[cell, cell] = (A_Phi[cell] - face_value[0:num_faces]) / URFactor
        // 2. What does A_phi represent? 
        // 3. What does Res represent?

        if ((timestep % 100) == 0 && mpi_config->particle_flow_rank == 0 && A_spmatrix.nonZeros() < 50 )
        {
            printf("\nFlow Rank %d SpmatrixA (rows %lu cols %lu nnz %lu) \n", mpi_config->particle_flow_rank, A_spmatrix.rows(), A_spmatrix.cols(), A_spmatrix.nonZeros() );
            cout << "A Matrix : " << endl << Eigen::MatrixXd(A_spmatrix) << endl;

            cout << "Phi   : " << endl;
            for ( uint64_t i = 0; i < mesh->local_mesh_size; i++ )  cout << phi_component[i] << " ";
            cout << endl;

            cout << "S_phi : " << endl;
            for ( uint64_t i = 0; i < mesh->local_mesh_size; i++ )  cout << S_phi_component[i] << " ";
            cout << endl << endl;
        }

        timestep++;

        // if( face_count /= NNZ ) write(*,*)'+ error: SetUpMatrixA: NNZ =',ia,' =/=',NNZ

        // T res0 = sqrt(sum(dble(abs(Res(1:Ncel)**2)))) * ResiNorm(iVar)

        // !   if( Res0 > 1.e8 ) Res0 = 10.0
    }

    template<typename T> void FlowSolver<T>::solve_sparse_matrix ( T *A_phi_component, T *phi_component, T *old_phi_component, T *S_phi_component )
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_sparse_matrix.\n", mpi_config->rank);
        
        Eigen::Map<Eigen::VectorXd>    S_phi_vector(S_phi_component, mesh->local_mesh_size);
        Eigen::Map<Eigen::VectorXd>      phi_vector(phi_component,   mesh->local_mesh_size + 1);
        // Eigen::Map<Eigen::VectorXd> phi_grad_vector(phi_component,   mesh->local_mesh_size + 1);

        eigen_solver.compute(A_spmatrix);

        // phi_vector = eigen_solver.solveWithGuess(S_phi_vector, phi_grad_vector);
        phi_vector = eigen_solver.solve(S_phi_vector);

    }

    template<typename T> void FlowSolver<T>::calculate_flux_UVW()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_flux_UVW.\n", mpi_config->rank);

        T pe0 =  9999.;
        T pe1 = -9999.;
        // TotalForce = 0.0

        T GammaBlend = 0.0; // NOTE: Change when implemented other differencing schemes.

        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            uint64_t cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            uint64_t cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if ( cell1 == MESH_BOUNDARY )  continue; // Skip boundary faces.
            if ( cell0 > mesh->local_mesh_size || cell1 > mesh->local_mesh_size ) continue; // Skip faces between blocks.

            // Also need condition to deal boundary cases
            T lambda0 = face_lambdas[face];    // dist(cell_center0, face_center) / dist(cell_center0, cell_center1)
            T lambda1 = 1.0 - lambda0;         // dist(face_center,  cell_center1) / dist(cell_center0, cell_center1)

            // T Uac     = phi.U[cell0] * lambda0 + phi.U[ip] * lambda1;
            // T Vac     = phi.V[cell0] * lambda0 + phi.V[ip] * lambda1;
            // T Wac     = phi.W[cell0] * lambda0 + phi.W[ip] * lambda1;

            vec<T> dUdXac  =   phi_grad.U[cell0] * lambda0 + phi_grad.U[cell1] * lambda1;
            vec<T> dVdXac  =   phi_grad.V[cell0] * lambda0 + phi_grad.V[cell1] * lambda1;
            vec<T> dWdXac  =   phi_grad.W[cell0] * lambda0 + phi_grad.W[cell1] * lambda1;

            T Visac   = effective_viscosity * lambda0 + effective_viscosity * lambda1;
            T VisFace = Visac * face_rlencos[face];
            
            vec<T> Xpn     = mesh->cell_centers[mesh->faces[face].cell1 - mesh->shmem_cell_disp] - mesh->cell_centers[mesh->faces[face].cell0 - mesh->shmem_cell_disp];
            

            // NOTE: ADD other differencing schemes. For now we just use Upwind Differencing Scheme (UDS)

            // call SelectDiffSchemeVector(i,iScheme,iP,iN,     &
            //                             U,V,W,               &
            //                             dUdX,dVdX,dWdX,      &
            //                             UFace,VFace,Wface);

            T UFace, VFace, WFace;
            if ( face_mass_fluxes[face] >= 0.0 )
            {
                UFace  = phi.U[cell0];
                VFace  = phi.V[cell0];
                WFace  = phi.W[cell0];
            }
            else
            {
                UFace  = phi.U[cell1];
                VFace  = phi.V[cell1];
                WFace  = phi.W[cell1];
            }

            //
            // explicit higher order convective flux (see eg. eq. 8.16)
            //

            T fuce = face_mass_fluxes[face] * UFace;
            T fvce = face_mass_fluxes[face] * VFace;
            T fwce = face_mass_fluxes[face] * WFace;

            T sx = face_normals[face].x;
            T sy = face_normals[face].y;
            T sz = face_normals[face].z;

            //
            // explicit higher order diffusive flux based on simple uncorrected
            // interpolated cell centred gradients(see eg. eq. 8.19)
            //

            T fude1 = (dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz;
            T fvde1 = (dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz;
            T fwde1 = (dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz;

            //
            // expliciete diffusieve flux/gecorrigeerde diffusie
            // Jasaks over-relaxed approach (thesis).
            //

            T fude = Visac * fude1;
            T fvde = Visac * fvde1;
            T fwde = Visac * fwde1;

            // !
            // ! implicit lower order (simple upwind)
            // ! convective and diffusive fluxes
            // !

            T fmin = min( face_mass_fluxes[face], 0.0 );
            T fmax = max( face_mass_fluxes[face], 0.0 );

            T fuci = fmin * phi.U[cell0] + fmax * phi.U[cell1];
            T fvci = fmin * phi.V[cell0] + fmax * phi.V[cell1];
            T fwci = fmin * phi.W[cell0] + fmax * phi.W[cell1];

            T fudi = VisFace * dot_product( dUdXac , Xpn );
            T fvdi = VisFace * dot_product( dVdXac , Xpn );
            T fwdi = VisFace * dot_product( dWdXac , Xpn );

            // !
            // ! convective coefficients with deferred correction with
            // ! gamma as the blending factor (0.0 <= gamma <= 1.0)
            // !
            // !      low            high    low  OLD
            // ! F = F    + gamma ( F     - F    )
            // !     ----   -------------------------
            // !      |                  |
            // !  implicit           explicit (dump into source term)
            // !
            // !            diffusion       convection
            // !                v               v
            
            face_fields[face].cell0 = -VisFace - max( face_mass_fluxes[face] , 0.0 );  // P (e);
            face_fields[face].cell1 = -VisFace + min( face_mass_fluxes[face] , 0.0 );  // N (w);

            T blend_u = GammaBlend * ( fuce - fuci );
            T blend_v = GammaBlend * ( fvce - fvci );
            T blend_w = GammaBlend * ( fwce - fwci );

            // !
            // ! assemble the two source terms
            // !

            S_phi.U[cell0] = S_phi.U[cell0] - blend_u + fude - fudi;
            S_phi.U[cell1] = S_phi.U[cell1] + blend_u - fude + fudi;

            S_phi.V[cell0] = S_phi.V[cell0] - blend_v + fvde - fvdi;
            S_phi.V[cell1] = S_phi.V[cell1] + blend_v - fvde + fvdi;

            S_phi.W[cell0] = S_phi.W[cell0] - blend_w + fwde - fwdi;
            S_phi.W[cell1] = S_phi.W[cell1] + blend_w - fwde + fwdi;

            const T small_epsilon = 1.e-20;
            T peclet = face_mass_fluxes[face] / face_areas[face] * magnitude(Xpn) / (Visac+small_epsilon);
            pe0 = min( pe0 , peclet );
            pe1 = max( pe1 , peclet );
        }
    }

    // integer :: IMoni  =   1          !! monitor cell for pressure ( material 1 )
    // integer :: Iter   =   0          !! iteration counter
    // real    :: Time   =   0.0        !! current time (if applicable)

    // real :: Tref      = 273.         !! ref. cell for temperature (material 1)
    // real :: Pref      =   0.0        !! ref. cell for pressure (material 1)

    // real :: DensRef   =   1.2        !! ref. density <= later lucht op 20C
    // real :: VisLam    =   0.001      !! lam. viscosity <= later lucht op 20C

    // real :: Prandtl   =   0.6905     !! Prandtl number (air 20C)
    // real :: Schmidt   =   0.9        !! Schmidt number

    // real :: Sigma_T   =   0.9        !! turb. model coefficient
    // real :: Sigma_k   =   1.0        !! turbulence diff. coef. factors
    // real :: Sigma_e   =   1.219      !! ie. turbulent Prandtl numbers
    // real :: Sigma_s   =   0.9        !! Schmidt number

    // real :: Gravity(3) =     0.0     !! gravity vector
    // real :: Beta       =     0.0     !! expansie coef.
    // real :: CpStd      =  1006.0     !! Cp
    // real :: CvStd      =  1006.0     !! Cv
    // real :: Lambda     =     0.02637 !! warmtegeleiding lucht

    // real :: Qtransfer  = 0.0
    // !                   u    v    w    p     k    e      T    Sc    Den  PP
    // real :: URF(10) =(/0.5, 0.5, 0.5, 0.2,  0.5, 0.5,   0.95, 0.95, 1.0, 0.8 /) !! underrelaxation factors
    // real :: RTOL(8) =(/0.1, 0.1, 0.1, 0.05, 0.1, 0.1,   0.1 , 0.1            /) !! rel. tolerance
    // real :: ATOL(8) =(/0.0, 0.0, 0.0, 0.0,  0.0, 0.0,   0.0 , 0.0            /) !! abs. tolerance
    // real :: Guess(8)=(/0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 293.0 , 0.0            /) !! initial guess

    // real    :: ResMax   = 1.e-4      !! target maximum residual
    // integer :: MaxPCOR  =    4       !! maximum number of pressure correction iterations
    // real    :: FactDPP  = 0.25       !! reduction in pressure correction


    template<typename T> void FlowSolver<T>::calculate_UVW()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_UVW.\n", mpi_config->rank);


        static double init_time  = 0.0;
        static double grad_time  = 0.0;
        static double flux_time  = 0.0;
        static double setup_time = 0.0;
        static double solve_time = 0.0;


        MPI_Barrier(mpi_config->particle_flow_world);
        init_time -= MPI_Wtime(); 

        // Initialise A_phi.U, S_phi.U and Aval vectors to 0.
        #pragma ivdep 
        for ( uint64_t i = 0; i < mesh->local_mesh_size; i++ )
        {
            A_phi.U[i] = 0.0;
            A_phi.V[i] = 0.0;
            A_phi.W[i] = 0.0;

            S_phi.U[i] = 0.0;
            S_phi.V[i] = 0.0;
            S_phi.W[i] = 0.0;
        }

        ptr_swap(&phi.U, &old_phi.U);
        ptr_swap(&phi.V, &old_phi.V);
        ptr_swap(&phi.W, &old_phi.W);
        ptr_swap(&phi.P, &old_phi.P);

        init_time += MPI_Wtime();
        grad_time -= MPI_Wtime();

        get_phi_gradient ( phi.U, phi_grad.U );
        get_phi_gradient ( phi.V, phi_grad.V );
        get_phi_gradient ( phi.W, phi_grad.W );

        grad_time += MPI_Wtime();
        flux_time -= MPI_Wtime();

        // calculate fluxes through all inner faces
        calculate_flux_UVW ();

        // Solve for Enthalpy
        // Use patches
        // Pressure force

        // If Transient and Euler
        if ( true )
        {
            double rdelta = 1.0 / delta;

            #pragma ivdep
            for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++ )
            {
                double f = cell_densities[i] * cell_volumes[i] * rdelta;

                S_phi.U[i] += f * old_phi.U[i];
                S_phi.V[i] += f * old_phi.V[i];
                S_phi.W[i] += f * old_phi.W[i];

                A_phi.U[i] += f;
                A_phi.V[i] += f;
                A_phi.W[i] += f;
            }
        }

        const double UVW_URFactor = 0.5;

        flux_time  += MPI_Wtime();
        setup_time -= MPI_Wtime();
        setup_sparse_matrix (UVW_URFactor, A_phi.U, phi.U, S_phi.U);   
        setup_time  += MPI_Wtime();
        solve_time  -= MPI_Wtime();     
        solve_sparse_matrix (A_phi.U, phi.U, old_phi.U, S_phi.U);




        setup_time  -= MPI_Wtime();
        solve_time  += MPI_Wtime();  
        setup_sparse_matrix (UVW_URFactor, A_phi.V, phi.V, S_phi.V); 
        setup_time  += MPI_Wtime();
        solve_time  -= MPI_Wtime();         
        solve_sparse_matrix (A_phi.V, phi.V, old_phi.V, S_phi.V);




        setup_time  -= MPI_Wtime();
        solve_time  += MPI_Wtime();  
        setup_sparse_matrix (UVW_URFactor, A_phi.W, phi.W, S_phi.W);
        setup_time  += MPI_Wtime();
        solve_time  -= MPI_Wtime();  
        solve_sparse_matrix (A_phi.W, phi.W, old_phi.W, S_phi.W);



        
        MPI_Barrier(mpi_config->particle_flow_world);
        solve_time += MPI_Wtime();

        static int timestep = 0; 

        if ((timestep++ == 1499) && mpi_config->particle_flow_rank == 0)
        {
            printf("Init  time: %7.2fs\n", init_time  );
            printf("Grad  time: %7.2fs\n", grad_time  );
            printf("Flux  time: %7.2fs\n", flux_time  );
            printf("Setup time: %7.2fs\n", setup_time );
            printf("Solve time: %7.2fs\n", solve_time );

        }
    }
    
    template<typename T> void FlowSolver<T>::solve_combustion_equations()
    {
        if (FLOW_SOLVER_DEBUG && mpi_config->particle_flow_rank == 0) printf("\tRunning function solve_combustion_equations.\n");
    }

    template<typename T> void FlowSolver<T>::update_combustion_fields()
    {
        if (FLOW_SOLVER_DEBUG && mpi_config->particle_flow_rank == 0) printf("\tRunning function update_combustion_fields.\n");
    }

    template<typename T> void FlowSolver<T>::solve_turbulence_equations()
    {
        if (FLOW_SOLVER_DEBUG && mpi_config->particle_flow_rank == 0) printf("\tRunning function solve_turbulence_equations.\n");
    }

    template<typename T> void FlowSolver<T>::update_turbulence_fields()
    {
        if (FLOW_SOLVER_DEBUG && mpi_config->particle_flow_rank == 0) printf("\tRunning function update_turbulence_fields.\n");
    }

    template<typename T> void FlowSolver<T>::solve_flow_equations()
    {
        if (FLOW_SOLVER_DEBUG && mpi_config->particle_flow_rank == 0) printf("\tRunning function solve_flow_equations.\n");
    }

        template<class T>
    void FlowSolver<T>::print_logger_stats(uint64_t timesteps, double runtime)
    {
        Flow_Logger loggers[mpi_config->particle_flow_world_size];
        MPI_Gather(&logger, sizeof(Flow_Logger), MPI_BYTE, &loggers, sizeof(Flow_Logger), MPI_BYTE, 0, mpi_config->particle_flow_world);

        double max_red_cells = loggers[0].reduced_recieved_cells;
        double min_red_cells = loggers[0].reduced_recieved_cells;
        double max_cells     = loggers[0].recieved_cells;
        double min_cells     = loggers[0].recieved_cells;
        double min_nodes = loggers[0].sent_nodes;
        double max_nodes = loggers[0].sent_nodes;

        double non_zero_blocks      = 0;
        double total_cells_recieved = 0;
        double total_reduced_cells_recieves = 0;
        if (mpi_config->particle_flow_rank == 0)
        {
            memset(&logger,           0, sizeof(Flow_Logger));
            for (int rank = 0; rank < mpi_config->particle_flow_world_size; rank++)
            {
                total_cells_recieved          += loggers[rank].recieved_cells;
                total_reduced_cells_recieves  += loggers[rank].reduced_recieved_cells;
                logger.reduced_recieved_cells += loggers[rank].reduced_recieved_cells;
                logger.recieved_cells         += loggers[rank].recieved_cells;
                logger.sent_nodes             += loggers[rank].sent_nodes;


                if ( min_cells > loggers[rank].recieved_cells )  min_cells = loggers[rank].recieved_cells ;
                if ( max_cells < loggers[rank].recieved_cells )  max_cells = loggers[rank].recieved_cells ;

                if ( min_red_cells > loggers[rank].reduced_recieved_cells )  min_red_cells = loggers[rank].reduced_recieved_cells ;
                if ( max_red_cells < loggers[rank].reduced_recieved_cells )  max_red_cells = loggers[rank].reduced_recieved_cells ;

                if ( min_nodes > loggers[rank].sent_nodes )  min_nodes = loggers[rank].sent_nodes ;
                if ( max_nodes < loggers[rank].sent_nodes )  max_nodes = loggers[rank].sent_nodes ;
            }
            
            for (int rank = 0; rank < mpi_config->particle_flow_world_size; rank++) 
                non_zero_blocks += loggers[rank].recieved_cells > (0.01 * max_cells) ;


            logger.reduced_recieved_cells /= non_zero_blocks;
            logger.recieved_cells /= non_zero_blocks;
            logger.sent_nodes     /= non_zero_blocks;
            
            printf("Flow Solver Stats:\t                            AVG       MIN       MAX\n");
            printf("\tReduced Recieved Cells ( per rank ) : %9.0f %9.0f %9.0f\n", round(logger.reduced_recieved_cells / timesteps), round(min_red_cells / timesteps), round(max_red_cells / timesteps));
            printf("\tRecieved Cells ( per rank )         : %9.0f %9.0f %9.0f\n", round(logger.recieved_cells / timesteps), round(min_cells / timesteps), round(max_cells / timesteps));
            printf("\tSent Nodes     ( per rank )         : %9.0f %9.0f %9.0f\n", round(logger.sent_nodes     / timesteps), round(min_nodes / timesteps), round(max_nodes / timesteps));
            printf("\tFlow blocks with <1%% max droplets  : %d\n", mpi_config->particle_flow_world_size - (int)non_zero_blocks); 
            printf("\tAvg Cells with droplets             : %.2f%%\n", 100 * total_cells_recieved / (timesteps * mesh->mesh_size));
            printf("\tCell copies across particle ranks   : %.2f%%\n", 100.*(1 - total_reduced_cells_recieves / total_cells_recieved ));

            
            MPI_Barrier (mpi_config->particle_flow_world);

            // printf("\tFlow Rank %4d: Recieved Cells %7.0f Sent Nodes %7.0f\n", mpi_config->particle_flow_rank, round(loggers[mpi_config->particle_flow_rank].recieved_cells / timesteps), round(loggers[mpi_config->particle_flow_rank].sent_nodes / timesteps));

            // MPI_Barrier (mpi_config->particle_flow_world);
            cout << endl;
        }
        else
        {
            MPI_Barrier (mpi_config->particle_flow_world);
            // printf("\tFlow Rank %4d: Recieved Cells %7.0f Sent Nodes %7.0f\n", mpi_config->particle_flow_rank, round(logger.recieved_cells / timesteps), round(logger.sent_nodes / timesteps));
            // MPI_Barrier (mpi_config->particle_flow_world);
        }

        MPI_Barrier(mpi_config->world);


        performance_logger.print_counters(mpi_config->rank, mpi_config->world_size, runtime);

    }

    template<typename T> void FlowSolver<T>::timestep()
    {
        if (FLOW_SOLVER_DEBUG) printf("Start flow timestep\n");
        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Start flow timestep.\n", mpi_config->rank);

        static int count = 0;
        int comms_timestep = 1;
        if ((count % comms_timestep) == 0)  update_flow_field();

        if ((count % 100) == 0)
        {
            double arr_usage  = ((double)get_array_memory_usage()) / 1.e9;
            double stl_usage  = ((double)get_stl_memory_usage())   / 1.e9 ;
            double mesh_usage = ((double)mesh->get_memory_usage()) / 1.e9 ;
            double arr_usage_total, stl_usage_total, mesh_usage_total;

            MPI_Reduce(&arr_usage,  &arr_usage_total,  1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&stl_usage,  &stl_usage_total,  1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&mesh_usage, &mesh_usage_total, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);

            if ( mpi_config->particle_flow_rank == 0 )
            {
                // printf("                Flow     array mem (TOTAL %8.3f GB) (AVG %8.3f GB) STL mem (TOTAL %8.3f GB) (AVG %8.3f GB) \n", arr_usage_total, arr_usage_total / mpi_config->particle_flow_world_size, 
                //                                                                                                                         stl_usage_total, stl_usage_total / mpi_config->particle_flow_world_size);
                printf("                Flow     mem (TOTAL %8.3f GB) (AVG %8.3f GB) \n", (arr_usage_total + stl_usage_total + mesh_usage_total), (arr_usage_total + stl_usage_total + mesh_usage_total) / mpi_config->particle_flow_world_size);

            }
        }

        calculate_UVW();

        // solve_combustion_equations();
        // update_combustion_fields();
        // solve_turbulence_equations();
        // update_turbulence_fields();
        // solve_flow_equations();
        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Stop flow timestep.\n", mpi_config->rank);
        count++;
    }

}   // namespace minicombust::flow 