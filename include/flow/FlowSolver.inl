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
        const uint64_t index_start = element_disps[recv_id];
        const uint64_t index_end   = element_disps[recv_id+1];


        double node_neighbours   = 8;
        const uint64_t cell_size = mesh->cell_size;

        resize_nodes_arrays(node_to_position_map.size() + (index_end - index_start) * cell_size + 1 ); // TODO: Move outside loop

        #pragma ivdep
        for (uint64_t i = index_start; i < index_end; i++)
        {
            // printf("Flow rank %d getting %lu cell %lu MAX INDEX %lu\n", mpi_config->rank, i, neighbour_indexes[i], cell_index_array_size / sizeof(uint64_t));
            uint64_t cell = neighbour_indexes[i];

            // if (cell == 515168)
            //     printf("Rank %d has cell %lu\n", mpi_config->rank, cell);

            local_particle_node_sets[recv_id].insert(&mesh->cells[(cell - mesh->shmem_cell_disp)      * mesh->cell_size], 
                                                     &mesh->cells[( 1 + cell - mesh->shmem_cell_disp) * mesh->cell_size]);


            // #pragma ivdep
            // for (uint64_t n = 0; n < cell_size; n++)
            // {
            //     const uint64_t node_id      = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];
            //     local_particle_node_sets[recv_id].insert(node_id);
            // }

            if ( new_cells_set.contains(cell) )  continue;

            new_cells_set.insert(cell);

            unordered_neighbours_set[0].insert(cell);         

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id      = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];
                // local_particle_node_sets[recv_id].insert(node_id);


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

                    // if (node_id == 540968 && mpi_config->rank == 56 )
                    //     printf("Rank %d cell %lu adds temp value %f (total %f) to new node %lu slot (boundary_neighbours %f) \n", mpi_config->rank, cell, mesh->dummy_gas_tem * (boundary_neighbours / node_neighbours), interp_node_flow_fields[node_to_position_map[node_id]].temp, node_id, boundary_neighbours);
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
                flow_grad_term = mesh->flow_grad_terms[displaced_c]; 
            }

            const vec<T> cell_centre         = mesh->cell_centres[c - mesh->shmem_cell_disp];

            // USEFUL ERROR CHECKING!
            // if (flow_term.temp     != mesh->dummy_gas_tem)   {printf("INTERP NODAL ERROR: Wrong temp value at %d max(%lu) \n", (int)displaced_c, mesh->block_element_disp[mpi_config->particle_flow_rank + 1]); exit(1);}
            // if (flow_term.pressure != mesh->dummy_gas_pre)   {printf("INTERP NODAL ERROR: Wrong pres value at %d max(%lu) \n", (int)displaced_c, mesh->block_element_disp[mpi_config->particle_flow_rank + 1]); exit(1);}
            // if (flow_term.vel.x    != mesh->dummy_gas_vel.x) {printf("INTERP NODAL ERROR: Wrong velo value at %d max(%lu) \n", (int)displaced_c, mesh->block_element_disp[mpi_config->particle_flow_rank + 1]); exit(1);}

            // if (flow_grad_term.temp     != 0.)                 {printf("INTERP NODAL ERROR: Wrong temp grad value\n"); exit(1);}
            // if (flow_grad_term.pressure != 0.)                 {printf("INTERP NODAL ERROR: Wrong pres grad value\n"); exit(1);}
            // if (flow_grad_term.vel.x    != 0.)                 {printf("INTERP NODAL ERROR: Wrong velo grad value\n"); exit(1);}

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

                    // if (node_id == 540968 && mpi_config->rank == 56)
                    //     printf("Rank %d cell %lu adds temp value %f (total %f) to existing node %lu (position %lu) slot\n", mpi_config->rank, c, (flow_term.temp + dot_product(flow_grad_term.temp, direction)) / node_neighbours, interp_node_flow_fields[node_to_position_map[node_id]].temp, node_id, node_to_position_map[node_id]);
                }
            }
        }

        // Useful for checking errors and comms

        // TODO: Can comment this with properly implemented halo exchange. Need cell neighbours for halo and nodes!
        // uint64_t const nsize = node_to_position_map.size();

        // #pragma ivdep
        // for ( uint64_t i = 0;  i < nsize; i++ ) 
        // {
        //     if (interp_node_flow_fields[i].temp     != mesh->dummy_gas_tem)              
        //         {printf("ERROR INTERP NODAL FINAL CHECK (RANK %d): Wrong temp value %f at %lu\n",  mpi_config->rank,           interp_node_flow_fields[i].temp,     interp_node_indexes[i]); exit(1);}
        //     if (interp_node_flow_fields[i].pressure != mesh->dummy_gas_pre)              
        //         {printf("ERROR INTERP NODAL FINAL CHECK (RANK %d): Wrong pres value %f at %lu\n",  mpi_config->rank,           interp_node_flow_fields[i].pressure, interp_node_indexes[i]); exit(1);}
        //     if (interp_node_flow_fields[i].vel.x != mesh->dummy_gas_vel.x) 
        //         {printf("ERROR INTERP NODAL FINAL CHECK (RANK %d): Wrong velo value {%.10f y z} at %lu\n",  mpi_config->rank,  interp_node_flow_fields[i].vel.x,    interp_node_indexes[i]); exit(1);}

        //     interp_node_flow_fields[i].temp     = mesh->dummy_gas_tem;
        //     interp_node_flow_fields[i].pressure = mesh->dummy_gas_pre;
        //     interp_node_flow_fields[i].vel      = mesh->dummy_gas_vel;
        // }
    }

    
    template<typename T> void FlowSolver<T>::update_flow_field()
    {
        if (FLOW_DEBUG) printf("\tRunning function update_flow_field.\n");
        
        int time_count = 0;
        time_stats[time_count]  -= MPI_Wtime(); //0
        
        cell_particle_field_map[0].clear();
        unordered_neighbours_set[0].clear();
        node_to_position_map.clear();
        new_cells_set.clear();
        
        performance_logger.my_papi_start();

        int recvs_complete = 0;
        uint64_t particle_recvs = 0;
        uint64_t total_elements = 0;
        element_disps[0] = 0;

        MPI_Ibcast(&recvs_complete, 1, MPI_INT, 0, mpi_config->world, &recv_requests[0]);
        
        int message_waiting = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[particle_recvs]);

        bool triggered_first_time = 0;

        while(!recvs_complete)
        {
            if ( message_waiting )
            {
                // printf("Flow rank %d recieving from block %d\n", mpi_config->rank, statuses[particle_recvs].MPI_SOURCE);
                MPI_Recv( &elements[particle_recvs], 1, MPI_UINT64_T, statuses[particle_recvs].MPI_SOURCE, 0, mpi_config->world, MPI_STATUS_IGNORE );

                if (!triggered_first_time)
                {
                    time_stats[time_count++] += MPI_Wtime();
                    time_stats[time_count]   -= MPI_Wtime(); //1
                    triggered_first_time = 1;
                }

                total_elements                 += elements[particle_recvs];
                element_disps[particle_recvs+1] = total_elements;
                ranks[particle_recvs]           = statuses[particle_recvs].MPI_SOURCE;
                particle_recvs++;
            }

            MPI_Test(&recv_requests[0], &recvs_complete, MPI_STATUS_IGNORE);
            MPI_Iprobe(MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[particle_recvs]);

            recvs_complete &= !message_waiting;
        }

        if (!triggered_first_time)
        {
            time_stats[time_count++] += MPI_Wtime();
            time_stats[time_count]   -= MPI_Wtime(); //1
            triggered_first_time = 1;
        }

        MPI_Barrier(mpi_config->world);


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //2

        resize_cell_particle(&total_elements, NULL, NULL);
        for (uint64_t p = 0; p < particle_recvs; p++)
        {
            logger.recieved_cells += elements[p];

            MPI_Irecv(&neighbour_indexes[element_disps[p]], elements[p], MPI_UINT64_T,                       ranks[p], 1, mpi_config->world, &recv_requests[2 * p + 0] );
            MPI_Irecv(&cell_particle_aos[element_disps[p]], elements[p], mpi_config->MPI_PARTICLE_STRUCTURE, ranks[p], 2, mpi_config->world, &recv_requests[2 * p + 1] );

            if ( p >= local_particle_node_sets.size() )
                local_particle_node_sets.push_back(unordered_set<uint64_t>());
            local_particle_node_sets[p].clear();

        }

        uint64_t p_recv = particle_recvs - 1;
        bool     all_processed        = true;
        bool    *processed_neighbours = async_locks;
        for (uint64_t pi = 0; pi < particle_recvs; pi++)  
        {
            processed_neighbours[pi] = false;
            all_processed           &= processed_neighbours[pi];
        }
        while (!all_processed)
        {
            p_recv = (p_recv + 1) % particle_recvs;

            int recieved_indexes = 0;
            MPI_Test(&recv_requests[2 * p_recv], &recieved_indexes, MPI_STATUS_IGNORE);
            if ( recieved_indexes && !processed_neighbours[p_recv] )
            {
                get_neighbour_cells (p_recv);
                processed_neighbours[p_recv] = true;
            }
            
            all_processed = true;
            for (uint64_t pi = 0; pi < particle_recvs; pi++)  all_processed &= processed_neighbours[pi];
        }

        logger.reduced_recieved_cells += new_cells_set.size();


        MPI_Barrier(mpi_config->world);     
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //3
        

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //4
        
        interpolate_to_nodes ();

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //5

        // Send size of reduced neighbours of cells back to ranks.
        uint64_t neighbour_point_size = node_to_position_map.size();

        logger.sent_nodes += neighbour_point_size;

        uint64_t ptr_disp = 0;
        for (uint64_t p = 0; p < particle_recvs; p++)
        {
            resize_send_buffers_nodes_arrays (ptr_disp + local_particle_node_sets[p].size());
            uint64_t local_disp = 0;
            for ( uint64_t node : local_particle_node_sets[p] )
            {
                send_buffers_interp_node_indexes[ptr_disp + local_disp]     = interp_node_indexes[node_to_position_map[node]];
                send_buffers_interp_node_flow_fields[ptr_disp + local_disp] = interp_node_flow_fields[node_to_position_map[node]];
                local_disp++;
            }
            // printf("Rank %3d is sending data to %d\n", mpi_config->rank, ranks[p]);

            MPI_Isend ( send_buffers_interp_node_indexes + ptr_disp,     local_disp, MPI_UINT64_T,                   ranks[p], 0, mpi_config->world, &send_requests[3*p + 0] );
            MPI_Isend ( send_buffers_interp_node_flow_fields + ptr_disp, local_disp, mpi_config->MPI_FLOW_STRUCTURE, ranks[p], 1, mpi_config->world, &send_requests[3*p + 1] );
            ptr_disp += local_disp;
        }

        MPI_Waitall(2 * particle_recvs, recv_requests, MPI_STATUSES_IGNORE); // Check field values later on!

        
        for (uint64_t i = 0; i < element_disps[particle_recvs]; i++)
        {
            const uint64_t cell = neighbour_indexes[i];
            if ( cell_particle_field_map[0].count(cell) )
            {
                const uint64_t index = cell_particle_field_map[0][cell];

                cell_particle_aos[index].momentum += cell_particle_aos[i].momentum;
                cell_particle_aos[index].energy   += cell_particle_aos[i].energy;
                cell_particle_aos[index].fuel     += cell_particle_aos[i].fuel;
            }
            else
            {
                const uint64_t index = cell_particle_field_map[0].size();

                neighbour_indexes[index]         = cell;
                cell_particle_aos[index]         = cell_particle_aos[i];
                cell_particle_field_map[0][cell] = index;
            }
        }


        MPI_Barrier(mpi_config->world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //6

        
        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //7

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //8

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
        
        time_stats[time_count++] += MPI_Wtime();

        static int timestep_count = 0;
        if (timestep_count++ == 1499)
        {
            if ( mpi_config->particle_flow_rank == 0 )
            {
                for (int i = 0; i < time_count; i++)
                    MPI_Reduce(MPI_IN_PLACE, &time_stats[i], 1, MPI_DOUBLE, MPI_MAX, 0, mpi_config->particle_flow_world);

                double total_time = 0.0;
                printf("\nUpdate Flow Field Communuication Timings\n");

                for (int i = 0; i < time_count; i++)
                    total_time += time_stats[i];
                for (int i = 0; i < time_count; i++)
                    printf("Time stats %d: %f %.2f\n", i, time_stats[i], 100 * time_stats[i] / total_time);
                printf("Total time %f\n", total_time);

            }
            else{
                for (int i = 0; i < time_count; i++)
                    MPI_Reduce(&time_stats[i], nullptr, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_config->particle_flow_world);
            }
        }
    } 
    
    template<typename T> void FlowSolver<T>::solve_combustion_equations()
    {
        if (FLOW_DEBUG) printf("\tRunning function solve_combustion_equations.\n");
    }

    template<typename T> void FlowSolver<T>::update_combustion_fields()
    {
        if (FLOW_DEBUG) printf("\tRunning function update_combustion_fields.\n");
    }

    template<typename T> void FlowSolver<T>::solve_turbulence_equations()
    {
        if (FLOW_DEBUG) printf("\tRunning function solve_turbulence_equations.\n");
    }

    template<typename T> void FlowSolver<T>::update_turbulence_fields()
    {
        if (FLOW_DEBUG) printf("\tRunning function update_turbulence_fields.\n");
    }

    template<typename T> void FlowSolver<T>::solve_flow_equations()
    {
        if (FLOW_DEBUG) printf("\tRunning function solve_flow_equations.\n");
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

            printf("\tFlow Rank %4d: Recieved Cells %7.0f Sent Nodes %7.0f\n", mpi_config->particle_flow_rank, round(loggers[mpi_config->particle_flow_rank].recieved_cells / timesteps), round(loggers[mpi_config->particle_flow_rank].sent_nodes / timesteps));

            // MPI_Barrier (mpi_config->particle_flow_world);
            cout << endl;
        }
        else
        {
            MPI_Barrier (mpi_config->particle_flow_world);
            printf("\tFlow Rank %4d: Recieved Cells %7.0f Sent Nodes %7.0f\n", mpi_config->particle_flow_rank, round(logger.recieved_cells / timesteps), round(logger.sent_nodes / timesteps));
            // MPI_Barrier (mpi_config->particle_flow_world);
        }

        MPI_Barrier(mpi_config->world);


        performance_logger.print_counters(mpi_config->rank, mpi_config->world_size, runtime);

    }

    template<typename T> void FlowSolver<T>::timestep()
    {
        if (FLOW_DEBUG) printf("Start flow timestep\n");
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

        // solve_combustion_equations();
        // update_combustion_fields();
        // solve_turbulence_equations();
        // update_turbulence_fields();
        // solve_flow_equations();
        if (FLOW_DEBUG) printf("Stop flow timestep\n");
        count++;
    }

}   // namespace minicombust::flow 