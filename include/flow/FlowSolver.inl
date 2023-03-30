#include <stdio.h>
#include <limits.h>

#include "flow/FlowSolver.hpp"


using namespace std;

namespace minicombust::flow 
{
    template<typename T> inline bool FlowSolver<T>::is_halo ( uint64_t cell )
    {
        return ( cell - mesh->local_cells_disp >= mesh->local_mesh_size );
    }


    template<typename T> void FlowSolver<T>::exchange_cell_info_halos ()
    {
        int num_requests = 2;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Isend( cell_densities,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            MPI_Isend( cell_volumes,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
            // printf("T%lu Flow %d: Sending phi to flow rank %d\n", timestep_count, mpi_config->particle_flow_rank, halo_ranks[r]);
        }

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &cell_densities[mesh->local_mesh_size + halo_disps[r]],  halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            MPI_Irecv( &cell_volumes[mesh->local_mesh_size + halo_disps[r]],    halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 1] );
            // printf("T%lu Flow %d: Recieving %d neighbour indexes from flow rank %d (disp %d max %lu) \n", timestep_count, mpi_config->particle_flow_rank, halo_sizes[r], halo_ranks[r], halo_disps[r], nboundaries);
        }

        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::exchange_phi_halos ()
    {
        int num_requests = 8;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Isend( phi.U,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            MPI_Isend( phi.V,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
            MPI_Isend( phi.W,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 2, mpi_config->particle_flow_world, &send_requests[num_requests*r + 2] );
            MPI_Isend( phi.P,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 3, mpi_config->particle_flow_world, &send_requests[num_requests*r + 3] );
            MPI_Isend( phi_grad.U, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 4, mpi_config->particle_flow_world, &send_requests[num_requests*r + 4] );
            MPI_Isend( phi_grad.V, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 5, mpi_config->particle_flow_world, &send_requests[num_requests*r + 5] );
            MPI_Isend( phi_grad.W, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 6, mpi_config->particle_flow_world, &send_requests[num_requests*r + 6] );
            MPI_Isend( phi_grad.P, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 7, mpi_config->particle_flow_world, &send_requests[num_requests*r + 7] );
            // printf("T%lu Flow %d: Sending phi to flow rank %d\n", timestep_count, mpi_config->particle_flow_rank, halo_ranks[r]);
        }

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &phi.U[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            MPI_Irecv( &phi.V[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 1] );
            MPI_Irecv( &phi.W[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 2, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 2] );
            MPI_Irecv( &phi.P[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 3] );
            MPI_Irecv( &phi_grad.U[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 4, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 4] );
            MPI_Irecv( &phi_grad.V[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 5, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 5] );
            MPI_Irecv( &phi_grad.W[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 6, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 6] );
            MPI_Irecv( &phi_grad.P[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 7, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 7] );
            // printf("T%lu Flow %d: Recieving %d neighbour indexes from flow rank %d (disp %d max %lu) \n", timestep_count, mpi_config->particle_flow_rank, halo_sizes[r], halo_ranks[r], halo_disps[r], nboundaries);
        }

        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::exchange_A_halos (T *A_phi_component)
    {
        int num_requests = 1;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Isend( A_phi_component,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            // printf("T%lu Flow %d: Sending phi to flow rank %d\n", timestep_count, mpi_config->particle_flow_rank, halo_ranks[r]);
        }

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &A_phi_component[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            // printf("T%lu Flow %d: Recieving %d neighbour indexes from flow rank %d (disp %d max %lu) \n", timestep_count, mpi_config->particle_flow_rank, halo_sizes[r], halo_ranks[r], halo_disps[r], nboundaries);
        }

        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::exchange_S_halos (T *S_phi_component)
    {
        int num_requests = 1;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Isend( S_phi_component,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            // printf("T%lu Flow %d: Sending phi to flow rank %d\n", timestep_count, mpi_config->particle_flow_rank, halo_ranks[r]);
        }

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &S_phi_component[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            // printf("T%lu Flow %d: Recieving %d neighbour indexes from flow rank %d (disp %d max %lu) \n", timestep_count, mpi_config->particle_flow_rank, halo_sizes[r], halo_ranks[r], halo_disps[r], nboundaries);
        }

        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
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
        for ( uint64_t cell : unordered_neighbours_set[0] )
        {
            const uint64_t block_cell      = cell - mesh->local_cells_disp;
            const uint64_t shmem_cell      = cell - mesh->shmem_cell_disp;

            flow_aos<T> flow_term;      
            flow_aos<T> flow_grad_term; 

            if (is_halo(cell)) 
            {
                flow_term.temp          = mesh->dummy_gas_tem;      
                flow_grad_term.temp     = 0.0;  

                flow_term.pressure      = phi.P[boundary_map[cell]];      
                flow_grad_term.pressure = 0.0; 

                // if (!boundary_map.contains(cell)) exit(1);
                
                flow_term.vel.x         = phi.U[boundary_map[cell]]; 
                flow_term.vel.y         = phi.V[boundary_map[cell]]; 
                flow_term.vel.z         = phi.W[boundary_map[cell]]; 
                // flow_term.vel      = mesh->dummy_gas_vel; 
                flow_grad_term.vel = { 0.0, 0.0, 0.0 }; 
            }
            else
            {

                flow_term.vel.x    = phi.U[block_cell];
                flow_term.vel.y    = phi.V[block_cell];
                flow_term.vel.z    = phi.W[block_cell];
                flow_term.pressure = phi.P[block_cell];
                flow_term.temp     = mesh->flow_terms[block_cell].temp;

                flow_grad_term = mesh->flow_grad_terms[block_cell]; 
            }

            // cout << "vel " << print_vec(flow_term.vel) << " pressure " << flow_term.pressure << " temperature " << flow_term.temp << endl;


            const vec<T> cell_centre         = mesh->cell_centers[shmem_cell];

            // check_flow_field_exit ( "INTERP NODAL ERROR: Flow value",      &flow_term,      &mesh->dummy_flow_field,      cell );
            // check_flow_field_exit ( "INTERP NODAL ERROR: Flow grad value", &flow_grad_term, &mesh->dummy_flow_field_grad, cell );

            // if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL ERROR: Flow value",      &flow_term,      &mesh->dummy_flow_field,      cell );
            // if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL ERROR: Flow grad value", &flow_grad_term, &mesh->dummy_flow_field_grad, cell );

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id = mesh->cells[shmem_cell*mesh->cell_size + n];

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
                // if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL FINAL ERROR: ", &interp_node_flow_fields[i], &mesh->dummy_flow_field, i );
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

        bool  all_processed        = false;
        bool *processed_neighbours = async_locks;
        while(!all_processed)
        {
            time0 -= MPI_Wtime(); //1

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

        if (timestep_count == 1499)
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

        Eigen::Matrix3d A;
        Eigen::Vector3d b;
        Eigen::Vector3d x;

        // static uint64_t timestep = 0;

        for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
        {
            const uint64_t cell = block_cell + mesh->local_cells_disp;
            // vec<T> center      = mesh->cell_centers[real_cell - mesh->shmem_cell_disp];

            A = Eigen::Matrix3d::Zero();
            b = Eigen::Vector3d::Zero();

            for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
            {
                const uint64_t face  = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];

                const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
                const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

                const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

                if ( mesh->faces[face].cell1 >= mesh->mesh_size ) continue; // Calculation is different for cells with boundary neighbours

                if ( (block_cell1 >= mesh->local_mesh_size) || (block_cell0 >= mesh->local_mesh_size) ) 
                    continue; // Cell on either side of face is owned by different block. Halos required!

                uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

                const T mask = ( mesh->faces[face].cell0 == cell ) ? 1. : -1.;

                const T dPhi    = mask * (      phi_component[phi_index1]  - phi_component[phi_index0] );
                const vec<T> dX = mask * ( mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0] );
                
                // Note: ADD code for porous cells here

                // cout << f << "dX: " << print_vec(dX) << " phi " << dPhi << "     ( " << phi_component[block_cell1] << " " << phi_component[block_cell0] << " )" << endl;


                A(0,0) = A(0,0) + dX.x * dX.x;
                A(1,0) = A(1,0) + dX.x * dX.y;
                A(2,0) = A(2,0) + dX.x * dX.z;

                A(0,1) = A(1,0);
                A(1,1) = A(1,1) + dX.y * dX.y;
                A(2,1) = A(2,1) + dX.y * dX.z;

                A(0,2) = A(2,0);
                A(1,2) = A(2,1);
                A(2,2) = A(2,2) + dX.z * dX.z;

                b(0) = b(0) + dX.x * dPhi;
                b(1) = b(1) + dX.y * dPhi;
                b(2) = b(2) + dX.z * dPhi;
            }

            // Swap out eigen solve for LAPACK SGESV?
            // call SGESV( 3, 1, A, 3, IPIV, RHS_A, 3, INFO )

            x = A.partialPivLu().solve(b);

            phi_grad_component[block_cell].x = x(0);
            phi_grad_component[block_cell].y = x(1);
            phi_grad_component[block_cell].z = x(2);

            // cout << "A matrix: " << endl << A << endl;
            // printf("b %f %f %f\n", b(0), b(1), b(2));
            // printf("%lu phi_component %8.2f phi_grad_component %8.2f %8.2f %8.2f\n", timestep, phi_component[block_cell], phi_grad_component[block_cell].x, phi_grad_component[block_cell].y, phi_grad_component[block_cell].z);
            // exit(1);
        }
    }

    template<typename T> void FlowSolver<T>::get_phi_gradients ()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function get_phi_gradients.\n", mpi_config->rank);
        // NOTE: Currently Least squares is the only method supported

        Eigen::Matrix3d A;             // Independent of phi, reusable for each variable
        Eigen::PartialPivLU<Eigen::Matrix3d> A_decomposition;


        Eigen::Vector3d bU, bV, bW, bP;

        for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
        {
            const uint64_t cell = block_cell + mesh->local_cells_disp;

            A  = Eigen::Matrix3d::Zero();
            bU = Eigen::Vector3d::Zero();
            bV = Eigen::Vector3d::Zero();
            bW = Eigen::Vector3d::Zero();
            bP = Eigen::Vector3d::Zero();

            for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
            {
                const uint64_t face  = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];

                const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
                const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

                const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

                T dU, dV, dW, dP;
                vec<T> dX;

                if ( mesh->faces[face].cell1 < mesh->mesh_size )  // Inner cell
                {
                    const uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                    const uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

                    const T mask = ( mesh->faces[face].cell0 == cell ) ? 1. : -1.;

                    dU = mask * ( phi.U[phi_index1] - phi.U[phi_index0] );
                    dV = mask * ( phi.V[phi_index1] - phi.V[phi_index0] );
                    dW = mask * ( phi.W[phi_index1] - phi.W[phi_index0] );
                    dP = mask * ( phi.P[phi_index1] - phi.P[phi_index0] );
                    dX = mask * ( mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0] );
                    
                    // Note: ADD code for porous cells here
                } 
                else // Boundary face
                {
                    const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;

                    dU = phi.U[mesh->local_mesh_size + nhalos + boundary_cell] - phi.U[block_cell0];
                    dV = phi.V[mesh->local_mesh_size + nhalos + boundary_cell] - phi.V[block_cell0];
                    dW = phi.W[mesh->local_mesh_size + nhalos + boundary_cell] - phi.W[block_cell0];
                    dP = phi.P[mesh->local_mesh_size + nhalos + boundary_cell] - phi.P[block_cell0];

                    dX = face_centers[face] - mesh->cell_centers[shmem_cell0];
                }

                A(0,0) = A(0,0) + dX.x * dX.x;
                A(1,0) = A(1,0) + dX.x * dX.y;
                A(2,0) = A(2,0) + dX.x * dX.z;

                A(0,1) = A(1,0);
                A(1,1) = A(1,1) + dX.y * dX.y;
                A(2,1) = A(2,1) + dX.y * dX.z;

                A(0,2) = A(2,0);
                A(1,2) = A(2,1);
                A(2,2) = A(2,2) + dX.z * dX.z;

                bU(0) = bU(0) + dX.x * dU;
                bU(1) = bU(1) + dX.y * dU;
                bU(2) = bU(2) + dX.z * dU;

                bV(0) = bV(0) + dX.x * dV;
                bV(1) = bV(1) + dX.y * dV;
                bV(2) = bV(2) + dX.z * dV;

                bW(0) = bW(0) + dX.x * dW;
                bW(1) = bW(1) + dX.y * dW;
                bW(2) = bW(2) + dX.z * dW;

                bP(0) = bP(0) + dX.x * dP;
                bP(1) = bP(1) + dX.y * dP;
                bP(2) = bP(2) + dX.z * dP;
            }

            // Swap out eigen solve for LAPACK SGESV?
            // call SGESV( 3, 1, A, 3, IPIV, RHS_A, 3, INFO )

            Eigen::Map<Eigen::Vector3d> xU(&phi_grad.U[block_cell].x);
            Eigen::Map<Eigen::Vector3d> xV(&phi_grad.V[block_cell].x);
            Eigen::Map<Eigen::Vector3d> xW(&phi_grad.W[block_cell].x);
            Eigen::Map<Eigen::Vector3d> xP(&phi_grad.P[block_cell].x);

            A_decomposition = A.partialPivLu();
            xU = A_decomposition.solve(bU);
            xV = A_decomposition.solve(bV);
            xW = A_decomposition.solve(bW);
            xP = A_decomposition.solve(bP);
        }
    }

    template<typename T> void FlowSolver<T>::setup_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component )
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function setup_sparse_matrix.\n", mpi_config->rank);

        uint64_t face_count = 0;
        T RURF = 1. / URFactor;

        static double init_time     = 0.0;
        static double res_phi_time  = 0.0;
        static double halo_time     = 0.0;
        static double diagonal_time = 0.0;
        static double compress_time = 0.0;

        init_time -= MPI_Wtime();


        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
            residual[i]         = 0.0;
        }

        init_time    += MPI_Wtime();
        res_phi_time -= MPI_Wtime();
        

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if (mesh->faces[face].cell1 >= mesh->mesh_size)  continue; // Remove when implemented boundary cells. Treat boundary as mesh size

            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

            A_spmatrix.coeffRef(phi_index0, phi_index1) = face_fields[face].cell1;
            A_spmatrix.coeffRef(phi_index1, phi_index0) = face_fields[face].cell0;

            // if (isnan(face_fields[face].cell1) || isnan(face_fields[face].cell0) )
            // {
            //     printf("T%lu Rank %d nan\n", timestep_count, mpi_config->rank );
            //     exit(1);
            // }

            residual[phi_index0] = residual[phi_index0] - face_fields[face].cell1 * phi_component[phi_index1];
            residual[phi_index1] = residual[phi_index1] - face_fields[face].cell0 * phi_component[phi_index0];

            face_count += 2;

            A_phi_component[phi_index0] -= face_fields[face].cell1;
            A_phi_component[phi_index1] -= face_fields[face].cell0;
        }

        res_phi_time += MPI_Wtime();
        halo_time    -= MPI_Wtime();

        Eigen::Map<Eigen::VectorXd> A_phi_vector(A_phi_component, mesh->local_mesh_size + nhalos);
        Eigen::Map<Eigen::VectorXd> S_phi_vector(S_phi_component, mesh->local_mesh_size + nhalos);

        exchange_A_halos ( A_phi_component );

        MPI_Barrier(mpi_config->particle_flow_world);


        halo_time    += MPI_Wtime();
        diagonal_time -= MPI_Wtime();

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep 

        for (uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++)
        {
            A_phi_component[i] *= RURF;
            S_phi_component[i] = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];


            // if (isnan(A_phi_component[i]))
            // {
            //     printf("T%lu Rank %d nan\n", timestep_count, mpi_config->rank );
            //     exit(1);
            // }

            A_spmatrix.coeffRef(i, i) = A_phi_component[i];

            residual[i] = residual[i] + S_phi_component[i] - A_phi_component[i] * phi_component[i];
            face_count++;
        }

        diagonal_time += MPI_Wtime();
        compress_time -= MPI_Wtime();

        exchange_S_halos ( S_phi_component );  // TODO: We don't really need to do this halo exchange.


        // for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        // {
        //     const uint64_t cell = i + mesh->local_cells_disp;
        //     T app = A_phi_component[i];

        //     for (uint64_t f = 0; f < mesh->faces_per_cell; f++)
        //     {
        //         T face_value = 0.0;
        //         uint64_t face  = mesh->cell_faces[i * mesh->faces_per_cell + f];

        //         const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
        //         const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

        //         if (mesh->faces[face].cell1 == MESH_BOUNDARY)  continue;  // Remove when implemented boundary cells. Treat boundary as mesh size
                
        //         uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
        //         uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

        //         // if ( NOT BOUNDARY ) // Calculations are different for inlets, outlets, walls etc.
        //         // {
        //             if ( mesh->faces[face].cell0 == cell )
        //             {
        //                 face_value = face_fields[face].cell1;

        //                 A_spmatrix.coeffRef(i, phi_index1) = face_value;

        //                 residual[i] = residual[i] - face_value * phi_component[phi_index1];
        //                 face_count++;
        //             }
        //             else if ( mesh->faces[face].cell1 == cell )
        //             {
        //                 face_value = face_fields[face].cell0;

        //                 A_spmatrix.coeffRef(i, phi_index0) = face_value;

        //                 residual[i] = residual[i] - face_value * phi_component[phi_index0];
        //                 face_count++;
        //             }
        //             app = app - face_value;
        //         // }
        //     }

        //     A_phi_component[i]  = app * RURF;
        //     S_phi_component[i]  = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];
            
        //     A_spmatrix.coeffRef(i, i) = A_phi_component[i];
        //     face_count++;

        //     residual[i] = residual[i] + S_phi_component[i] - A_phi_component[i] * phi_component[i];
        // }

        A_spmatrix.makeCompressed();

        compress_time += MPI_Wtime();

        if ( mpi_config->particle_flow_rank == 0 && timestep_count == 1499)
        {
            printf("SETUP Init  time:         %7.2fs\n", init_time  );
            printf("SETUP res_phi_time  time: %7.2fs\n", res_phi_time  );
            printf("SETUP Halo time:          %7.2fs\n", halo_time );
            printf("SETUP Diagonal time:      %7.2fs\n", diagonal_time );
            printf("SETUP compress time:      %7.2fs\n", compress_time );
        }


        // if( face_count /= NNZ ) write(*,*)'+ error: SetUpMatrixA: NNZ =',ia,' =/=',NNZ

        // T res0 = sqrt(sum(dble(abs(Res(1:Ncel)**2)))) * ResiNorm(iVar)

        // !   if( Res0 > 1.e8 ) Res0 = 10.0
    }

    template<typename T> void FlowSolver<T>::update_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component )
    {
        // The idea for this function is to reduce the number of insertions into the sparse A matrix, 
        // given that face_fields (RFace) is constant for U, V and W. Possible to write app to seperate array and then only do one halo exchange here.

        uint64_t face_count = 0;
        T RURF = 1. / URFactor;

        static double init_time     = 0.0;
        static double res_phi_time  = 0.0;
        static double halo_time     = 0.0;
        static double diagonal_time = 0.0;

        init_time -= MPI_Wtime();

        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
            residual[i]         = 0.0;
            A_phi_component[i] *= RURF;
        }

        init_time    += MPI_Wtime();
        res_phi_time -= MPI_Wtime();

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if ( mesh->faces[face].cell1 >= mesh->mesh_size )  continue; // Remove when implemented boundary cells. Treat boundary as mesh size

            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

            residual[phi_index0] = residual[phi_index0] - face_fields[face].cell1 * phi_component[phi_index1];
            residual[phi_index1] = residual[phi_index1] - face_fields[face].cell0 * phi_component[phi_index0];

            face_count += 2;

            A_phi_component[phi_index0] -= RURF * face_fields[face].cell1;
            A_phi_component[phi_index1] -= RURF * face_fields[face].cell0;
        }

        res_phi_time += MPI_Wtime();
        halo_time    -= MPI_Wtime();

        exchange_A_halos (A_phi_component);

        halo_time    += MPI_Wtime();
        diagonal_time -= MPI_Wtime();

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++)
        {
            S_phi_component[i] = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];
            A_spmatrix.coeffRef(i, i) = A_phi_component[i];

            residual[i] = residual[i] + S_phi_component[i] - A_phi_component[i] * phi_component[i];
            face_count++;
        }

        diagonal_time += MPI_Wtime();

        if (mpi_config->particle_flow_rank == 0 && timestep_count == 1499)
        {
            printf("UPDATE Init  time:         %7.2fs\n", init_time  );
            printf("UPDATE res_phi_time  time: %7.2fs\n", res_phi_time  );
            printf("UPDATE Halo time:          %7.2fs\n", halo_time );
            printf("UPDATE Diagonal time:      %7.2fs\n", diagonal_time );
        }
    }


    template<typename T> void FlowSolver<T>::solve_sparse_matrix ( T *phi_component, T *S_phi_component )
    {
        static double init_time     = 0.0;
        static double compute_time  = 0.0;
        static double solve_time     = 0.0;

        init_time -= MPI_Wtime();

        Eigen::Map<Eigen::VectorXd> S_phi_vector(S_phi_component, mesh->local_mesh_size + nhalos);
        Eigen::Map<Eigen::VectorXd>   phi_vector(phi_component,   mesh->local_mesh_size + nhalos);
        // Eigen::Map<Eigen::VectorXd>   phi_vector(phi_component,   mesh->local_mesh_size + nhalos + mesh->boundary_cells_size);

        printf("\tRank %d: mesh_size %lu nhalos %lu boundary_cells %lu\n", mpi_config->rank, mesh->local_mesh_size, nhalos, mesh->boundary_cells_size );


        printf("\tRank %d: Running function solve_sparse_matrix A = (%lu %lu) x = (%lu %lu) b = (%lu %lu).\n", mpi_config->rank, 
                                                                                                                                       A_spmatrix.rows(),   A_spmatrix.cols(), 
                                                                                                                                       phi_vector.rows(),   phi_vector.cols(),
                                                                                                                                       S_phi_vector.rows(), S_phi_vector.cols());

        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_sparse_matrix A = (%lu %lu) x = (%lu %lu) b = (%lu %lu).\n", mpi_config->rank, 
                                                                                                                                       A_spmatrix.rows(),   A_spmatrix.cols(), 
                                                                                                                                       phi_vector.rows(),   phi_vector.cols(),
                                                                                                                                       S_phi_vector.rows(), S_phi_vector.cols());
        init_time += MPI_Wtime();
        compute_time -= MPI_Wtime();

        eigen_solver.setTolerance(0.1);
        // eigen_solver.setMaxIterations(4);

        if (mpi_config->particle_flow_rank == 0 && A_spmatrix.cols() < 20 )
        {
            cout << endl << mpi_config->particle_flow_rank << "Phi   : " << endl << phi_vector   << endl;
        }

        MPI_Barrier(mpi_config->particle_flow_world);


        if (mpi_config->particle_flow_rank == 1 && A_spmatrix.cols() < 20)
        {
            cout << endl << mpi_config->particle_flow_rank << "Phi   : " << endl << phi_vector   << endl;
        }
        
        check_array_nan("S_phi_vector", S_phi_component, mesh->local_mesh_size + nhalos, mpi_config, timestep_count);

        eigen_solver.compute(A_spmatrix);

        compute_time += MPI_Wtime();
        solve_time -= MPI_Wtime();

        phi_vector = eigen_solver.solveWithGuess(S_phi_vector, phi_vector);

        // phi_vector = eigen_solver.solve(S_phi_vector);

        if (mpi_config->particle_flow_rank == 0 && A_spmatrix.cols() < 20 )
        {
            printf("Timestep %lu\n", timestep_count);
            cout << "A Matrix     : " << endl << Eigen::MatrixXd(A_spmatrix) << endl;

            cout << endl << mpi_config->particle_flow_rank << "S_Phi : " << endl << S_phi_vector << endl;
            cout << endl << mpi_config->particle_flow_rank << "Phi   : " << endl << phi_vector   << endl;
        }

        MPI_Barrier(mpi_config->particle_flow_world);


        if (mpi_config->particle_flow_rank == 1 && A_spmatrix.cols() < 20)
        {
            printf("Timestep %lu\n", timestep_count);
            cout << "A Matrix     : " << endl << Eigen::MatrixXd(A_spmatrix) << endl;

            cout << endl << mpi_config->particle_flow_rank << "S_Phi : " << endl << S_phi_vector << endl;
            cout << endl << mpi_config->particle_flow_rank << "Phi   : " << endl << phi_vector   << endl;
        }

        if (timestep_count > 0)
        {
            exit(1);
        }

        // MPI_Barrier(mpi_config->particle_flow_world);


        // if (mpi_config->particle_flow_rank == 2 && A_spmatrix.cols() < 20)
        // {
        //     cout << "A Matrix     : " << endl << Eigen::MatrixXd(A_spmatrix) << endl;

        //     cout << endl << mpi_config->particle_flow_rank << "S_Phi : " << endl << S_phi_vector << endl;
        //     cout << endl << mpi_config->particle_flow_rank << "Phi   : " << endl << phi_vector   << endl;
        // }
        // MPI_Barrier(mpi_config->particle_flow_world);


        // if (mpi_config->particle_flow_rank == 3 && A_spmatrix.cols() < 20)
        // {
        //     cout << "A Matrix     : " << endl << Eigen::MatrixXd(A_spmatrix) << endl;

        //     cout << endl << mpi_config->particle_flow_rank << "S_Phi : " << endl << S_phi_vector << endl;
        //     cout << endl << mpi_config->particle_flow_rank << "Phi   : " << endl << phi_vector   << endl;
        // }


        check_array_nan("Phi_vector", phi_component, mesh->local_mesh_size + nhalos + mesh->boundary_cells_size, mpi_config, timestep_count);

        // if (timestep_count > 0)
        // {
        //     exit(1);
        // }

        solve_time += MPI_Wtime();

        if (mpi_config->particle_flow_rank == 0 && timestep_count == 1499)
        {
            printf("SOLVE Init  time:          %7.2fs\n", init_time  );
            printf("SOLVE compute  time:       %7.2fs\n", compute_time  );
            printf("SOLVE solve time:          %7.2fs\n", solve_time );
        }
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
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

            if ( mesh->faces[face].cell1 < mesh->mesh_size )  // INTERNAL
            {
                
                uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

                // Also need condition to deal boundary cases
                const T lambda0 = face_lambdas[face];    // dist(cell_center0, face_center) / dist(cell_center0, cell_center1)
                const T lambda1 = 1.0 - lambda0;         // dist(face_center,  cell_center1) / dist(cell_center0, cell_center1)

                // T Uac     = phi.U[block_cell0] * lambda0 + phi.U[ip] * lambda1;
                // T Vac     = phi.V[block_cell0] * lambda0 + phi.V[ip] * lambda1;
                // T Wac     = phi.W[block_cell0] * lambda0 + phi.W[ip] * lambda1;

                const vec<T> dUdXac  =   phi_grad.U[phi_index0] * lambda0 + phi_grad.U[phi_index1] * lambda1;
                const vec<T> dVdXac  =   phi_grad.V[phi_index0] * lambda0 + phi_grad.V[phi_index1] * lambda1;
                const vec<T> dWdXac  =   phi_grad.W[phi_index0] * lambda0 + phi_grad.W[phi_index1] * lambda1;

                T Visac   = effective_viscosity * lambda0 + effective_viscosity * lambda1;
                T VisFace = Visac * face_rlencos[face];
                
                vec<T> Xpn     = mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0];

                // NOTE: ADD other differencing schemes. For now we just use Upwind Differencing Scheme (UDS)

                // call SelectDiffSchemeVector();

                T UFace, VFace, WFace;
                if ( face_mass_fluxes[face] >= 0.0 )
                {
                    UFace  = phi.U[phi_index0];
                    VFace  = phi.V[phi_index0];
                    WFace  = phi.W[phi_index0];
                }
                else
                {
                    UFace  = phi.U[phi_index1];
                    VFace  = phi.V[phi_index1];
                    WFace  = phi.W[phi_index1];
                }

                // explicit higher order convective flux (see eg. eq. 8.16)

                const T fuce = face_mass_fluxes[face] * UFace;
                const T fvce = face_mass_fluxes[face] * VFace;
                const T fwce = face_mass_fluxes[face] * WFace;

                const T sx = face_normals[face].x;
                const T sy = face_normals[face].y;
                const T sz = face_normals[face].z;

                // explicit higher order diffusive flux based on simple uncorrected
                // interpolated cell centred gradients(see eg. eq. 8.19)

                const T fude = Visac * (dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz;
                const T fvde = Visac * (dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz;
                const T fwde = Visac * (dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz;

                // ! implicit lower order (simple upwind)
                // ! convective and diffusive fluxes

                const T fmin = min( face_mass_fluxes[face], 0.0 );
                const T fmax = max( face_mass_fluxes[face], 0.0 );

                const T fuci = fmin * phi.U[phi_index0] + fmax * phi.U[phi_index1];
                const T fvci = fmin * phi.V[phi_index0] + fmax * phi.V[phi_index1];
                const T fwci = fmin * phi.W[phi_index0] + fmax * phi.W[phi_index1];

                const T fudi = VisFace * dot_product( dUdXac , Xpn );
                const T fvdi = VisFace * dot_product( dVdXac , Xpn );
                const T fwdi = VisFace * dot_product( dWdXac , Xpn );

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

                const T blend_u = GammaBlend * ( fuce - fuci );
                const T blend_v = GammaBlend * ( fvce - fvci );
                const T blend_w = GammaBlend * ( fwce - fwci );

                // ! assemble the two source terms
                // ! Is it faster to just write to source term vectors?? Can we vectorize this function??

                // if (mpi_config->particle_flow_rank == 0)
                // {
                //     cout << "cell0 " << mesh->faces[face].cell0 << " cell1 " << mesh->faces[face].cell1 << " phi_grad.U[phi_index0] " << print_vec(phi_grad.U[phi_index0]) << " phi_grad.U[phi_index1] " << print_vec(phi_grad.U[phi_index1])  << endl;
                //     // printf("S_phi.U[%lu] = %.3e fude %.3e fudi %.3e\n", phi_index0, S_phi.U[phi_index0], fude, fudi);
                // }

                S_phi.U[phi_index0] = S_phi.U[phi_index0] - blend_u + fude - fudi;
                S_phi.V[phi_index0] = S_phi.V[phi_index0] - blend_v + fvde - fvdi;
                S_phi.W[phi_index0] = S_phi.W[phi_index0] - blend_w + fwde - fwdi;

                S_phi.U[phi_index1] = S_phi.U[phi_index1] + blend_u - fude + fudi;
                S_phi.V[phi_index1] = S_phi.V[phi_index1] + blend_v - fvde + fvdi;
                S_phi.W[phi_index1] = S_phi.W[phi_index1] + blend_w - fwde + fwdi;

                const T small_epsilon = 1.e-20;
                const T peclet = face_mass_fluxes[face] / face_areas[face] * magnitude(Xpn) / (Visac+small_epsilon);
                pe0 = min( pe0 , peclet );
                pe1 = max( pe1 , peclet );

            }
            else // BOUNDARY
            {

                // Boundary faces
                const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                const uint64_t boundary_type = mesh->boundary_types[boundary_cell];

                if ( boundary_type == INLET )
                {   
                    // if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Encountered INLET boundary \n", mpi_config->rank);

                    // Option to add more inlet region information and functions here.
                    const vec<T> dUdXac = phi_grad.U[block_cell0];
                    const vec<T> dVdXac = phi_grad.V[block_cell0];
                    const vec<T> dWdXac = phi_grad.W[block_cell0];

                    const T UFace = 30.;
                    const T VFace = 0.;
                    const T WFace = 0.;

                    const T Visac = effective_viscosity;

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];
                    const T VisFace  = Visac * face_rlencos[face];

                    // const T fuce = face_mass_fluxes[face] * UFace;
                    // const T fvce = face_mass_fluxes[face] * VFace;
                    // const T fwce = face_mass_fluxes[face] * WFace;

                    const T sx = face_normals[face].x;
                    const T sy = face_normals[face].y;
                    const T sz = face_normals[face].z;

                    const T fude = Visac * (dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz;
                    const T fvde = Visac * (dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz;
                    const T fwde = Visac * (dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz;

                    // const T fmin = min( face_mass_fluxes[face], 0.0 );
                    // const T fmax = max( face_mass_fluxes[face], 0.0 );

                    // const T fuci = fmin * UFace + fmax * phi.U[block_cell0];
                    // const T fvci = fmin * VFace + fmax * phi.V[block_cell0];
                    // const T fwci = fmin * WFace + fmax * phi.W[block_cell0];

                    const T fudi = VisFace * dot_product( dUdXac , Xpn );
                    const T fvdi = VisFace * dot_product( dVdXac , Xpn );
                    const T fwdi = VisFace * dot_product( dWdXac , Xpn );

                    // ! by definition points a boundary normal outwards
                    // ! therefore an inlet results in a mass flux < 0.0

                    const T f = -VisFace + min( face_mass_fluxes[face], 0.0 );

                    A_phi.U[block_cell0] = A_phi.U[block_cell0] - f;
                    S_phi.U[block_cell0] = S_phi.U[block_cell0] - f * UFace + fude - fudi;
                    phi.U[mesh->local_mesh_size + nhalos + boundary_cell] = UFace;

                    A_phi.V[block_cell0] = A_phi.V[block_cell0] - f;
                    S_phi.V[block_cell0] = S_phi.V[block_cell0] - f * VFace + fvde - fvdi;
                    phi.V[mesh->local_mesh_size + nhalos + boundary_cell] = VFace;

                    A_phi.W[block_cell0] = A_phi.W[block_cell0] - f;
                    S_phi.W[block_cell0] = S_phi.W[block_cell0] - f * WFace + fwde - fwdi;
                    phi.W[mesh->local_mesh_size + nhalos + boundary_cell] = WFace;
                }
                else if( boundary_type == OUTLET )
                {
                    // if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Encountered OUTLET boundary \n", mpi_config->rank);

                    const vec<T> dUdXac = phi_grad.U[block_cell0];
                    const vec<T> dVdXac = phi_grad.V[block_cell0];
                    const vec<T> dWdXac = phi_grad.W[block_cell0];

                    const T Visac = effective_viscosity;

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];

                    const T UFace = phi.U[block_cell0];
                    const T VFace = phi.V[block_cell0];
                    const T WFace = phi.W[block_cell0];

                    const T VisFace  = Visac * face_rlencos[face];

                    // const T fuce = face_mass_fluxes[face] * UFace;
                    // const T fvce = face_mass_fluxes[face] * VFace;
                    // const T fwce = face_mass_fluxes[face] * WFace;

                    const T sx = face_normals[face].x;
                    const T sy = face_normals[face].y;
                    const T sz = face_normals[face].z;

                    const T fude = Visac * (dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz;
                    const T fvde = Visac * (dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz;
                    const T fwde = Visac * (dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz;

                    // const T fmin = min( face_mass_fluxes[face], 0.0 );
                    // const T fmax = max( face_mass_fluxes[face], 0.0 );

                    // const T fuci = fmin * UFace + fmax * phi.U[block_cell0];
                    // const T fvci = fmin * VFace + fmax * phi.V[block_cell0];
                    // const T fwci = fmin * WFace + fmax * phi.W[block_cell0];

                    const T fudi = VisFace * dot_product( dUdXac , Xpn );
                    const T fvdi = VisFace * dot_product( dVdXac , Xpn );
                    const T fwdi = VisFace * dot_product( dWdXac , Xpn );

                    // !
                    // ! by definition points a boundary normal outwards
                    // ! therefore an outlet results in a mass flux >= 0.0
                    // !

                    if( face_mass_fluxes[face] < 0.0 )
                    {
                        printf("Error: neg. massflux in outlet\n");
                        face_mass_fluxes[face] = 1e-15;
                    }
                    
                    const T f = -VisFace + min( face_mass_fluxes[face], 0.0 );

                    A_phi.U[block_cell0] = A_phi.U[block_cell0] - f;
                    S_phi.U[block_cell0] = S_phi.U[block_cell0] - f * UFace + fude - fudi;
                    phi.U[mesh->local_mesh_size + nhalos + boundary_cell] = UFace;

                    A_phi.V[block_cell0] = A_phi.V[block_cell0] - f;
                    S_phi.V[block_cell0] = S_phi.V[block_cell0] - f * VFace + fvde - fvdi;
                    phi.V[mesh->local_mesh_size + nhalos + boundary_cell] = VFace;

                    A_phi.W[block_cell0] = A_phi.W[block_cell0] - f;
                    S_phi.W[block_cell0] = S_phi.W[block_cell0] - f * WFace + fwde - fwdi;
                    phi.W[mesh->local_mesh_size + nhalos + boundary_cell] = WFace;
                }
                else if( boundary_type == WALL )
                {
                    // if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Encountered WALL boundary \n", mpi_config->rank);

                    const T UFace = 0.; // Customisable (add regions here later)
                    const T VFace = 0.; // Customisable (add regions here later)
                    const T WFace = 0.; // Customisable (add regions here later)

                    // const vec<T> dUdXac = phi_grad.U[block_cell0];
                    // const vec<T> dVdXac = phi_grad.V[block_cell0];
                    // const vec<T> dWdXac = phi_grad.W[block_cell0];

                    const T Visac = effective_viscosity;

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];

                    const T coef = Visac * face_areas[face] / magnitude(Xpn);

                    vec<T> Up;
                    Up.x = phi.U[block_cell0] - UFace;
                    Up.y = phi.V[block_cell0] - VFace;
                    Up.z = phi.W[block_cell0] - WFace;

                    const T dp = dot_product( Up , face_normals[face] );
                    vec<T> Ut  = Up - dp * face_normals[face];

                    const T Uvel = abs(Ut.x) + abs(Ut.y) + abs(Ut.z);
                    
                    vec<T> force;
                    if ( Uvel > 0.0  )
                    {
                        const T distance_to_face = magnitude(Xpn); // TODO: Correct for different meshes
                        force = face_areas[face] * Visac * Ut / distance_to_face;
                        // Bnd(ib)%shear = force;
                    }
                    else
                    {
                        force = {0.0, 0.0, 0.0};
                        // Bnd(ib)%shear = Force
                    }

                    // if( !initialisation )
                    // {

                        // TotalForce = TotalForce + Force

                        // !               standard
                        // !               implicit
                        // !                  V

                        A_phi.U[block_cell0] = A_phi.U[block_cell0] + coef;
                        A_phi.V[block_cell0] = A_phi.V[block_cell0] + coef;
                        A_phi.W[block_cell0] = A_phi.W[block_cell0] + coef;

                        // !
                        // !                    corr.                     expliciet
                        // !                  impliciet
                        // !                     V                         V

                        S_phi.U[block_cell0] = S_phi.U[block_cell0] + coef*phi.U[block_cell0] - force.x;
                        S_phi.V[block_cell0] = S_phi.V[block_cell0] + coef*phi.V[block_cell0] - force.y;
                        S_phi.W[block_cell0] = S_phi.W[block_cell0] + coef*phi.W[block_cell0] - force.z;
                    // }

                    phi.U[mesh->local_mesh_size + nhalos + boundary_cell] = UFace;
                    phi.V[mesh->local_mesh_size + nhalos + boundary_cell] = VFace;
                    phi.W[mesh->local_mesh_size + nhalos + boundary_cell] = WFace;

                }
            }   
        }
    }

    template<typename T> void FlowSolver<T>::calculate_UVW()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_UVW.\n", mpi_config->rank);


        static double init_time  = 0.0;
        static double flux_time  = 0.0;
        static double setup_time = 0.0;
        static double solve_time = 0.0;


        init_time -= MPI_Wtime(); 

        // Initialise A_phi.U, S_phi.U and Aval vectors to 0.
        #pragma ivdep 
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            A_phi.U[i] = 0.0;
            A_phi.V[i] = 0.0;
            A_phi.W[i] = 0.0;

            S_phi.U[i] = 0.0;
            S_phi.V[i] = 0.0;
            S_phi.W[i] = 0.0;
        }

        // ptr_swap(&phi.U, &old_phi.U);
        // ptr_swap(&phi.V, &old_phi.V);
        // ptr_swap(&phi.W, &old_phi.W);
        // ptr_swap(&phi.P, &old_phi.P);

        init_time += MPI_Wtime();
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
            for ( uint64_t i = 0 ; i < mesh->local_mesh_size + nhalos; i++ ) // Combine with flux?? Or just initialise with these values
            {
                double f = cell_densities[i] * cell_volumes[i] * rdelta;

                S_phi.U[i] += f * phi.U[i];
                S_phi.V[i] += f * phi.V[i];
                S_phi.W[i] += f * phi.W[i];


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

        MPI_Barrier(mpi_config->particle_flow_world); 
        solve_sparse_matrix (phi.U, S_phi.U);
        MPI_Barrier(mpi_config->particle_flow_world); 


        setup_time  -= MPI_Wtime();
        solve_time  += MPI_Wtime();  
        update_sparse_matrix (UVW_URFactor, A_phi.V, phi.V, S_phi.V); 
        setup_time  += MPI_Wtime();
        solve_time  -= MPI_Wtime();     

        MPI_Barrier(mpi_config->particle_flow_world); 
        // solve_sparse_matrix (phi.V, S_phi.V);
        MPI_Barrier(mpi_config->particle_flow_world);  

        setup_time  -= MPI_Wtime();
        solve_time  += MPI_Wtime();  
        update_sparse_matrix (UVW_URFactor, A_phi.W, phi.W, S_phi.W);
        setup_time  += MPI_Wtime();
        solve_time  -= MPI_Wtime();  

        MPI_Barrier(mpi_config->particle_flow_world); 
        // solve_sparse_matrix (phi.W, S_phi.W);
        MPI_Barrier(mpi_config->particle_flow_world);

        
        solve_time += MPI_Wtime();


        if (mpi_config->particle_flow_rank == 0 && timestep_count == 1499)
        {
            printf("TOTAL Init  time: %7.2fs\n", init_time  );
            printf("TOTAL Flux  time: %7.2fs\n", flux_time  );
            printf("TOTAL Setup time: %7.2fs\n", setup_time );
            printf("TOTAL Solve time: %7.2fs\n", solve_time );
        }
    }

    template<typename T> void FlowSolver<T>::calculate_mass_flux()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_mass_flux.\n", mpi_config->rank);

        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

            if ( mesh->faces[face].cell1 < mesh->mesh_size )  // INTERNAL
            {
                uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;


                const T lambda0 = face_lambdas[face];    // dist(cell_center0, face_center) / dist(cell_center0, cell_center1)
                const T lambda1 = 1.0 - lambda0;         // dist(face_center,  cell_center1) / dist(cell_center0, cell_center1)

                const vec<T> dUdXac  =   phi_grad.U[phi_index0] * lambda0 + phi_grad.U[phi_index1] * lambda1;
                const vec<T> dVdXac  =   phi_grad.V[phi_index0] * lambda0 + phi_grad.V[phi_index1] * lambda1;
                const vec<T> dWdXac  =   phi_grad.W[phi_index0] * lambda0 + phi_grad.W[phi_index1] * lambda1;

                vec<T> Xac = mesh->cell_centers[shmem_cell1] * lambda1 + mesh->cell_centers[shmem_cell0] * lambda0;

                const vec<T> delta  = face_centers[face] - Xac;

                const T UFace = phi.U[phi_index1]*lambda1 + phi.U[phi_index0]*lambda0 + dot_product( dUdXac , delta );
                const T VFace = phi.V[phi_index1]*lambda1 + phi.V[phi_index0]*lambda0 + dot_product( dVdXac , delta );
                const T WFace = phi.W[phi_index1]*lambda1 + phi.W[phi_index0]*lambda0 + dot_product( dWdXac , delta );

                const T densityf = cell_densities[phi_index0] * lambda0 + cell_densities[phi_index1] * lambda1;

                face_mass_fluxes[face] = densityf * (UFace * face_normals[face].x +
                                                     VFace * face_normals[face].y +
                                                     WFace * face_normals[face].z );

                const vec<T> Xpac = face_centers[face] - dot_product(face_centers[face] - mesh->cell_centers[shmem_cell0], face_normals[face])*face_normals[face];
                const vec<T> Xnac = face_centers[face] - dot_product(face_centers[face] - mesh->cell_centers[shmem_cell1], face_normals[face])*face_normals[face];


                const vec<T> delp = Xpac - face_centers[face];
                const vec<T> deln = Xnac - face_centers[face];

                const T cell0_P = phi.P[phi_index0] + dot_product( phi_grad.P[phi_index0] , delp );
                const T cell1_P = phi.P[phi_index1] + dot_product( phi_grad.P[phi_index1] , deln );

                const vec<T> Xpn  = Xnac - Xpac;
                const vec<T> Xpn2 = mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0]; 

                const T ApV0 = (A_phi.U[phi_index0] == 0.0) ? 1.0 / A_phi.U[phi_index0] : 0.0;
                const T ApV1 = (A_phi.U[phi_index1] == 0.0) ? 1.0 / A_phi.U[phi_index1] : 0.0;

                T ApV = cell_densities[phi_index0] * ApV0 * lambda0 + cell_densities[phi_index1] * ApV1 * lambda1;

                const T volume_avg = cell_volumes[phi_index0] * lambda0 + cell_volumes[phi_index1] * lambda1;

                ApV  = ApV * face_areas[face] * volume_avg/dot_product(Xpn2, face_normals[face]);

                const T dpx  = ( phi_grad.P[phi_index1].x * lambda1 + phi_grad.P[phi_index0].x * lambda0) * Xpn.x; 
                const T dpy  = ( phi_grad.P[phi_index1].y * lambda1 + phi_grad.P[phi_index0].y * lambda0) * Xpn.y;  
                const T dpz  = ( phi_grad.P[phi_index1].z * lambda1 + phi_grad.P[phi_index0].z * lambda0) * Xpn.z; 

                face_fields[face].cell0 = -ApV;
                face_fields[face].cell1 = -ApV;

                printf("Rank %d face %lu ApV %f face_areas[face] %f volume_avg %f volume0 %f volume1 %f \n", mpi_config->rank, face, -ApV, face_areas[face], volume_avg, cell_volumes[phi_index0], cell_volumes[phi_index1]);

                face_mass_fluxes[face] -= ApV * ((cell0_P - cell1_P) - dpx - dpy - dpz);
            }
            else // BOUNDARY
            {
                // Boundary faces
                const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                const uint64_t boundary_type = mesh->boundary_types[boundary_cell];


                if ( boundary_type == INLET )
                {
                    // Constant inlet values for velocities and densities. Add custom regions laters
                    const vec<T> vel_inward = {30., 0.0, 0.0};
                    const T Din = 1.2;

                    face_mass_fluxes[face] = Din * dot_product( vel_inward, face_normals[face] );

                    S_phi.P[block_cell0] = S_phi.P[block_cell0] - face_mass_fluxes[face];
                }
                else if( boundary_type == OUTLET )
                {
                    // vec<T> delta  = face_centers[face] - mesh->cell_centers[shmem_cell0];

                    const vec<T> vel_outward = { phi.U[block_cell0], 
                                                 phi.V[block_cell0], 
                                                 phi.W[block_cell0] };

                    const T Din = 1.2;

                    face_mass_fluxes[face] = Din * dot_product(vel_outward, face_normals[face]);
                    
                    // !
                    // ! For an outlet face_mass_fluxes must be 0.0 or positive
                    // !
                    if( face_mass_fluxes[face] < 0.0 )
                    {
                        cout << "vel " << print_vec(vel_outward) << " normal " << print_vec(face_normals[face]) << " mass " << face_mass_fluxes[face] << endl;
                        printf("NEGATIVE OUTFLOW %f\n", face_mass_fluxes[face]);
                        face_mass_fluxes[face] = 1e-15;

                        // !
                        // ! to be sure reset add. variables too
                        // !
                        // if( SolveTurbEnergy ) TE(Ncel+ib) = TE(ip)
                        // if( SolveTurbDiss   ) ED(Ncel+ib) = ED(ip)
                        // if( SolveVisc       ) VisEff(Ncel+ib) = VisEff(ip)
                        // if( SolveEnthalpy   ) T(Ncel+ib) = T(ip)
                        // if( SolveScalars    ) SC(Ncel+ib,1:Nscal) = SC(ip,1:Nscal)
                    }
                }
                else if( boundary_type == WALL )
                {
                    face_mass_fluxes[face] = 0.0;   
                }
            }
        }
    }

    template<typename T> void FlowSolver<T>::setup_pressure_matrix()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function setup_pressure_matrix.\n", mpi_config->rank);

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if (mesh->faces[face].cell1 >= mesh->mesh_size)  continue; // Remove when implemented boundary cells. Treat boundary as mesh size

            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

            A_spmatrix.coeffRef(phi_index0, phi_index1) = face_fields[face].cell1;
            A_spmatrix.coeffRef(phi_index1, phi_index0) = face_fields[face].cell0;

            S_phi.P[phi_index0] -= face_mass_fluxes[face];
            S_phi.P[phi_index1] += face_mass_fluxes[face];

            A_phi.P[phi_index0] += face_fields[face].cell1;
            A_phi.P[phi_index1] += face_fields[face].cell0;
        }

        exchange_A_halos ( A_phi.P );
        exchange_S_halos ( S_phi.P );

        MPI_Barrier(mpi_config->particle_flow_world);

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep
        for (uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++)
        {
            A_spmatrix.coeffRef(i, i) = A_phi.P[i];
        }

        A_spmatrix.makeCompressed();
    }

    template<typename T> void FlowSolver<T>::solve_pressure_matrix()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_pressure_matrix.\n", mpi_config->rank);

        Eigen::Map<Eigen::VectorXd> S_phi_vector(S_phi.P, mesh->local_mesh_size + nhalos);
        Eigen::Map<Eigen::VectorXd>   phi_vector(phi.P,   mesh->local_mesh_size + nhalos);

        printf("\tRank %d: Running function solve_pressure_matrix A = (%lu %lu) x = (%lu %lu) b = (%lu %lu).\n", mpi_config->rank, 
                                                                                                                 A_spmatrix.rows(),   A_spmatrix.cols(), 
                                                                                                                 phi_vector.rows(),   phi_vector.cols(),
                                                                                                                 S_phi_vector.rows(), S_phi_vector.cols());

        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_pressure_matrix A = (%lu %lu) x = (%lu %lu) b = (%lu %lu).\n", mpi_config->rank, 
                                                                                                                                         A_spmatrix.rows(),   A_spmatrix.cols(), 
                                                                                                                                         phi_vector.rows(),   phi_vector.cols(),
                                                                                                                                         S_phi_vector.rows(), S_phi_vector.cols());

                                                                                                                                        
        if (mpi_config->particle_flow_rank == 0 && A_spmatrix.cols() < 20 )
        {
            cout << endl << mpi_config->particle_flow_rank << "Phi   : " << endl << phi_vector   << endl;
        }


        eigen_solver.setTolerance(0.1);
        eigen_solver.compute(A_spmatrix);
        phi_vector = eigen_solver.solve(S_phi_vector);

        if (mpi_config->particle_flow_rank == 0 && A_spmatrix.cols() < 20 )
        {
            printf("Timestep %lu\n", timestep_count);
            cout << "A Matrix     : " << endl << Eigen::MatrixXd(A_spmatrix) << endl;

            cout << endl << mpi_config->particle_flow_rank << "S_Phi : " << endl << S_phi_vector << endl;
            cout << endl << mpi_config->particle_flow_rank << "Phi   : " << endl << phi_vector   << endl;
        }

    }

    template<typename T> void FlowSolver<T>::calculate_pressure()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_pressure.\n", mpi_config->rank);

        #pragma ivdep 
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            A_phi.P[i] = 0.0;
            S_phi.P[i] = 0.0;
        }

        calculate_mass_flux();

        setup_pressure_matrix();

        solve_pressure_matrix();
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
        if (FLOW_SOLVER_DEBUG)    printf("Start flow timestep\n");
        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Start flow timestep.\n", mpi_config->rank);

        int comms_timestep = 1;

        static double halo_time = 0.0, grad_time = 0.0;

        grad_time -= MPI_Wtime();

        // Note: After pointer swap, last iterations phi is now in phi.
        // get_phi_gradient ( phi.U, phi_grad.U ); 
        // get_phi_gradient ( phi.V, phi_grad.V ); 
        // get_phi_gradient ( phi.W, phi_grad.W ); 

        get_phi_gradients ();

        grad_time += MPI_Wtime();
        halo_time -= MPI_Wtime();
        exchange_phi_halos();
        halo_time += MPI_Wtime();

        if ((timestep_count % comms_timestep) == 0)  
            update_flow_field();

        if ((timestep_count % 100) == 0)
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

        calculate_pressure();

        if ((timestep_count == 1499) && mpi_config->particle_flow_rank == 0)
        {
            printf("Halo time: %7.2fs\n", halo_time );
            printf("Grad time: %7.2fs\n", grad_time );
        }

        if ((timestep_count % 20) == 0)
        {
            VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh, mpi_config);
            vtk_writer->write_flow_velocities("minicombust", timestep_count, &phi);
        }

        // solve_combustion_equations();
        // update_combustion_fields();
        // solve_turbulence_equations();
        // update_turbulence_fields();
        // solve_flow_equations();
        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Stop flow timestep.\n", mpi_config->rank);
        timestep_count++;
    }

}   // namespace minicombust::flow 