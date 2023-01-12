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

    template<typename T> void FlowSolver<T>::get_neighbour_cells ()
    {
        for (auto& cell_it: cell_particle_field_map[0])
        {
            uint64_t cell = cell_it.first;

            unordered_neighbours_set[0].insert(cell); 

            const uint64_t block_cell_disp  = mesh->local_cells_disp;

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
            if ( around_left_neighbour != MESH_BOUNDARY && !is_halo(around_left_neighbour)  )   // If neighbour isn't edge of mesh and isn't a halo cell
            {
                const uint64_t around_left_front_neighbour    = mesh->cell_neighbours[ (around_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                const uint64_t around_left_back_neighbour     = mesh->cell_neighbours[ (around_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(around_left_front_neighbour);    
                unordered_neighbours_set[0].insert(around_left_back_neighbour);     
            }
            if ( around_right_neighbour != MESH_BOUNDARY && !is_halo(around_right_neighbour) )
            {
                const uint64_t around_right_front_neighbour   = mesh->cell_neighbours[ (around_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell + FRONT_FACE] ;
                const uint64_t around_right_back_neighbour    = mesh->cell_neighbours[ (around_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(around_right_front_neighbour);   
                unordered_neighbours_set[0].insert(around_right_back_neighbour); 
            }
            if ( below_neighbour != MESH_BOUNDARY && !is_halo(below_neighbour) )
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
                if ( below_left_neighbour != MESH_BOUNDARY && !is_halo(below_left_neighbour) )
                {
                    const uint64_t below_left_front_neighbour     = mesh->cell_neighbours[ (below_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + FRONT_FACE] ;
                    const uint64_t below_left_back_neighbour      = mesh->cell_neighbours[ (below_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(below_left_front_neighbour);     
                    unordered_neighbours_set[0].insert(below_left_back_neighbour);      
                }
                if ( below_right_neighbour != MESH_BOUNDARY && !is_halo(below_right_neighbour) )
                {
                    const uint64_t below_right_front_neighbour    = mesh->cell_neighbours[ (below_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                    const uint64_t below_right_back_neighbour     = mesh->cell_neighbours[ (below_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(below_right_front_neighbour);    
                    unordered_neighbours_set[0].insert(below_right_back_neighbour); 
                }
            }
            if ( above_neighbour != MESH_BOUNDARY && !is_halo(above_neighbour) )
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
                if ( above_left_neighbour != MESH_BOUNDARY && !is_halo(above_left_neighbour) )
                {
                    const uint64_t above_left_front_neighbour     = mesh->cell_neighbours[ (above_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + FRONT_FACE] ;
                    const uint64_t above_left_back_neighbour      = mesh->cell_neighbours[ (above_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(above_left_front_neighbour);     
                    unordered_neighbours_set[0].insert(above_left_back_neighbour);      
                }
                if ( above_right_neighbour != MESH_BOUNDARY && !is_halo(above_right_neighbour) )
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
            const uint64_t block_point_disp = mesh->local_points_disp;
            const uint64_t c                = cell_it;
            const uint64_t displaced_c      = cell_it - block_cell_disp;

            if (is_halo(cell_it)) continue;


            // const uint64_t *cell             = mesh->cells + (c - mesh->shmem_cell_disp)*cell_size;
            const vec<T> cell_centre         = mesh->cell_centres[c - mesh->shmem_cell_disp];

            const flow_aos<T> flow_term      = mesh->flow_terms[displaced_c];      
            const flow_aos<T> flow_grad_term = mesh->flow_grad_terms[displaced_c]; 

            // USEFUL ERROR CHECKING!
            // if (flow_term.temp     != mesh->dummy_gas_tem) {printf("INTERP NODAL ERROR: Wrong temp value\n"); exit(1);}
            // if (flow_term.pressure != mesh->dummy_gas_pre) {printf("INTERP NODAL ERROR: Wrong pres value\n"); exit(1);}
            // if (flow_term.vel.x    != mesh->dummy_gas_vel.x) {printf("INTERP NODAL ERROR: Wrong velo value\n"); exit(1);}

            // if (flow_grad_term.temp     != 0.)                 {printf("INTERP NODAL ERROR: Wrong temp grad value\n"); exit(1);}
            // if (flow_grad_term.pressure != 0.)                 {printf("INTERP NODAL ERROR: Wrong pres grad value\n"); exit(1);}
            // if (flow_grad_term.vel.x    != 0.)                 {printf("INTERP NODAL ERROR: Wrong velo grad value\n"); exit(1);}

            resize_nodes_arrays(node_to_position_map.size() + cell_size );

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id      = mesh->cells[(c - mesh->shmem_cell_disp)*mesh->cell_size + n];
                // const uint64_t real_node_id = mesh->cells[(c - mesh->shmem_cell_disp)*mesh->cell_size + n] + block_point_disp;
                // const uint64_t real_node_id = mesh->cells[(c - mesh->shmem_cell_disp)*mesh->cell_size + n];
                const vec<T> direction      = mesh->points[node_id - mesh->shmem_point_disp] - cell_centre;

                if (node_to_position_map.contains(node_id))
                {
                    interp_node_flow_fields[node_to_position_map[node_id]].vel      += (flow_term.vel      + dot_product(flow_grad_term.vel,      direction)) / node_neighbours;
                    interp_node_flow_fields[node_to_position_map[node_id]].pressure += (flow_term.pressure + dot_product(flow_grad_term.pressure, direction)) / node_neighbours;
                    interp_node_flow_fields[node_to_position_map[node_id]].temp     += (flow_term.temp     + dot_product(flow_grad_term.temp,     direction)) / node_neighbours;

                    // if (node_id == 16)
                    //     printf("rank %d node cell %d flow size %f temp %f\n", mpi_config->rank, c, interp_node_flow_fields[node_to_position_map[16]].temp , flow_term.temp);
                }
                else
                {
                    const T boundary_neighbours = node_neighbours - mesh->cells_per_point[node_id - mesh->shmem_point_disp];

                    flow_aos<T> temp_term;
                    temp_term.vel      = ((mesh->dummy_gas_vel * boundary_neighbours) + flow_term.vel      + dot_product(flow_grad_term.vel,      direction)) / node_neighbours;
                    temp_term.pressure = ((mesh->dummy_gas_pre * boundary_neighbours) + flow_term.pressure + dot_product(flow_grad_term.pressure, direction)) / node_neighbours;
                    temp_term.temp     = ((mesh->dummy_gas_tem * boundary_neighbours) + flow_term.temp     + dot_product(flow_grad_term.temp,     direction)) / node_neighbours;
                    
                    interp_node_indexes[node_to_position_map.size()]     = node_id;
                    interp_node_flow_fields[node_to_position_map.size()] = temp_term; // TODO: Overlap getting flow fields?
                    
                    node_to_position_map[node_id] = node_to_position_map.size();

                    // if (node_id == 16)
                    //     printf("rank %d node cell %d flow size %f boundary_neighbours %f temp %f\n", mpi_config->rank, c, interp_node_flow_fields[node_to_position_map[16]].temp , boundary_neighbours, temp_term.temp);
                }
            }
        }

        // Useful for checking errors and comms

        // TODO: Can comment this with properly implemented halo exchange. Need cell neighbours for halo and nodes!
        uint64_t const nsize = node_to_position_map.size();

        #pragma ivdep
        for ( uint64_t i = 0;  i < nsize; i++ ) 
        {
            // if (interp_node_flow_fields[node_it.second].temp     != mesh->dummy_gas_tem)              
            //     {printf("ERROR UPDATE FLOW: Wrong temp value %f at %lu\n", interp_node_flow_fields[node_it.second].temp,           interp_node_indexes[node_it.second]); exit(1);}
            // if (interp_node_flow_fields[node_it.second].pressure != mesh->dummy_gas_pre)              
            //     {printf("ERROR UPDATE FLOW: Wrong pres value %f at %lu\n", interp_node_flow_fields[node_it.second].pressure,       interp_node_indexes[node_it.second]); exit(1);}
            // if (interp_node_flow_fields[node_it.second].vel.x != mesh->dummy_gas_vel.x) 
            //     {printf("ERROR UPDATE FLOW: Wrong velo value {%.10f y z} at %lu\n", interp_node_flow_fields[node_it.second].vel.x, interp_node_indexes[node_it.second]); exit(1);}

            interp_node_flow_fields[i].temp     = mesh->dummy_gas_tem;
            interp_node_flow_fields[i].pressure = mesh->dummy_gas_pre;
            interp_node_flow_fields[i].vel      = mesh->dummy_gas_vel;
        }
    }

    
    template<typename T> void FlowSolver<T>::update_flow_field()
    {
        if (FLOW_DEBUG) printf("\tRunning function update_flow_field.\n");
        
        int time_count = 0;
        time_stats[time_count]  -= MPI_Wtime(); //0
        
        cell_particle_field_map[0].clear();
        unordered_neighbours_set[0].clear();
        node_to_position_map.clear();

        
        uint64_t block_world_size = 1;
        MPI_Iallreduce(MPI_IN_PLACE, &block_world_size, 1, MPI_INT, MPI_SUM, mpi_config->every_one_flow_world[mpi_config->particle_flow_rank], &requests[0]);
        mpi_config->one_flow_world_size[mpi_config->particle_flow_rank] = block_world_size;

        MPI_Barrier(mpi_config->world);
        performance_logger.my_papi_start();

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //1

        MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
        if (block_world_size > 1) 
        {
            MPI_Comm_split(mpi_config->every_one_flow_world[mpi_config->particle_flow_rank], 1, mpi_config->rank, &mpi_config->one_flow_world[mpi_config->particle_flow_rank]);
            MPI_Comm_rank(mpi_config->one_flow_world[mpi_config->particle_flow_rank], &mpi_config->one_flow_rank[mpi_config->particle_flow_rank]);
            MPI_Comm_size(mpi_config->one_flow_world[mpi_config->particle_flow_rank], &mpi_config->one_flow_world_size[mpi_config->particle_flow_rank]);
        }

        MPI_Barrier(mpi_config->world);
        // printf("Flow rank %d world size %d\n", mpi_config->rank, mpi_config->one_flow_world_size[mpi_config->particle_flow_rank]);

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //2

        // Gather and reduce each rank's neighbour indexes into unordered_neighbours_set and neighbour_indexes.
        uint64_t elements;
        // auto resize_cell_indexes _fn = [this] (uint64_t *elements, uint64_t ***indexes) { return resize_cell_indexes(elements, indexes); };
        // MPI_GatherSet ( mpi_config, mesh->num_blocks, unordered_neighbours_set, &neighbour_indexes, &elements, resize_cell_indexes_fn );

        function<void(uint64_t *, uint64_t ***, particle_aos<T> ***)> resize_cell_particles_fn = [this] (uint64_t *elements, uint64_t ***indexes, particle_aos<T> ***cell_particle_fields) { return resize_cell_particle(elements, indexes, cell_particle_fields); };
        MPI_GatherMap (mpi_config, mesh->num_blocks, cell_particle_field_map, &neighbour_indexes, &cell_particle_aos, &elements, async_locks, send_counts, recv_indexes, recv_indexed_fields, requests, resize_cell_particles_fn);

        MPI_Barrier(mpi_config->world);     

        // printf("Flow Rank %d elements %lu \n", mpi_config->rank, unordered_neighbours_set[0].size());

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //3

        get_neighbour_cells ();
        
        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //4
        
        interpolate_to_nodes ();

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //5

        // Send size of reduced neighbours of cells back to ranks.
        uint64_t neighbour_size       = unordered_neighbours_set[0].size();
        uint64_t neighbour_point_size = node_to_position_map.size();
        
        static uint64_t neighbour_avg = 0;
        neighbour_avg += neighbour_size;

        if (neighbour_point_size != 0)  MPI_Bcast(&neighbour_point_size, 1, MPI_UINT64_T, mpi_config->one_flow_rank[mpi_config->particle_flow_rank], mpi_config->one_flow_world[mpi_config->particle_flow_rank]);

        // Send neighbours of cells back to ranks.
        if (neighbour_point_size != 0)  MPI_Ibcast(interp_node_indexes,     neighbour_point_size, MPI_UINT64_T,                   mpi_config->one_flow_rank[mpi_config->particle_flow_rank], mpi_config->one_flow_world[mpi_config->particle_flow_rank], &requests[0]);
        if (neighbour_point_size != 0)  MPI_Ibcast(interp_node_flow_fields, neighbour_point_size, mpi_config->MPI_FLOW_STRUCTURE, mpi_config->one_flow_rank[mpi_config->particle_flow_rank], mpi_config->one_flow_world[mpi_config->particle_flow_rank], &requests[1]);

        if (neighbour_point_size != 0)  MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //6

        
        MPI_Barrier(mpi_config->world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //7

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            if (block_world_size > 1) 
            {
                if (((uint64_t)mpi_config->particle_flow_rank == b))  MPI_Comm_free(&mpi_config->one_flow_world[b]);
            }
        }

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //8

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
        
        time_stats[time_count++] += MPI_Wtime();

        // Check how many flow ranks are seen.
        // int empty_world = (mpi_config->one_flow_world_size[mpi_config->particle_flow_rank] == 1) ? 1 : 0;
        // int total_empty_worlds;
        // MPI_Reduce(&empty_world, &total_empty_worlds, 1, MPI_INT, MPI_SUM, 0, mpi_config->particle_flow_world);
        // if ( mpi_config->particle_flow_rank == 0)
        //     printf("%d flow ranks are all alone, out of %d\n", total_empty_worlds, mpi_config->particle_flow_world_size);


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

                printf("Reduced neighbour count = %f\n", (double)neighbour_avg    / 1500.);
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

    template<typename T> void FlowSolver<T>::timestep()
    {
        if (FLOW_DEBUG) printf("Start flow timestep\n");
        static int count = 0;
        int comms_timestep = 1;
        if ((count % comms_timestep) == 0)  update_flow_field();

        if ((count % 100) == 0)
        {
            double arr_usage = ((double)get_array_memory_usage())  / 1.e9;
            double stl_usage = ((double)get_stl_memory_usage())    / 1.e9 ;
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