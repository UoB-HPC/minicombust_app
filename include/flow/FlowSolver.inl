#include <stdio.h>
#include <limits.h>

#include "flow/FlowSolver.hpp"

#define FLOW_DEBUG 0

using namespace std;

namespace minicombust::flow 
{

    template<typename T> void FlowSolver<T>::interpolate_to_nodes ()
    {
        const uint64_t cell_size = mesh->cell_size;
        double node_neighbours   = 8;

        uint64_t neighbour_size = unordered_neighbours_set[0].size();

        // Process the allocation of cell fields (NOTE: Imperfect solution near edges. Fix by doing interpolation on flow side.)
        #pragma ivdep
        for (uint64_t i = 0; i < neighbour_size; i++)
        {
            const uint64_t c = neighbour_indexes[i];

            const uint64_t *cell             = mesh->cells + c*cell_size;
            const vec<T> cell_centre         = mesh->cell_centres[c];

            const flow_aos<T> flow_term      = mesh->flow_terms[i];      
            const flow_aos<T> flow_grad_term = mesh->flow_grad_terms[i]; 

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
                const uint64_t node_id = cell[n];
                const vec<T> direction             = mesh->points[node_id] - cell_centre;

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
                    const T boundary_neighbours = node_neighbours - mesh->cells_per_point[node_id];

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
        for (auto& node_it: node_to_position_map)
        {
            // if (interp_node_flow_fields[node_it.second].temp     != mesh->dummy_gas_tem)              
            //     {printf("ERROR UPDATE FLOW: Wrong temp value %f at %lu\n", interp_node_flow_fields[node_it.second].temp,           interp_node_indexes[node_it.second]); exit(1);}
            // if (interp_node_flow_fields[node_it.second].pressure != mesh->dummy_gas_pre)              
            //     {printf("ERROR UPDATE FLOW: Wrong pres value %f at %lu\n", interp_node_flow_fields[node_it.second].pressure,       interp_node_indexes[node_it.second]); exit(1);}
            // if (interp_node_flow_fields[node_it.second].vel.x != mesh->dummy_gas_vel.x) 
            //     {printf("ERROR UPDATE FLOW: Wrong velo value {%.10f y z} at %lu\n", interp_node_flow_fields[node_it.second].vel.x, interp_node_indexes[node_it.second]); exit(1);}

            interp_node_flow_fields[node_it.second].temp = mesh->dummy_gas_tem;
            interp_node_flow_fields[node_it.second].pressure = mesh->dummy_gas_pre;
            interp_node_flow_fields[node_it.second].vel = mesh->dummy_gas_vel;
        }
    }
    
    template<typename T> void FlowSolver<T>::update_flow_field(bool receive_particle_fields)
    {
        if (FLOW_DEBUG) printf("\tRunning function update_flow_field.\n");
        
        int time_count = 0;
        time_stats[time_count]  -= MPI_Wtime(); //0
        
        cell_particle_field_map[0].clear();
        unordered_neighbours_set[0].clear();
        node_to_position_map.clear();

        MPI_Barrier(mpi_config->world);
        performance_logger.my_papi_start();

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //1

        MPI_Comm_split(mpi_config->every_one_flow_world[mpi_config->particle_flow_rank], 1, mpi_config->rank, &mpi_config->one_flow_world[mpi_config->particle_flow_rank]);
        MPI_Comm_rank(mpi_config->one_flow_world[mpi_config->particle_flow_rank], &mpi_config->one_flow_rank[mpi_config->particle_flow_rank]);
        MPI_Comm_size(mpi_config->one_flow_world[mpi_config->particle_flow_rank], &mpi_config->one_flow_world_size[mpi_config->particle_flow_rank]);

        MPI_Barrier(mpi_config->world);


        // printf("Flow rank %d world size %d\n", mpi_config->rank, mpi_config->one_flow_world_size[mpi_config->particle_flow_rank]);

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //2

        // Gather and reduce each rank's neighbour indexes into unordered_neighbours_set and neighbour_indexes.
        uint64_t elements;
        auto resize_cell_indexes_fn = [this] (uint64_t *elements, uint64_t ***indexes) { return resize_cell_indexes(elements, indexes); };
        MPI_GatherSet ( mpi_config, mesh->num_blocks, unordered_neighbours_set, &neighbour_indexes, &elements, resize_cell_indexes_fn );

        // printf("Flow Rank %d elements %lu \n", mpi_config->rank, unordered_neighbours_set[0].size());
        
        unordered_neighbours_set[0].erase(MESH_BOUNDARY);

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //3
        
        interpolate_to_nodes ();
        // Send size of reduced neighbours of cells back to ranks.
        uint64_t neighbour_size       = unordered_neighbours_set[0].size();
        uint64_t neighbour_point_size = node_to_position_map.size();
        
        static uint64_t neighbour_avg = 0;
        neighbour_avg += neighbour_size;

        // MPI_Bcast(&neighbour_size, 1, MPI_UINT64_T, mpi_config->one_flow_rank, mpi_config->one_flow_world);
        if (neighbour_point_size != 0)  MPI_Bcast(&neighbour_point_size, 1, MPI_UINT64_T, mpi_config->one_flow_rank[mpi_config->particle_flow_rank], mpi_config->one_flow_world[mpi_config->particle_flow_rank]);

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //4

        // Get size of sizes and displacements
        // int neighbour_disp = 0;
        // for (uint64_t rank = 0; rank < (uint64_t)mpi_config->one_flow_world_size; rank++)
        // {
        //     neighbour_sizes[rank] = neighbour_size / (mpi_config->one_flow_world_size - mpi_config->particle_flow_world_size);

        //     if (rank < (neighbour_size % (uint64_t)(mpi_config->one_flow_world_size - mpi_config->particle_flow_world_size)))
        //         neighbour_sizes[rank]++;

        //     int last_disp         = neighbour_disp;
        //     neighbour_disps[rank] = neighbour_disp;
        //     neighbour_disp       += neighbour_sizes[rank];

        //     if ((neighbour_disp < last_disp) || (neighbour_sizes[rank] < 0)) {
        //         printf("OVERFLOW!!!!!");
        //         exit(1);
        //     }

        // }
        // neighbour_sizes[mpi_config->one_flow_rank] = 0;

        // Send neighbours of cells back to ranks.
        MPI_Request scatter_requests[2];
        // MPI_Iscatterv(neighbour_indexes, neighbour_sizes, neighbour_disps, MPI_UINT64_T, NULL, 0, MPI_UINT64_T, mpi_config->one_flow_rank, mpi_config->one_flow_world, &scatter_requests[0]);
        if (neighbour_point_size != 0) MPI_Ibcast(interp_node_indexes, neighbour_point_size, MPI_UINT64_T, mpi_config->one_flow_rank[mpi_config->particle_flow_rank], mpi_config->one_flow_world[mpi_config->particle_flow_rank], &scatter_requests[0]);


        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); // 5

        // resize_cell_flow(neighbour_size);
        // for (uint64_t i = 0; i < neighbour_size; i++)
        // {
        //     neighbour_flow_aos_buffer[i]      = mesh->flow_terms[neighbour_indexes[i]];
        // }

        // MPI_Wait(&scatter_requests[0], MPI_STATUS_IGNORE);
        // MPI_Iscatterv(neighbour_flow_aos_buffer,      neighbour_sizes, neighbour_disps, mpi_config->MPI_FLOW_STRUCTURE, NULL, 0, mpi_config->MPI_FLOW_STRUCTURE, mpi_config->one_flow_rank, mpi_config->one_flow_world, &scatter_requests[0]);
        if (neighbour_point_size != 0) MPI_Ibcast(interp_node_flow_fields, neighbour_point_size, mpi_config->MPI_FLOW_STRUCTURE, mpi_config->one_flow_rank[mpi_config->particle_flow_rank], mpi_config->one_flow_world[mpi_config->particle_flow_rank], &scatter_requests[1]);

        // for (uint64_t i = 0; i < neighbour_size; i++)
        // {
        //     neighbour_flow_grad_aos_buffer[i] = mesh->flow_grad_terms[neighbour_indexes[i]];
        // }

        // MPI_Iscatterv(neighbour_flow_grad_aos_buffer, neighbour_sizes, neighbour_disps, mpi_config->MPI_FLOW_STRUCTURE, NULL, 0, mpi_config->MPI_FLOW_STRUCTURE, mpi_config->one_flow_rank, mpi_config->one_flow_world, &scatter_requests[1]);
        if (neighbour_point_size != 0) MPI_Waitall(2, scatter_requests, MPI_STATUSES_IGNORE);

        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //6

        if (receive_particle_fields)
        {
            function<void(uint64_t *, uint64_t ***, particle_aos<T> ***)> resize_cell_particles_fn = [this] (uint64_t *elements, uint64_t ***indexes, particle_aos<T> ***cell_particle_fields) { return resize_cell_particle(elements, indexes, cell_particle_fields); };
            MPI_GatherMap (mpi_config, mesh->num_blocks, cell_particle_field_map, &neighbour_indexes, &cell_particle_aos, &elements, resize_cell_particles_fn);
        }
        
        MPI_Barrier(mpi_config->world);
        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //7

        for (uint64_t b = 0; b < mesh->num_blocks; b++)
        {
            if (((uint64_t)mpi_config->particle_flow_rank == b))  MPI_Comm_free(&mpi_config->one_flow_world[b]);
        }

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
        if ((count % comms_timestep) == 0)  update_flow_field(count > 0);

        if ((count % 100) == 0)
        {
            double arr_usage = ((double)get_array_memory_usage()) / 1.e9;
            double stl_usage = ((double)get_stl_memory_usage())   / 1.e9 ;
            double arr_usage_total = arr_usage;
            double stl_usage_total = stl_usage;


            if ( unordered_neighbours_set[0].size() != 0 )
            {
                printf("                Flow     Array mem (GB) %8.3f Array mem total (GB) %8.3f STL mem (GB) %8.3f STL mem total (GB) %8.3f\n", arr_usage, arr_usage_total, stl_usage, stl_usage_total);
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