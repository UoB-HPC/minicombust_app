#include <stdio.h>
#include <limits.h>

#include "flow/FlowSolver.hpp"

#define FLOW_DEBUG 0

using namespace std;

namespace minicombust::flow 
{
    
    template<typename T> void FlowSolver<T>::update_flow_field(bool receive_particle_fields)
    {
        if (FLOW_DEBUG) printf("\tRunning function update_flow_field.\n");
        
        int time_count = 0;
        time_stats[time_count]  -= MPI_Wtime(); //0
        
        cell_particle_field_map.clear();
        unordered_neighbours_set.clear();

        MPI_Barrier(mpi_config->world);
        performance_logger.my_papi_start();

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //1

        // Gather and reduce each rank's neighbour indexes into unordered_neighbours_set and neighbour_indexes.
        auto resize_cell_indexes_fn = [this] (uint64_t elements, uint64_t **indexes) { return resize_cell_indexes(elements, indexes); };
        MPI_GatherSet ( mpi_config, unordered_neighbours_set, neighbour_indexes, resize_cell_indexes_fn );
        
        unordered_neighbours_set.erase(MESH_BOUNDARY);

        
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //2
        
        // Send size of reduced neighbours of cells back to ranks.
        uint64_t neighbour_size = unordered_neighbours_set.size();
        
        static uint64_t neighbour_avg = 0;
        neighbour_avg += neighbour_size;

        MPI_Bcast(&neighbour_size, 1, MPI_UINT64_T, mpi_config->rank, mpi_config->world);


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //3

        // Get size of sizes and displacements
        int neighbour_disp = 0;
        for (uint64_t rank = 0; rank < (uint64_t)mpi_config->world_size; rank++)
        {
            neighbour_sizes[rank] = neighbour_size / (mpi_config->world_size - mpi_config->particle_flow_world_size);

            if (rank < (neighbour_size % (uint64_t)(mpi_config->world_size - mpi_config->particle_flow_world_size)))
                neighbour_sizes[rank]++;
            

            int last_disp         = neighbour_disp;
            neighbour_disps[rank] = neighbour_disp;
            neighbour_disp       += neighbour_sizes[rank];
            if ((neighbour_disp < last_disp) || (neighbour_sizes[rank] < 0)) {
                printf("OVERFLOW!!!!!");
                exit(1);
            }

        }
        neighbour_sizes[mpi_config->rank] = 0;

        // Send neighbours of cells back to ranks.
        MPI_Request scatter_requests[2];
        MPI_Iscatterv(neighbour_indexes, neighbour_sizes, neighbour_disps, MPI_UINT64_T, NULL, 0, MPI_UINT64_T, mpi_config->rank, mpi_config->world, &scatter_requests[0]);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); // 4

        resize_cell_flow(neighbour_size);
        for (uint64_t i = 0; i < neighbour_size; i++)
        {
            neighbour_flow_aos_buffer[i]      = mesh->flow_terms[neighbour_indexes[i]];
        }

        MPI_Wait(&scatter_requests[0], MPI_STATUS_IGNORE);
        MPI_Iscatterv(neighbour_flow_aos_buffer,      neighbour_sizes, neighbour_disps, mpi_config->MPI_FLOW_STRUCTURE, NULL, 0, mpi_config->MPI_FLOW_STRUCTURE, mpi_config->rank, mpi_config->world, &scatter_requests[0]);

        for (uint64_t i = 0; i < neighbour_size; i++)
        {
            neighbour_flow_grad_aos_buffer[i] = mesh->flow_grad_terms[neighbour_indexes[i]];
        }

        MPI_Iscatterv(neighbour_flow_grad_aos_buffer, neighbour_sizes, neighbour_disps, mpi_config->MPI_FLOW_STRUCTURE, NULL, 0, mpi_config->MPI_FLOW_STRUCTURE, mpi_config->rank, mpi_config->world, &scatter_requests[1]);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //5

        if (receive_particle_fields)
        {
            function<void(uint64_t, uint64_t **, particle_aos<T> **)> resize_cell_particles_fn = [this] (uint64_t elements, uint64_t **indexes, particle_aos<T> **cell_particle_fields) { return resize_cell_particle(elements, indexes, cell_particle_fields); };
            MPI_GatherMap (mpi_config, cell_particle_field_map, neighbour_indexes, cell_particle_aos, resize_cell_particles_fn);
        }
        MPI_Waitall(2, scatter_requests, MPI_STATUSES_IGNORE);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //6

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
        
        time_stats[time_count++] += MPI_Wtime();

        static int timestep_count = 0;
        if (timestep_count++ == 1499)
        {
            double total_time = 0.0;
            printf("\nUpdate Flow Field Communuication Timings\n");

            for (int i = 0; i < time_count; i++)
                total_time += time_stats[i];
            for (int i = 0; i < time_count; i++)
                printf("Time stats %d: %f %.2f\n", i, time_stats[i], 100 * time_stats[i] / total_time);
            printf("Total time %f\n", total_time);

            printf("Reduced neighbour count = %f\n", (double)neighbour_avg    / 1500.);
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

            if ( mpi_config->particle_flow_rank == 0 )
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