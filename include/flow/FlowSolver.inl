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

        MPI_Barrier(mpi_config->world);
        performance_logger.my_papi_start();

        static uint64_t unreduced_counts = 0;
        static uint64_t reduced_counts   = 0;

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //1


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //2

        MPI_GatherSet ( mpi_config, unordered_neighbours_set, neighbour_indexes );
    
        unordered_cells_set.erase(MESH_BOUNDARY);
        unordered_neighbours_set.erase(MESH_BOUNDARY);
        reduced_counts += unordered_neighbours_set.size();

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //3


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //4


        uint64_t count = 0;
        int neighbour_size = unordered_neighbours_set.size();
        resize_cells_arrays(neighbour_size);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();//5

        for (unordered_set<uint64_t>::iterator cell_it = unordered_neighbours_set.begin(); cell_it != unordered_neighbours_set.end(); ++cell_it)
        {
            neighbour_indexes[count]          = *cell_it;
            int_neighbour_indexes[count]      = (int)*cell_it;
            count++;
        }

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();//6
        
        // Send size of neighbours of cells back to ranks.
        MPI_Bcast(&neighbour_size, 1, MPI_INT, mpi_config->rank, mpi_config->world);


        static uint64_t neighbour_avg = 0;
        neighbour_avg += neighbour_size;

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();//7

        int neighbour_disp = 0;
        for (int rank = 0; rank < mpi_config->world_size; rank++)
        {
            neighbour_sizes[rank] = neighbour_size / (mpi_config->world_size - mpi_config->particle_flow_world_size);

            if (rank < ((int)neighbour_size % (mpi_config->world_size - mpi_config->particle_flow_world_size)))
                neighbour_sizes[rank]++;

            neighbour_disps[rank] = neighbour_disp;
            neighbour_disp       += neighbour_sizes[rank];
        }
        neighbour_sizes[mpi_config->rank] = 0;

        // Send neighbours of cells back to ranks.
        MPI_Request scatter_requests[2];
        MPI_Iscatterv(neighbour_indexes, neighbour_sizes, neighbour_disps, MPI_UINT64_T, NULL, 0, MPI_UINT64_T, mpi_config->rank, mpi_config->world, &scatter_requests[0]);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //8

        for (int i = 0; i < neighbour_size; i++)
        {
            neighbour_flow_aos_buffer[i]      = mesh->flow_terms[int_neighbour_indexes[i]];
            neighbour_flow_grad_aos_buffer[i] = mesh->flow_grad_terms[int_neighbour_indexes[i]];
        }
        MPI_Wait(&scatter_requests[0], MPI_STATUS_IGNORE);

        // Change these from broadcast
        MPI_Iscatterv(neighbour_flow_aos_buffer,      neighbour_sizes, neighbour_disps, mpi_config->MPI_FLOW_STRUCTURE, NULL, 0, mpi_config->MPI_FLOW_STRUCTURE, mpi_config->rank, mpi_config->world, &scatter_requests[0]);
        MPI_Iscatterv(neighbour_flow_grad_aos_buffer, neighbour_sizes, neighbour_disps, mpi_config->MPI_FLOW_STRUCTURE, NULL, 0, mpi_config->MPI_FLOW_STRUCTURE, mpi_config->rank, mpi_config->world, &scatter_requests[1]);
        
        
        MPI_Waitall(2, scatter_requests, MPI_STATUSES_IGNORE);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();//9

        if (receive_particle_fields)
        {
            MPI_GatherSet (mpi_config, cell_particle_field_map, cell_particle_aos);
            // MPI_Gatherv(MPI_IN_PLACE,    1, mpi_config->MPI_PARTICLE_STRUCTURE, cell_particle_aos,  cell_sizes, cell_disps, mpi_config->MPI_PARTICLE_STRUCTURE,  mpi_config->rank, mpi_config->world);
        }

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();//10

        unordered_cells_set.clear();
        cell_particle_field_map.clear();
        unordered_neighbours_set.clear();
        
        // PROCESS DUPLICATE CELL INDEXES AND PARTICLE TERMS HERE
        // for (uint64_t i = 0; i < cell_size; i++)
        // {
        //     unordered_cells_set.insert(cells[i]);
        // }
        
        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
        
        time_stats[time_count++] += MPI_Wtime();

        static int timestep_count = 0;
        if (timestep_count++ == 1499)
        {
            double total_time = 0.0;
            for (int i = 0; i < 11; i++)
            {
                total_time += time_stats[i];
            }
            for (int i = 0; i < 11; i++)
            {
                printf("Time stats %d: %f %.2f\n", i, time_stats[i], 100 * time_stats[i] / total_time);
            }
            printf("Total time %f\n", total_time);

            printf("Reduced average count = %f\n",   (double)reduced_counts   / 1500.);
            printf("Unreduced average count = %f\n", (double)unreduced_counts / 1500.);
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
        // solve_combustion_equations();
        // update_combustion_fields();
        // solve_turbulence_equations();
        // update_turbulence_fields();
        // solve_flow_equations();
        if (FLOW_DEBUG) printf("Stop flow timestep\n");
        count++;
    }

}   // namespace minicombust::flow 