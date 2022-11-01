#include <stdio.h>
#include <limits.h>

#include "flow/FlowSolver.hpp"

#define FLOW_DEBUG 0

using namespace std;

namespace minicombust::flow 
{
    
    template<typename T> void FlowSolver<T>::update_flow_field(bool receive_particle)
    {
        if (FLOW_DEBUG) printf("\tRunning function update_flow_field.\n");
        
        int time_count = 0;
        time_stats[time_count]  -= MPI_Wtime();

        // Gather the size of each rank's cell array
        MPI_Gather(MPI_IN_PLACE,     1, MPI_INT, cell_sizes,      1,    MPI_INT,  mpi_config->rank, mpi_config->world);
        performance_logger.my_papi_start();
        MPI_Gather(MPI_IN_PLACE,     1, MPI_INT, neighbour_sizes, 1,    MPI_INT,  mpi_config->rank, mpi_config->world);

        cell_sizes[mpi_config->rank]      = 0;
        neighbour_sizes[mpi_config->rank] = 0;

        // TODO: Fix for large counts and displacements over INT_MAX
        int cell_size      = 0;
        int neighbour_size = 0;
        for (int rank = 0; rank < mpi_config->world_size; rank++) 
        {
            cell_disps[rank]  = cell_size;
            cell_size        += cell_sizes[rank];

            neighbour_disps[rank] = neighbour_size;
            neighbour_size       += neighbour_sizes[rank];
        }
        resize_cells_arrays(cell_size);
        resize_cells_arrays(neighbour_size);

        static uint64_t unreduced_counts = 0;
        static uint64_t reduced_counts   = 0;
        unreduced_counts += neighbour_size;

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();

        if(cell_size > INT_MAX) 
        {
            printf("ERROR: DISPLACEMENT OVER INT_MAX\n");
            printf("ERROR: DISPLACEMENT OVER INT_MAX\n");
            printf("ERROR: DISPLACEMENT OVER INT_MAX\n");
        }



        // Receive the cells array of each rank
        MPI_Gatherv(MPI_IN_PLACE,    1, MPI_UINT64_T,    cell_indexes,       cell_sizes,      cell_disps,      MPI_UINT64_T, mpi_config->rank, mpi_config->world);
        MPI_Gatherv(MPI_IN_PLACE,    1, MPI_UINT64_T,    neighbour_indexes,  neighbour_sizes, neighbour_disps, MPI_UINT64_T, mpi_config->rank, mpi_config->world);


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();

        #pragma ivdep
        for (int i = 0; i < neighbour_size; i++)
        {
            unordered_neighbours_set.insert(neighbour_indexes[i]);
        }

        unordered_cells_set.erase(MESH_BOUNDARY);
        unordered_neighbours_set.erase(MESH_BOUNDARY);
        reduced_counts += unordered_neighbours_set.size();

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();
        
        // #pragma ivdep
        // for (uint64_t cell: unordered_cells_set)
        // {
        //     // Get 9 cells neighbours below
        //     const uint64_t below_neighbour                = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + DOWN_FACE];
        //     const uint64_t below_left_neighbour           = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + LEFT_FACE];
        //     const uint64_t below_right_neighbour          = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + RIGHT_FACE];
        //     const uint64_t below_front_neighbour          = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + FRONT_FACE];
        //     const uint64_t below_back_neighbour           = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + BACK_FACE];
        //     const uint64_t below_left_front_neighbour     = mesh->cell_neighbours[below_left_neighbour*mesh->faces_per_cell   + FRONT_FACE];
        //     const uint64_t below_left_back_neighbour      = mesh->cell_neighbours[below_left_neighbour*mesh->faces_per_cell   + BACK_FACE];
        //     const uint64_t below_right_front_neighbour    = mesh->cell_neighbours[below_right_neighbour*mesh->faces_per_cell  + FRONT_FACE];
        //     const uint64_t below_right_back_neighbour     = mesh->cell_neighbours[below_right_neighbour*mesh->faces_per_cell  + BACK_FACE];

        //     // Get 9 cells neighbours above
        //     const uint64_t above_neighbour                = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + UP_FACE];
        //     const uint64_t above_left_neighbour           = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + LEFT_FACE];
        //     const uint64_t above_right_neighbour          = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + RIGHT_FACE];
        //     const uint64_t above_front_neighbour          = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + FRONT_FACE];
        //     const uint64_t above_back_neighbour           = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + BACK_FACE];
        //     const uint64_t above_left_front_neighbour     = mesh->cell_neighbours[above_left_neighbour*mesh->faces_per_cell   + FRONT_FACE];
        //     const uint64_t above_left_back_neighbour      = mesh->cell_neighbours[above_left_neighbour*mesh->faces_per_cell   + BACK_FACE];
        //     const uint64_t above_right_front_neighbour    = mesh->cell_neighbours[above_right_neighbour*mesh->faces_per_cell  + FRONT_FACE];
        //     const uint64_t above_right_back_neighbour     = mesh->cell_neighbours[above_right_neighbour*mesh->faces_per_cell  + BACK_FACE];

        //     // Get 8 cells neighbours around
        //     const uint64_t around_left_neighbour          = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + LEFT_FACE];
        //     const uint64_t around_right_neighbour         = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + RIGHT_FACE];
        //     const uint64_t around_front_neighbour         = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + FRONT_FACE];
        //     const uint64_t around_back_neighbour          = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + BACK_FACE];
        //     const uint64_t around_left_front_neighbour    = mesh->cell_neighbours[around_left_neighbour*mesh->faces_per_cell  + FRONT_FACE];
        //     const uint64_t around_left_back_neighbour     = mesh->cell_neighbours[around_left_neighbour*mesh->faces_per_cell  + BACK_FACE];
        //     const uint64_t around_right_front_neighbour   = mesh->cell_neighbours[around_right_neighbour*mesh->faces_per_cell + FRONT_FACE];
        //     const uint64_t around_right_back_neighbour    = mesh->cell_neighbours[around_right_neighbour*mesh->faces_per_cell + BACK_FACE];

            
        //     unordered_neighbours_set.insert(cell);

        //     // Get 9 cells neighbours below
        //     unordered_neighbours_set.insert(below_neighbour);                
        //     unordered_neighbours_set.insert(below_left_neighbour);           
        //     unordered_neighbours_set.insert(below_right_neighbour);          
        //     unordered_neighbours_set.insert(below_front_neighbour);          
        //     unordered_neighbours_set.insert(below_back_neighbour);           
        //     unordered_neighbours_set.insert(below_left_front_neighbour);     
        //     unordered_neighbours_set.insert(below_left_back_neighbour);      
        //     unordered_neighbours_set.insert(below_right_front_neighbour);    
        //     unordered_neighbours_set.insert(below_right_back_neighbour);     

        //     // Get 9 cells neighbours above
        //     unordered_neighbours_set.insert(above_neighbour);                
        //     unordered_neighbours_set.insert(above_left_neighbour);           
        //     unordered_neighbours_set.insert(above_right_neighbour);          
        //     unordered_neighbours_set.insert(above_front_neighbour);          
        //     unordered_neighbours_set.insert(above_back_neighbour);           
        //     unordered_neighbours_set.insert(above_left_front_neighbour);     
        //     unordered_neighbours_set.insert(above_left_back_neighbour);      
        //     unordered_neighbours_set.insert(above_right_front_neighbour);    
        //     unordered_neighbours_set.insert(above_right_back_neighbour);     

        //     // Get 8 cells neighbours around
        //     unordered_neighbours_set.insert(around_left_neighbour);          
        //     unordered_neighbours_set.insert(around_right_neighbour);         
        //     unordered_neighbours_set.insert(around_front_neighbour);         
        //     unordered_neighbours_set.insert(around_back_neighbour);          
        //     unordered_neighbours_set.insert(around_left_front_neighbour);    
        //     unordered_neighbours_set.insert(around_left_back_neighbour);     
        //     unordered_neighbours_set.insert(around_right_front_neighbour);   
        //     unordered_neighbours_set.insert(around_right_back_neighbour); 
        // }

        // printf("FLOW R0 sent %d R512 sent %d reduced %d\n", int_cell_sizes[0], int_cell_sizes[mpi_config->world_size-2], unordered_cells_set.size());

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();

        // set<uint64_t>  neighbours_set(unordered_neighbours_set.begin(), unordered_neighbours_set.end());

        uint64_t count = 0;
        neighbour_size = unordered_neighbours_set.size();
        resize_cells_arrays(neighbour_size);


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();

        for (unordered_set<uint64_t>::iterator cell_it = unordered_neighbours_set.begin(); cell_it != unordered_neighbours_set.end(); ++cell_it)
        {
            neighbour_indexes[count]          = *cell_it;
            int_neighbour_indexes[count]      = (int)*cell_it;
            count++;
        }

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();
        
        // Send size of neighbours of cells back to ranks.
        MPI_Bcast(&neighbour_size, 1, MPI_INT, mpi_config->rank, mpi_config->world);


        static uint64_t neighbour_avg = 0;
        neighbour_avg += neighbour_size;

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();

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
        // MPI_Bcast(neighbour_indexes, neighbour_size,                    MPI_UINT64_T, mpi_config->rank, mpi_config->world);
        MPI_Scatterv(neighbour_indexes, neighbour_sizes, neighbour_disps, MPI_UINT64_T, NULL, 0, MPI_UINT64_T, mpi_config->rank, mpi_config->world);


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();

        // // Create indexed type and send flow terms
        MPI_Datatype MPI_CELL_INDEXES;
        MPI_Type_create_indexed_block(neighbour_size, 1, int_neighbour_indexes, mpi_config->MPI_FLOW_STRUCTURE, &MPI_CELL_INDEXES);
        MPI_Type_commit(&MPI_CELL_INDEXES);

        // Change these from broadcast
        MPI_Bcast(mesh->flow_terms,       1,  MPI_CELL_INDEXES, mpi_config->rank, mpi_config->world);
        MPI_Bcast(mesh->flow_grad_terms,  1,  MPI_CELL_INDEXES, mpi_config->rank, mpi_config->world);


        MPI_Type_free(&MPI_CELL_INDEXES);


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();

        if (receive_particle)
        {
            MPI_Gatherv(MPI_IN_PLACE,    1, mpi_config->MPI_PARTICLE_STRUCTURE, cell_particle_aos,  cell_sizes, cell_disps, mpi_config->MPI_PARTICLE_STRUCTURE,  mpi_config->rank, mpi_config->world);
        }


        
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime();

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