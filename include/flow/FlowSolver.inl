#include <stdio.h>

#include "flow/FlowSolver.hpp"

#define FLOW_DEBUG 0

using namespace std;

namespace minicombust::flow 
{
    
    template<typename T> void FlowSolver<T>::update_flow_field(bool receive_particle)
    {
        if (FLOW_DEBUG) printf("\tRunning function update_flow_field.\n");
        // uint64_t cell_sizes[mpi_config->world_size];
        // int      int_cell_sizes[mpi_config->world_size];
        // int      int_cell_displacements[mpi_config->world_size];

        receive_time  -= MPI_Wtime();
        performance_logger.my_papi_start();

        // // Gather the size of each rank's cell array
        // MPI_Gather(MPI_IN_PLACE,     1, MPI_UINT64_T, cell_sizes, 1,    MPI_UINT64_T,  mpi_config->rank, mpi_config->world);


        // cell_sizes[mpi_config->rank] = 0;

        // // TODO: Fix for large counts and displacements over INT_MAX
        // uint64_t cell_size      = 0;
        // for (int i = 0; i < mpi_config->world_size; i++) 
        // {
        //     int_cell_sizes[i]          = (int)cell_sizes[i];
        //     int_cell_displacements[i]  = cell_size;
        //     cell_size                 += cell_sizes[i];
        // }

        // if(cell_size > INT_MAX) 
        // {
        //     printf("ERROR: DISPLACEMENT OVER INT_MAX\n");
        //     printf("ERROR: DISPLACEMENT OVER INT_MAX\n");
        //     printf("ERROR: DISPLACEMENT OVER INT_MAX\n");
        // }
        // uint64_t cells[cell_size];
        // // Receive the cells array of each rank
        // MPI_Gatherv(MPI_IN_PLACE,    1, MPI_UINT64_T,    cells,  int_cell_sizes, int_cell_displacements, MPI_UINT64_T,  mpi_config->rank, mpi_config->world);


        // map<uint64_t, particle_aos<T>>  cell_particle_field_map;
        // if (receive_particle)
        // {
        //     particle_aos<T> neighbour_particle_aos[cell_size];
        //     MPI_Gatherv(MPI_IN_PLACE,    1, mpi_config->MPI_PARTICLE_STRUCTURE, neighbour_particle_aos,  int_cell_sizes, int_cell_displacements, mpi_config->MPI_PARTICLE_STRUCTURE,  mpi_config->rank, mpi_config->world);

        //     for (int i = 0; i < cell_size; i++)
        //     {
        //         cell_particle_field_map[cells[i]].momentum  += neighbour_particle_aos[i].momentum;
        //         cell_particle_field_map[cells[i]].energy    += neighbour_particle_aos[i].energy;
        //         cell_particle_field_map[cells[i]].fuel      += neighbour_particle_aos[i].fuel;
        //     }
        // }
        // else
        // {
        //     for (int i = 0; i < cell_size; i++)
        //     {
        //         cell_particle_field_map[cells[i]].momentum = {0.0, 0.0, 0.0};
        //         cell_particle_field_map[cells[i]].energy    = 0.0;
        //         cell_particle_field_map[cells[i]].fuel      = 0.0;
        //     }
        // }

        // Send size to world
        uint64_t cell_size = 0;
        uint64_t max_cell_size = 0;
        MPI_Allreduce(&cell_size, &max_cell_size, 1, MPI_UINT64_T, MPI_MAX, mpi_config->world);
        max_cell_size *= 2;

        // Create array with world cell arrays
        uint64_t        cells[max_cell_size];
        particle_aos<T> cell_particle_fields[max_cell_size];

        MPI_Recv(cells,                       max_cell_size, MPI_UINT64_T,                       0, 0, mpi_config->world,  MPI_STATUS_IGNORE);
        MPI_Recv(cell_particle_fields,        max_cell_size, mpi_config->MPI_PARTICLE_STRUCTURE, 0, 0, mpi_config->world,  MPI_STATUS_IGNORE);

        receive_time += MPI_Wtime();
        process_time -= MPI_Wtime();

        set<uint64_t>  neighbours_set;
        
        // TODO: Fix find cell neighbours, to get all 26 neighbours without extras!
        #pragma ivdep
        for (int i = 0; i < max_cell_size; i++)
        {
            uint64_t cell = cells[i];
            if (cell == MESH_BOUNDARY)  break;

            // Get 9 cells neighbours below
            const uint64_t below_neighbour                = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + DOWN_FACE];
            const uint64_t below_left_neighbour           = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + LEFT_FACE];
            const uint64_t below_right_neighbour          = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + RIGHT_FACE];
            const uint64_t below_front_neighbour          = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + FRONT_FACE];
            const uint64_t below_back_neighbour           = mesh->cell_neighbours[below_neighbour*mesh->faces_per_cell        + BACK_FACE];
            const uint64_t below_left_front_neighbour     = mesh->cell_neighbours[below_left_neighbour*mesh->faces_per_cell   + FRONT_FACE];
            const uint64_t below_left_back_neighbour      = mesh->cell_neighbours[below_left_neighbour*mesh->faces_per_cell   + BACK_FACE];
            const uint64_t below_right_front_neighbour    = mesh->cell_neighbours[below_right_neighbour*mesh->faces_per_cell  + FRONT_FACE];
            const uint64_t below_right_back_neighbour     = mesh->cell_neighbours[below_right_neighbour*mesh->faces_per_cell  + BACK_FACE];

            // Get 9 cells neighbours above
            const uint64_t above_neighbour                = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + UP_FACE];
            const uint64_t above_left_neighbour           = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + LEFT_FACE];
            const uint64_t above_right_neighbour          = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + RIGHT_FACE];
            const uint64_t above_front_neighbour          = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + FRONT_FACE];
            const uint64_t above_back_neighbour           = mesh->cell_neighbours[above_neighbour*mesh->faces_per_cell        + BACK_FACE];
            const uint64_t above_left_front_neighbour     = mesh->cell_neighbours[above_left_neighbour*mesh->faces_per_cell   + FRONT_FACE];
            const uint64_t above_left_back_neighbour      = mesh->cell_neighbours[above_left_neighbour*mesh->faces_per_cell   + BACK_FACE];
            const uint64_t above_right_front_neighbour    = mesh->cell_neighbours[above_right_neighbour*mesh->faces_per_cell  + FRONT_FACE];
            const uint64_t above_right_back_neighbour     = mesh->cell_neighbours[above_right_neighbour*mesh->faces_per_cell  + BACK_FACE];

            // Get 8 cells neighbours around
            const uint64_t around_left_neighbour          = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + LEFT_FACE];
            const uint64_t around_right_neighbour         = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + RIGHT_FACE];
            const uint64_t around_front_neighbour         = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + FRONT_FACE];
            const uint64_t around_back_neighbour          = mesh->cell_neighbours[cell*mesh->faces_per_cell                   + BACK_FACE];
            const uint64_t around_left_front_neighbour    = mesh->cell_neighbours[around_left_neighbour*mesh->faces_per_cell  + FRONT_FACE];
            const uint64_t around_left_back_neighbour     = mesh->cell_neighbours[around_left_neighbour*mesh->faces_per_cell  + BACK_FACE];
            const uint64_t around_right_front_neighbour   = mesh->cell_neighbours[around_right_neighbour*mesh->faces_per_cell + FRONT_FACE];
            const uint64_t around_right_back_neighbour    = mesh->cell_neighbours[around_right_neighbour*mesh->faces_per_cell + BACK_FACE];

            
            neighbours_set.insert(cell);

            // Get 9 cells neighbours below
            if (below_neighbour              != MESH_BOUNDARY)               neighbours_set.insert(below_neighbour);                
            if (below_left_neighbour         != MESH_BOUNDARY)          neighbours_set.insert(below_left_neighbour);           
            if (below_right_neighbour        != MESH_BOUNDARY)         neighbours_set.insert(below_right_neighbour);          
            if (below_front_neighbour        != MESH_BOUNDARY)         neighbours_set.insert(below_front_neighbour);          
            if (below_back_neighbour         != MESH_BOUNDARY)          neighbours_set.insert(below_back_neighbour);           
            if (below_left_front_neighbour   != MESH_BOUNDARY)    neighbours_set.insert(below_left_front_neighbour);     
            if (below_left_back_neighbour    != MESH_BOUNDARY)     neighbours_set.insert(below_left_back_neighbour);      
            if (below_right_front_neighbour  != MESH_BOUNDARY)   neighbours_set.insert(below_right_front_neighbour);    
            if (below_right_back_neighbour   != MESH_BOUNDARY)    neighbours_set.insert(below_right_back_neighbour);     

            // Get 9 cells neighbours above
            if (above_neighbour              != MESH_BOUNDARY)               neighbours_set.insert(above_neighbour);                
            if (above_left_neighbour         != MESH_BOUNDARY)          neighbours_set.insert(above_left_neighbour);           
            if (above_right_neighbour        != MESH_BOUNDARY)         neighbours_set.insert(above_right_neighbour);          
            if (above_front_neighbour        != MESH_BOUNDARY)         neighbours_set.insert(above_front_neighbour);          
            if (above_back_neighbour         != MESH_BOUNDARY)          neighbours_set.insert(above_back_neighbour);           
            if (above_left_front_neighbour   != MESH_BOUNDARY)    neighbours_set.insert(above_left_front_neighbour);     
            if (above_left_back_neighbour    != MESH_BOUNDARY)     neighbours_set.insert(above_left_back_neighbour);      
            if (above_right_front_neighbour  != MESH_BOUNDARY)   neighbours_set.insert(above_right_front_neighbour);    
            if (above_right_back_neighbour   != MESH_BOUNDARY)    neighbours_set.insert(above_right_back_neighbour);     

            // Get 8 cells neighbours around
            if (around_left_neighbour        != MESH_BOUNDARY)         neighbours_set.insert(around_left_neighbour);          
            if (around_right_neighbour       != MESH_BOUNDARY)        neighbours_set.insert(around_right_neighbour);         
            if (around_front_neighbour       != MESH_BOUNDARY)        neighbours_set.insert(around_front_neighbour);         
            if (around_back_neighbour        != MESH_BOUNDARY)         neighbours_set.insert(around_back_neighbour);          
            if (around_left_front_neighbour  != MESH_BOUNDARY)   neighbours_set.insert(around_left_front_neighbour);    
            if (around_left_back_neighbour   != MESH_BOUNDARY)    neighbours_set.insert(around_left_back_neighbour);     
            if (around_right_front_neighbour != MESH_BOUNDARY)  neighbours_set.insert(around_right_front_neighbour);   
            if (around_right_back_neighbour  != MESH_BOUNDARY)   neighbours_set.insert(around_right_back_neighbour); 



            // for (uint64_t face = 0; face < mesh->faces_per_cell; face++)
            // {
            //     const uint64_t neighbour_id = mesh->cell_neighbours[cell*mesh->faces_per_cell + face];
            //     if (neighbour_id == MESH_BOUNDARY)  continue;

            //     neighbours_set.insert(neighbour_id);
            //     for (uint64_t face2 = 0; face2 < mesh->faces_per_cell; face2++)
            //     {
            //         const uint64_t neighbour_id2 = mesh->cell_neighbours[neighbour_id*mesh->faces_per_cell + face2];
            //         if (neighbour_id2 == MESH_BOUNDARY)  continue;

            //         neighbours_set.insert(neighbour_id2);
            //     }
            // }
        }

        uint64_t count = 0;
        neighbours_size = neighbours_set.size();
        int      int_neighbour_indexes[neighbours_size];

        for (set<uint64_t>::iterator cell_it = neighbours_set.begin(); cell_it != neighbours_set.end(); ++cell_it)
        {
            neighbour_indexes[count]          = *cell_it;
            int_neighbour_indexes[count]      = (int)*cell_it;
            count++;
        }

        process_time += MPI_Wtime();
        bcast_time   -= MPI_Wtime();
        
        // Send size of neighbours of cells back to ranks.
        MPI_Bcast(&neighbours_size,                1,                    MPI_UINT64_T, mpi_config->rank, mpi_config->world);

        // Send neighbours of cells back to ranks.
        MPI_Bcast(neighbour_indexes, neighbours_size,                    MPI_UINT64_T, mpi_config->rank, mpi_config->world);


        // // Create indexed type and send flow terms
        MPI_Datatype MPI_CELL_INDEXES;
        MPI_Type_create_indexed_block((int)neighbours_size, 1, int_neighbour_indexes, mpi_config->MPI_FLOW_STRUCTURE, &MPI_CELL_INDEXES);
        MPI_Type_commit(&MPI_CELL_INDEXES);

        MPI_Bcast(mesh->flow_terms,       1,  MPI_CELL_INDEXES, mpi_config->rank, mpi_config->world);
        MPI_Bcast(mesh->flow_grad_terms,  1,  MPI_CELL_INDEXES, mpi_config->rank, mpi_config->world);

        MPI_Type_free(&MPI_CELL_INDEXES);

        for (int i = 0; i < max_cell_size; i++)
        {
            uint64_t                   cell = cells[i];
            if (cell == MESH_BOUNDARY)  break;

            mesh->particle_energy_rate[cell]      = cell_particle_fields[i].energy;
            mesh->particle_momentum_rate[cell]    = cell_particle_fields[i].momentum;
            mesh->evaporated_fuel_mass_rate[cell] = cell_particle_fields[i].fuel;
        }

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);

        bcast_time   += MPI_Wtime();


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