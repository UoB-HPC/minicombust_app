#pragma once

#include "utils/utils.hpp"

namespace minicombust::flow 
{
    template<class T>
    class FlowSolver 
    {
        private:

            Mesh<T> *mesh;

            uint64_t *cell_indexes;
            uint64_t *neighbour_indexes;
            int      *int_neighbour_indexes;
           
            T turbulence_field;
            T combustion_field;
            T flow_field;

            unordered_set<uint64_t>  unordered_cells_set;
            unordered_set<uint64_t>  unordered_neighbours_set;

            unordered_map<uint64_t, particle_aos<T>> cell_particle_field_map;

            int *cell_sizes;
            int *cell_disps;

            int *neighbour_sizes;
            int *neighbour_disps;

            particle_aos<T> *cell_particle_aos;
            flow_aos<T>    *neighbour_flow_aos_buffer;
            flow_aos<T>    *neighbour_flow_grad_aos_buffer;

            uint64_t max_cell_storage;
            size_t cell_index_array_size;
            size_t cell_particle_array_size;
            size_t cell_flow_array_size;

        public:
            MPI_Config *mpi_config;
            PerformanceLogger<T> performance_logger;

            double time_stats[11] = {0.0};

            FlowSolver(MPI_Config *mpi_config, Mesh<T> *mesh) : mesh(mesh), mpi_config(mpi_config)
            {
                const float fraction = 0.125;
                max_cell_storage     = fraction * mesh->mesh_size;

                cell_index_array_size    = max_cell_storage * sizeof(uint64_t);
                cell_particle_array_size = max_cell_storage * sizeof(particle_aos<T>);
                cell_flow_array_size     = max_cell_storage * sizeof(flow_aos<T>);

                cell_indexes          = (uint64_t*)malloc(cell_index_array_size);
                neighbour_indexes     = (uint64_t*)malloc(cell_index_array_size);
                int_neighbour_indexes = (int*)     malloc(cell_index_array_size);
                
                neighbour_flow_aos_buffer      = (flow_aos<T> * )    malloc(cell_flow_array_size);
                neighbour_flow_grad_aos_buffer = (flow_aos<T> * )    malloc(cell_flow_array_size);
                cell_particle_aos              = (particle_aos<T> * )malloc(cell_particle_array_size);

                cell_sizes         = (int *)     malloc(sizeof(int) * mpi_config->world_size);
                cell_disps         = (int *)     malloc(sizeof(int) * mpi_config->world_size);
                neighbour_sizes    = (int *)     malloc(sizeof(int) * mpi_config->world_size);
                neighbour_disps    = (int *)     malloc(sizeof(int) * mpi_config->world_size);

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);
            }

            void resize_cells_arrays(int elements)
            {
                if ( max_cell_storage < (uint64_t) elements )
                {
                    max_cell_storage          *= 2;
                    cell_index_array_size     *= 2;
                    cell_particle_array_size  *= 2;
                    cell_flow_array_size      *= 2;

                    cell_indexes          = (uint64_t*) realloc(cell_indexes,          cell_index_array_size);
                    neighbour_indexes     = (uint64_t*) realloc(neighbour_indexes,     cell_index_array_size);
                    int_neighbour_indexes = (int*)      realloc(int_neighbour_indexes, cell_index_array_size);

                    neighbour_flow_aos_buffer       = (flow_aos<T> *)realloc(neighbour_flow_aos_buffer,      cell_flow_array_size);
                    neighbour_flow_grad_aos_buffer  = (flow_aos<T> *)realloc(neighbour_flow_grad_aos_buffer, cell_flow_array_size);
                    cell_particle_aos               = (particle_aos<T> *)realloc(cell_particle_aos,              cell_particle_array_size);
                }
            }

            void update_flow_field(bool receive_particle);  // Synchronize point with flow solver
            
            void solve_combustion_equations();
            void update_combustion_fields();

            void solve_turbulence_equations();
            void update_turbulence_fields();

            void solve_flow_equations();

            void timestep();


    }; // class FlowSolver

}   // namespace minicombust::flow 