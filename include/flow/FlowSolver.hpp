#pragma once

#include "utils/utils.hpp"

namespace minicombust::flow 
{
    template<class T>
    class FlowSolver 
    {
        private:

            Mesh<T> *mesh;

            uint64_t *neighbour_indexes;
           
            T turbulence_field;
            T combustion_field;
            T flow_field;

            vector<unordered_set<uint64_t>>                  unordered_neighbours_set;
            vector<unordered_map<uint64_t, particle_aos<T>>> cell_particle_field_map;
            unordered_map<uint64_t, uint64_t>        node_to_position_map;

            int *neighbour_sizes;
            int *neighbour_disps;

            uint64_t    *interp_node_indexes;
            flow_aos<T> *interp_node_flow_fields;

            particle_aos<T> *cell_particle_aos;
            flow_aos<T>    *neighbour_flow_aos_buffer;
            flow_aos<T>    *neighbour_flow_grad_aos_buffer;

        public:
            MPI_Config *mpi_config;
            PerformanceLogger<T> performance_logger;

            uint64_t max_cell_storage;
            size_t cell_index_array_size;
            size_t cell_particle_array_size;
            size_t cell_flow_array_size;

            uint64_t max_point_storage;
            size_t point_index_array_size;
            size_t point_flow_array_size;

            double time_stats[11] = {0.0};

            FlowSolver(MPI_Config *mpi_config, Mesh<T> *mesh) : mesh(mesh), mpi_config(mpi_config)
            {
                const float fraction  = 0.125;
                max_cell_storage      = fraction * mesh->mesh_size;
                max_point_storage     = fraction * mesh->points_size;

                cell_index_array_size    = max_cell_storage * sizeof(uint64_t);
                cell_particle_array_size = max_cell_storage * sizeof(particle_aos<T>);
                cell_flow_array_size     = max_cell_storage * sizeof(flow_aos<T>);

                point_index_array_size   = max_point_storage * sizeof(uint64_t);
                point_flow_array_size    = max_point_storage * sizeof(flow_aos<T>);

                neighbour_indexes     = (uint64_t*)malloc(cell_index_array_size);
                
                neighbour_flow_aos_buffer      = (flow_aos<T> * )    malloc(cell_flow_array_size);
                neighbour_flow_grad_aos_buffer = (flow_aos<T> * )    malloc(cell_flow_array_size);
                cell_particle_aos              = (particle_aos<T> * )malloc(cell_particle_array_size);

                interp_node_indexes      = (uint64_t * )    malloc(point_index_array_size);
                interp_node_flow_fields  = (flow_aos<T> * ) malloc(point_flow_array_size);

                neighbour_sizes    = (int *)     malloc(sizeof(int) * mpi_config->world_size);
                neighbour_disps    = (int *)     malloc(sizeof(int) * mpi_config->world_size);
                
                unordered_neighbours_set.push_back(unordered_set<uint64_t>());
                cell_particle_field_map.push_back(unordered_map<uint64_t, particle_aos<T>>());

                for (uint64_t b = 0; b < mesh->num_blocks; b++)
                {
                    MPI_Comm_split(mpi_config->world, ((uint64_t)mpi_config->particle_flow_rank == b) ? 1 : MPI_UNDEFINED, mpi_config->rank, &mpi_config->every_one_flow_world[b]);
                }

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);
            }

            void resize_cell_indexes(uint64_t *elements, uint64_t ***new_cell_indexes)
            {
                while ( cell_index_array_size < ((uint64_t) *elements * sizeof(uint64_t)) )
                {
                    cell_index_array_size     *= 2;

                    neighbour_indexes     = (uint64_t*) realloc(neighbour_indexes,     cell_index_array_size);
                }

                if (*new_cell_indexes != NULL) **new_cell_indexes = neighbour_indexes;
            }

            void resize_cell_flow (uint64_t elements)
            {
                resize_cell_indexes(elements, NULL);
                while ( cell_flow_array_size < ((size_t) elements * sizeof(flow_aos<T>)) )
                {
                    cell_flow_array_size *= 2;

                    neighbour_flow_aos_buffer      = (flow_aos<T> *)realloc(neighbour_flow_aos_buffer,      cell_flow_array_size);
                    neighbour_flow_grad_aos_buffer = (flow_aos<T> *)realloc(neighbour_flow_grad_aos_buffer, cell_flow_array_size);
                }
            }

            void resize_cell_particle (uint64_t *elements, uint64_t ***new_cell_indexes, particle_aos<T> ***new_cell_particle)
            {
                resize_cell_indexes(elements, new_cell_indexes);
                while ( cell_particle_array_size < ((size_t) *elements * sizeof(particle_aos<T>)) )
                {
                    cell_particle_array_size *= 2;

                    cell_particle_aos = (particle_aos<T> *)realloc(cell_particle_aos,  cell_particle_array_size);
                }

                if (*new_cell_particle != NULL)  **new_cell_particle = cell_particle_aos;
            }

            void resize_nodes_arrays (uint64_t elements)
            {
                while ( max_point_storage < (uint64_t) elements )
                {
                    max_point_storage      *= 2;
                    point_index_array_size *= 2;
                    point_flow_array_size  *= 2;

                    interp_node_indexes     = (uint64_t*)    realloc(interp_node_indexes,     point_index_array_size);
                    interp_node_flow_fields = (flow_aos<T> *)realloc(interp_node_flow_fields, point_flow_array_size);
                }
            }

            size_t get_array_memory_usage ()
            {
                return cell_index_array_size + 2 * cell_flow_array_size + cell_particle_array_size;
            }

            size_t get_stl_memory_usage ()
            {
                return unordered_neighbours_set.size()*sizeof(uint64_t) + cell_particle_field_map.size()*sizeof(particle_aos<T>);
            }

            void interpolate_to_nodes();
            void update_flow_field(bool receive_particle);  // Synchronize point with flow solver
            
            void solve_combustion_equations();
            void update_combustion_fields();

            void solve_turbulence_equations();
            void update_turbulence_fields();

            void solve_flow_equations();

            void timestep();


    }; // class FlowSolver

}   // namespace minicombust::flow 