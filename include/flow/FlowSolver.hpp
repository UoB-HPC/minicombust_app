#pragma once

#include "utils/utils.hpp"

namespace minicombust::flow 
{
    template<class T>
    class FlowSolver 
    {
        private:

            Mesh<T> *mesh;

            vector<uint64_t *>        neighbour_indexes;
            vector<particle_aos<T> *> cell_particle_aos;


            T turbulence_field;
            T combustion_field;
            T flow_field;

            vector<unordered_set<uint64_t>>             unordered_neighbours_set;
            unordered_set<uint64_t>                     new_cells_set;
            vector<unordered_map<uint64_t, uint64_t>>   cell_particle_field_map;
            unordered_map<uint64_t, uint64_t>           node_to_position_map;
            vector<unordered_set<uint64_t>>             local_particle_node_sets;

            uint64_t    *interp_node_indexes;
            flow_aos<T> *interp_node_flow_fields;
            uint64_t    *send_buffers_interp_node_indexes;
            flow_aos<T> *send_buffers_interp_node_flow_fields;


            bool *async_locks;
            
            uint64_t         *send_counts;         
            uint64_t        **recv_indexes;        
            particle_aos<T> **recv_indexed_fields; 

            vector<int> ranks;
            int      *elements;
            uint64_t *element_disps;

            MPI_Request bcast_request;
            vector<MPI_Status>  statuses;
            vector<MPI_Request> send_requests;
            vector<MPI_Request> recv_requests;

            Flow_Logger logger;


        public:
            MPI_Config *mpi_config;
            PerformanceLogger<T> performance_logger;

            vector<size_t> cell_index_array_size;
            vector<size_t> cell_particle_array_size;


            size_t cell_flow_array_size;

            size_t node_index_array_size;
            size_t node_flow_array_size;

            size_t send_buffers_node_index_array_size;
            size_t send_buffers_node_flow_array_size;

            uint64_t max_storage;

            double time_stats[11] = {0.0};

            const MPI_Status empty_mpi_status = { 0, 0, 0, 0, 0};


            FlowSolver(MPI_Config *mpi_config, Mesh<T> *mesh) : mesh(mesh), mpi_config(mpi_config)
            {
                const float fraction  = 0.125;
                max_storage           = fraction * mesh->local_mesh_size;

                int particle_ranks = mpi_config->world_size - mpi_config->particle_flow_world_size;

                // Compute array sizes
                cell_index_array_size.push_back(max_storage    * sizeof(uint64_t));
                cell_particle_array_size.push_back(max_storage * sizeof(particle_aos<T>));

                node_index_array_size   = max_storage * sizeof(uint64_t);
                node_flow_array_size    = max_storage * sizeof(flow_aos<T>);

                send_buffers_node_index_array_size   = max_storage * sizeof(uint64_t);
                send_buffers_node_flow_array_size    = max_storage * sizeof(flow_aos<T>);

                async_locks = (bool*)malloc(4 * mesh->num_blocks * sizeof(bool));
                
                send_counts    =              (uint64_t*) malloc(mesh->num_blocks * sizeof(uint64_t));
                recv_indexes   =             (uint64_t**) malloc(mesh->num_blocks * sizeof(uint64_t*));
                recv_indexed_fields = (particle_aos<T>**) malloc(mesh->num_blocks * sizeof(particle_aos<T>*));

                elements        = (int*)malloc(particle_ranks          * sizeof(int));
                element_disps   = (uint64_t*)malloc((particle_ranks+1) * sizeof(uint64_t));

                // Allocate arrays
                neighbour_indexes.push_back((uint64_t*)         malloc(cell_index_array_size[0]));
                cell_particle_aos.push_back((particle_aos<T> * )malloc(cell_particle_array_size[0]));

                local_particle_node_sets.push_back(unordered_set<uint64_t>());

                interp_node_indexes      = (uint64_t * )    malloc(node_index_array_size);
                interp_node_flow_fields  = (flow_aos<T> * ) malloc(node_flow_array_size);

                send_buffers_interp_node_indexes      = (uint64_t * )    malloc(send_buffers_node_index_array_size);
                send_buffers_interp_node_flow_fields  = (flow_aos<T> * ) malloc(send_buffers_node_flow_array_size);

                unordered_neighbours_set.push_back(unordered_set<uint64_t>());
                cell_particle_field_map.push_back(unordered_map<uint64_t, uint64_t>());

                send_requests.push_back( MPI_REQUEST_NULL );
                send_requests.push_back( MPI_REQUEST_NULL );
                recv_requests.push_back( MPI_REQUEST_NULL );
                recv_requests.push_back( MPI_REQUEST_NULL );
                statuses.push_back( empty_mpi_status );

                memset(&logger, 0, sizeof(Flow_Logger));

                // Array sizes
                
                uint64_t total_node_index_array_size              = node_index_array_size;
                uint64_t total_node_flow_array_size               = node_flow_array_size;
                uint64_t total_send_buffers_node_index_array_size = send_buffers_node_index_array_size;
                uint64_t total_send_buffers_node_flow_array_size  = send_buffers_node_flow_array_size;

                // STL sizes
                uint64_t total_unordered_neighbours_set_size   = unordered_neighbours_set[0].size() * sizeof(uint64_t) ;
                uint64_t total_cell_particle_field_map_size    = cell_particle_field_map[0].size()  * sizeof(uint64_t);
                uint64_t total_node_to_position_map_size       = node_to_position_map.size()        * sizeof(uint64_t);
                uint64_t total_mpi_requests_size               = recv_requests.size() * send_requests.size() * sizeof(MPI_Request);
                uint64_t total_mpi_statuses_size               = statuses.size()                             * sizeof(MPI_Status);
                uint64_t total_new_cells_size                  = new_cells_set.size()                        * sizeof(uint64_t);
                uint64_t total_ranks_size                      = ranks.size()                                * sizeof(uint64_t);
                
                uint64_t total_local_particle_node_sets_size   = 0;
                for ( uint64_t i = 0; i < local_particle_node_sets.size(); i++ )
                    total_local_particle_node_sets_size += local_particle_node_sets[i].size() * sizeof(uint64_t);

                uint64_t total_cell_index_array_size    = 0;
                uint64_t total_cell_particle_array_size = 0;
                for ( uint64_t i = 0; i < cell_index_array_size.size(); i++ )
                {
                    total_cell_index_array_size    = cell_index_array_size[i];
                    total_cell_particle_array_size = cell_particle_array_size[i];
                }

                uint64_t total_memory_usage = get_array_memory_usage() + get_stl_memory_usage();
                if (mpi_config->particle_flow_rank == 0)
                {
                    MPI_Reduce(MPI_IN_PLACE, &total_memory_usage,                           1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);

                    MPI_Reduce(MPI_IN_PLACE, &total_cell_index_array_size,                  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_particle_array_size,               1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_node_index_array_size,                  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_node_flow_array_size,                   1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_send_buffers_node_index_array_size,     1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_send_buffers_node_flow_array_size,      1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_unordered_neighbours_set_size,          1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_particle_field_map_size,           1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_node_to_position_map_size,              1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_mpi_requests_size,                      1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_mpi_statuses_size,                      1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_new_cells_size,                         1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_ranks_size,                             1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_local_particle_node_sets_size,          1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);


                    printf("Flow solver storage requirements (%d processes) : \n", mpi_config->particle_flow_world_size);
                    printf("\ttotal_cell_index_array_size                               (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_index_array_size              / 1000000.0, (float) total_cell_index_array_size              / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_cell_particle_array_size                            (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_array_size           / 1000000.0, (float) total_cell_particle_array_size           / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_node_index_array_size                               (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_index_array_size              / 1000000.0, (float) total_node_index_array_size              / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_node_flow_array_size                                (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_flow_array_size               / 1000000.0, (float) total_node_flow_array_size               / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_send_buffers_node_index_array_size                  (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_send_buffers_node_index_array_size / 1000000.0, (float) total_send_buffers_node_index_array_size / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_send_buffers_node_flow_array_size                   (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_send_buffers_node_flow_array_size  / 1000000.0, (float) total_send_buffers_node_flow_array_size  / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_unordered_neighbours_set_size       (STL set)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_unordered_neighbours_set_size      / 1000000.0, (float) total_unordered_neighbours_set_size      / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_cell_particle_field_map_size        (STL map)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_field_map_size       / 1000000.0, (float) total_cell_particle_field_map_size       / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_node_to_position_map_size           (STL map)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_to_position_map_size          / 1000000.0, (float) total_node_to_position_map_size          / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_mpi_requests_size                   (STL vector)    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_mpi_requests_size                  / 1000000.0, (float) total_mpi_requests_size                  / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_mpi_statuses_size                   (STL vector)    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_mpi_statuses_size                  / 1000000.0, (float) total_mpi_statuses_size                  / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_ranks_size                          (STL vector)    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_ranks_size                         / 1000000.0, (float) total_ranks_size                         / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_new_cells_size                      (STL set)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_new_cells_size                     / 1000000.0, (float) total_new_cells_size                     / (1000000.0 * mpi_config->world_size));
                    printf("\ttotal_local_particle_node_sets_size       (STL set)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n\n"  , (float) total_local_particle_node_sets_size      / 1000000.0, (float) total_local_particle_node_sets_size      / (1000000.0 * mpi_config->world_size));
                    
                    printf("\tFlow solver size                                          (TOTAL %12.2f MB) (AVG %.2f MB) \n\n"  , (float)total_memory_usage                                          /1000000.0,  (float)total_memory_usage / (1000000.0 * mpi_config->particle_flow_world_size));
                }
                else
                {
                    MPI_Reduce(&total_memory_usage,                       nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);

                    MPI_Reduce(&total_cell_index_array_size,              nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_cell_particle_array_size,           nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_node_index_array_size,              nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_node_flow_array_size,               nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_send_buffers_node_index_array_size, nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_send_buffers_node_flow_array_size,  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_unordered_neighbours_set_size,      nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_cell_particle_field_map_size,       nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_node_to_position_map_size,          nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_mpi_requests_size,                  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_mpi_statuses_size,                  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_new_cells_size,                     nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_ranks_size,                         nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_local_particle_node_sets_size,      nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);

                }

                MPI_Barrier(mpi_config->world);

                // for (uint64_t b = 0; b < mesh->num_blocks; b++)
                // {
                //     if ((uint64_t)mpi_config->particle_flow_rank == b)
                //     {
                //         MPI_Comm_split(mpi_config->world, 1, mpi_config->rank, &mpi_config->every_one_flow_world[b]);
                //         MPI_Comm_rank(mpi_config->every_one_flow_world[b], &mpi_config->every_one_flow_rank[b]);
                //         MPI_Comm_size(mpi_config->every_one_flow_world[b], &mpi_config->every_one_flow_world_size[b]);
                //     }
                //     else
                //     {
                //         MPI_Comm_split(mpi_config->world, MPI_UNDEFINED, mpi_config->rank, &mpi_config->every_one_flow_world[b]);
                //     }
                // }

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);
            }

            // void resize_cell_indexes(uint64_t *elements, uint64_t ***new_cell_indexes)
            // {
            //     while ( cell_index_array_size < ((uint64_t) *elements * sizeof(uint64_t)) )
            //     {
            //         cell_index_array_size     *= 2;

            //         neighbour_indexes     = (uint64_t*) realloc(neighbour_indexes, cell_index_array_size);
            //     }

            //     if (new_cell_indexes != NULL) **new_cell_indexes = neighbour_indexes;
            // }

            // void resize_cell_flow (uint64_t elements)
            // {
            //     resize_cell_indexes(elements, NULL);
            //     while ( cell_flow_array_size < ((size_t) elements * sizeof(flow_aos<T>)) )
            //     {
            //         cell_flow_array_size *= 2;

            //         neighbour_flow_aos_buffer      = (flow_aos<T> *)realloc(neighbour_flow_aos_buffer,      cell_flow_array_size);
            //         neighbour_flow_grad_aos_buffer = (flow_aos<T> *)realloc(neighbour_flow_grad_aos_buffer, cell_flow_array_size);
            //     }
            // }

            // void resize_cell_particle (uint64_t *elements, uint64_t ***new_cell_indexes, particle_aos<T> ***new_cell_particle)
            // {
            //     resize_cell_indexes(elements, new_cell_indexes);
            //     while ( cell_particle_array_size < ((size_t) *elements * sizeof(particle_aos<T>)) )
            //     {
            //         cell_particle_array_size *= 2;

            //         cell_particle_aos = (particle_aos<T> *)realloc(cell_particle_aos,  cell_particle_array_size);
            //     }

            //     if (new_cell_particle != NULL)  **new_cell_particle = cell_particle_aos;
            // }


            void resize_cell_particle (uint64_t elements, uint64_t index)
            {
                while ( cell_index_array_size[index] < ((size_t) elements * sizeof(uint64_t)) )
                {
                    cell_index_array_size[index]    *= 2;
                    cell_particle_array_size[index] *= 2;

                    neighbour_indexes[index] = (uint64_t *)       realloc(neighbour_indexes[index],  cell_index_array_size[index]);
                    cell_particle_aos[index] = (particle_aos<T> *)realloc(cell_particle_aos[index],  cell_particle_array_size[index]);
                }
            }

            void resize_nodes_arrays (uint64_t elements)
            {
                while ( node_index_array_size < ((size_t) elements * sizeof(uint64_t)) )
                {
                    node_index_array_size *= 2;
                    node_flow_array_size  *= 2;

                    interp_node_indexes     = (uint64_t*)    realloc(interp_node_indexes,     node_index_array_size);
                    interp_node_flow_fields = (flow_aos<T> *)realloc(interp_node_flow_fields, node_flow_array_size);
                }
            }

            void resize_send_buffers_nodes_arrays (uint64_t elements)
            {
                while ( send_buffers_node_index_array_size < ((size_t) elements * sizeof(uint64_t)) )
                {
                    send_buffers_node_index_array_size *= 2;
                    send_buffers_node_flow_array_size  *= 2;

                    send_buffers_interp_node_indexes     = (uint64_t*)    realloc(send_buffers_interp_node_indexes,     send_buffers_node_index_array_size);
                    send_buffers_interp_node_flow_fields = (flow_aos<T> *)realloc(send_buffers_interp_node_flow_fields, send_buffers_node_flow_array_size);
                }
            }

            size_t get_array_memory_usage ()
            {
                uint64_t total_node_index_array_size              = node_index_array_size;
                uint64_t total_node_flow_array_size               = node_flow_array_size;
                uint64_t total_send_buffers_node_index_array_size = send_buffers_node_index_array_size;
                uint64_t total_send_buffers_node_flow_array_size  = send_buffers_node_flow_array_size;


                uint64_t total_cell_index_array_size    = 0;
                uint64_t total_cell_particle_array_size = 0;
                for ( uint64_t i = 0; i < cell_index_array_size.size(); i++ )
                {
                    total_cell_index_array_size    = cell_index_array_size[i];
                    total_cell_particle_array_size = cell_particle_array_size[i];
                }

                return total_cell_index_array_size + total_cell_particle_array_size + total_node_index_array_size + total_node_flow_array_size + total_send_buffers_node_index_array_size + total_send_buffers_node_flow_array_size;
            }

            size_t get_stl_memory_usage ()
            {
                uint64_t total_unordered_neighbours_set_size   = unordered_neighbours_set[0].size()          * sizeof(uint64_t) ;
                uint64_t total_cell_particle_field_map_size    = cell_particle_field_map[0].size()           * sizeof(uint64_t);
                uint64_t total_node_to_position_map_size       = node_to_position_map.size()                 * sizeof(uint64_t);
                uint64_t total_mpi_requests_size               = recv_requests.size() * send_requests.size() * sizeof(MPI_Request);
                uint64_t total_mpi_statuses_size               = statuses.size()                             * sizeof(MPI_Status);
                uint64_t total_new_cells_size                  = new_cells_set.size()                        * sizeof(uint64_t);
                uint64_t total_ranks_size                      = ranks.size()                                * sizeof(uint64_t);
                uint64_t total_local_particle_node_sets_size   = 0;
                for ( uint64_t i = 0; i < local_particle_node_sets.size(); i++ )
                    total_local_particle_node_sets_size += local_particle_node_sets[i].size() * sizeof(uint64_t);


                return total_unordered_neighbours_set_size + total_cell_particle_field_map_size + total_node_to_position_map_size + total_mpi_requests_size + total_mpi_statuses_size + total_new_cells_size + total_local_particle_node_sets_size + total_ranks_size;
            }

            bool is_halo( uint64_t cell );

            void print_logger_stats(uint64_t timesteps, double runtime);
            
            void get_neighbour_cells(const uint64_t recv_id);
            void interpolate_to_nodes();

            void update_flow_field();  // Synchronize point with flow solver
            
            void solve_combustion_equations();
            void update_combustion_fields();

            void solve_turbulence_equations();
            void update_turbulence_fields();

            void solve_flow_equations();

            void timestep();


    }; // class FlowSolver

}   // namespace minicombust::flow 