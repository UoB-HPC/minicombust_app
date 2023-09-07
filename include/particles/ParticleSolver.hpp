#pragma once

#include <map>
#include <memory.h>
#include <vector>

#include "utils/utils.hpp"
#include "particles/Particle.hpp"
#include "particles/ParticleDistribution.hpp"
#include "performance/PerformanceLogger.hpp"

using namespace minicombust::performance; 

namespace minicombust::particles 
{

    using namespace std; 

    template<class T>
    class ParticleSolver 
    {
        private:

            T delta;

            const uint64_t num_timesteps;
            const uint64_t reserve_particles_size;
           
            vector<uint64_t>                             active_blocks;
            vector<Particle<T>>                          particles;
            vector<unordered_map<uint64_t, uint64_t>>    cell_particle_field_map;
            unordered_map<uint64_t, flow_aos<T> *>       node_to_field_address_map;
            vector<unordered_set<uint64_t>>              neighbours_sets;
            ParticleDistribution<T>                     *particle_dist;

            Mesh<T> *mesh;
            
            Particle_Logger logger;
            
            PerformanceLogger<T> performance_logger;

            T flow_field;

            T domega_Z_dt; // For mixture fraction equation
            T S_i_d;       // For momentum equation
            T Q_d;         // For energy equation

            particle_aos<T> **cell_particle_aos;

            int *neighbours_size;
            uint64_t  rank_neighbours_size;
            uint64_t **cell_particle_indexes;

            
            uint64_t    **all_interp_node_indexes;
            flow_aos<T> **all_interp_node_flow_fields;

            size_t  *node_index_array_sizes;
            size_t  *node_flow_array_sizes;

            size_t  *cell_particle_index_array_sizes;
            size_t  *cell_particle_array_sizes;

            int *block_ranks;

            double time_stats[6] = {0.0};

            bool *async_locks;

            MPI_Request bcast_request;
            vector<MPI_Request> send_requests;
            vector<MPI_Request> recv_requests;
            vector<MPI_Status>  statuses;

            uint64_t         *send_counts;         
            uint64_t        **recv_indexes;        
            particle_aos<T> **recv_indexed_fields; 

        public:
            MPI_Config *mpi_config;

            
            template<typename M>
            ParticleSolver(MPI_Config *mpi_config, uint64_t ntimesteps, T delta, ParticleDistribution<T> *particle_dist, Mesh<M> *mesh, uint64_t reserve_particles_size) : 
                           delta(delta), num_timesteps(ntimesteps), reserve_particles_size(reserve_particles_size), particle_dist(particle_dist), mesh(mesh), mpi_config(mpi_config)
            {
                // Allocate space for the size of each block array size
                node_index_array_sizes           = (size_t *)malloc(mesh->num_blocks * sizeof(size_t));
                node_flow_array_sizes            = (size_t *)malloc(mesh->num_blocks * sizeof(size_t));
                
                cell_particle_index_array_sizes  = (size_t *)malloc(mesh->num_blocks * sizeof(size_t));
                cell_particle_array_sizes        = (size_t *)malloc(mesh->num_blocks * sizeof(size_t));

                block_ranks = (int *)malloc(mesh->num_blocks * mpi_config->particle_flow_world_size * sizeof(int)); // TODO: Check how big this gets!

                // Allocate each blocks cell arrays
                neighbours_size             = (int *)          malloc(mesh->num_blocks * sizeof(int));
                all_interp_node_indexes     = (uint64_t **)    malloc(mesh->num_blocks * sizeof(uint64_t *));
                all_interp_node_flow_fields = (flow_aos<T> **) malloc(mesh->num_blocks * sizeof(flow_aos<T> *));

                cell_particle_indexes = (uint64_t **)         malloc(mesh->num_blocks * sizeof(uint64_t *));
                cell_particle_aos     = (particle_aos<T>  **) malloc(mesh->num_blocks * sizeof(particle_aos<T>  *));

                async_locks    = (bool*)malloc(5 * mesh->num_blocks * sizeof(bool));

                send_counts    =              (uint64_t*) malloc(mesh->num_blocks * sizeof(uint64_t));
                recv_indexes   =             (uint64_t**) malloc(mesh->num_blocks * sizeof(uint64_t*));
                recv_indexed_fields = (particle_aos<T>**) malloc(mesh->num_blocks * sizeof(particle_aos<T>*));

                // For each block, allocate a fraction of their local mesh size
                for ( uint64_t b = 0; b < mesh->num_blocks; b++ )
                {
                    // Compute sizes for each block
                    const double fraction = 0.1;
                    const uint64_t block_size  = max(mesh->block_element_disp[b + 1] - mesh->block_element_disp[b], 1UL);

                    const uint64_t storage = min(max(fraction * block_size,  1.), 1. + (double)((100 * particle_dist->even_particles_per_timestep) / mpi_config->particle_flow_world_size)) ;

                    node_index_array_sizes[b]   = storage * sizeof(uint64_t); 
                    node_flow_array_sizes[b]    = storage * sizeof(flow_aos<T>); 

                    cell_particle_index_array_sizes[b] = storage * sizeof(uint64_t); 
                    cell_particle_array_sizes[b]       = storage * sizeof(particle_aos<T>); 

                    // Allocate each block array
                    all_interp_node_indexes[b]      = (uint64_t *)   malloc(node_index_array_sizes[b]);
                    all_interp_node_flow_fields[b]  = (flow_aos<T> *)malloc(node_flow_array_sizes[b]);

                    cell_particle_indexes[b]        =        (uint64_t *)malloc(cell_particle_index_array_sizes[b]);
                    cell_particle_aos[b]            = (particle_aos<T> *)malloc(cell_particle_array_sizes[b]);

                    neighbours_sets.push_back(unordered_set<uint64_t>());
                    cell_particle_field_map.push_back(unordered_map<uint64_t, uint64_t>());
                }

                // TODO: Play with these for performance
                particles.reserve(reserve_particles_size);
                // cell_particle_field_map.reserve(mesh->mesh_size / 10);

                memset(&logger,           0, sizeof(Particle_Logger));


                // Array sizes
                uint64_t total_node_index_array_size           = 0;
                uint64_t total_node_flow_array_size            = 0;
                uint64_t total_cell_particle_index_array_size  = 0;
                uint64_t total_cell_particle_array_size        = 0;

                // STL sizes
                uint64_t total_neighbours_sets_size            = 0;
                uint64_t total_cell_particle_field_map_size    = 0;

                uint64_t total_particles_size                  = particles.size() * sizeof(Particle<T>);
                uint64_t total_node_to_field_address_map_size  = node_to_field_address_map.size() * sizeof(flow_aos<T> *);

                uint64_t total_memory_usage = get_array_memory_usage() + get_stl_memory_usage();

                for (uint64_t b = 0; b < mesh->num_blocks; b++)
                {

                    total_node_index_array_size  += node_index_array_sizes[b];
                    total_node_flow_array_size   += node_flow_array_sizes[b];

                    total_cell_particle_index_array_size += cell_particle_index_array_sizes[b];
                    total_cell_particle_array_size       += cell_particle_array_sizes[b];

                    total_neighbours_sets_size            += neighbours_sets[b].size() * sizeof(uint64_t);
                    total_cell_particle_field_map_size    += cell_particle_field_map[b].size() * sizeof(uint64_t);
                }

                MPI_Barrier(mpi_config->world);

                if (mpi_config->particle_flow_rank == 0)
                {
                    MPI_Reduce(MPI_IN_PLACE, &total_memory_usage,                                 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);

                    MPI_Reduce(MPI_IN_PLACE, &total_node_index_array_size,                        1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_node_flow_array_size,                         1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_particle_index_array_size,               1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_particle_array_size,                     1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_neighbours_sets_size,                         1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_particle_field_map_size,                 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_particles_size,                               1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_node_to_field_address_map_size,               1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);


                    printf("Particle solver storage requirements (%d processes) : \n", mpi_config->particle_flow_world_size);
                    printf("\ttotal_node_index_array_size                           (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_index_array_size           / 1000000.0, (float) total_node_index_array_size          / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_node_flow_array_size                            (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_flow_array_size            / 1000000.0, (float) total_node_flow_array_size           / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_cell_particle_index_array_size                  (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_index_array_size  / 1000000.0, (float) total_cell_particle_index_array_size / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_cell_particle_array_size                        (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_array_size        / 1000000.0, (float) total_cell_particle_array_size       / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_neighbours_sets_size            (STL set)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_neighbours_sets_size            / 1000000.0, (float) total_neighbours_sets_size           / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_cell_particle_field_map_size    (STL map)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_field_map_size    / 1000000.0, (float) total_cell_particle_field_map_size   / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_particles_size                  (STL vector)    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_particles_size                  / 1000000.0, (float) total_particles_size                 / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_node_to_field_address_map_size  (STL map)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n\n"  , (float) total_node_to_field_address_map_size  / 1000000.0, (float) total_node_to_field_address_map_size / (1000000.0 * mpi_config->particle_flow_world_size));

                    printf("\tParticle solver size                                  (TOTAL %12.2f MB) (AVG %.2f MB) \n\n"  , (float)total_memory_usage                      /1000000.0,  (float)total_memory_usage / (1000000.0 * mpi_config->particle_flow_world_size));
                }
                else
                {
                    MPI_Reduce(&total_memory_usage,                   nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);

                    MPI_Reduce(&total_node_index_array_size ,         nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_node_flow_array_size ,          nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_cell_particle_index_array_size, nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_cell_particle_array_size,       nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_neighbours_sets_size,           nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_cell_particle_field_map_size,   nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_particles_size,                 nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_node_to_field_address_map_size, nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                }

                // for (uint64_t b = 0; b < mesh->num_blocks; b++)
                // {
                //     MPI_Comm_split(mpi_config->world, 1, mpi_config->rank, &mpi_config->every_one_flow_world[b]); 
                //     MPI_Comm_rank(mpi_config->every_one_flow_world[b], &mpi_config->every_one_flow_rank[b]);
                //     MPI_Comm_size(mpi_config->every_one_flow_world[b], &mpi_config->every_one_flow_world_size[b]);
                // }

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);

            }

            // void resize_cell_indexes (uint64_t *elements, uint64_t ***new_cell_indexes)
            // {
            //     for ( uint64_t b = 0; b < mesh->num_blocks; b++)
            //     {
            //         while ( cell_index_array_sizes[b] < ((size_t) elements[b] * sizeof(uint64_t)) )
            //         {
            //             cell_index_array_sizes[b] *= 2;

            //             cell_indexes[b] = (uint64_t*)realloc(cell_indexes[b], cell_index_array_sizes[b]);
            //             if (cell_indexes[b] == NULL)
            //             {
            //                 printf("REALLOC FAILED\n");
            //             }
            //         }
                    
            //         if (new_cell_indexes != NULL)  (*new_cell_indexes)[b] = cell_indexes[b];
            //     }

            // }


            void resize_cell_particle_indexes (uint64_t *elements, uint64_t ***new_cell_indexes)
            {
                for ( uint64_t b = 0; b < mesh->num_blocks; b++)
                {
                    while ( cell_particle_index_array_sizes[b] < ((size_t) elements[b] * sizeof(uint64_t)) )
                    {
                        // printf("Rank %d Resizing index block %lu: size %lu to %lu\n", mpi_config->rank, b, cell_particle_index_array_sizes[b] / sizeof(uint64_t), 2* cell_particle_index_array_sizes[b] / sizeof(uint64_t) );
                        cell_particle_index_array_sizes[b] *= 2;

                        cell_particle_indexes[b] = (uint64_t*)realloc(cell_particle_indexes[b], cell_particle_index_array_sizes[b]);
                    }
                    
                    if (new_cell_indexes != NULL)  (*new_cell_indexes)[b] = cell_particle_indexes[b];
                }
            }

            void resize_cell_particle (uint64_t *elements, uint64_t ***new_cell_indexes, particle_aos<T> ***new_cell_particle)
            {
                resize_cell_particle_indexes(elements, new_cell_indexes);

                for ( uint64_t b = 0; b < mesh->num_blocks; b++)
                {

                    while ( cell_particle_array_sizes[b] < ((size_t) elements[b] * sizeof(particle_aos<T>)) )
                    {
                        // printf("Rank %d Resizing field block %lu: size %lu to %lu\n", mpi_config->rank, b, cell_particle_array_sizes[b] / sizeof(particle_aos<T>), 2 * cell_particle_array_sizes[b] / sizeof(particle_aos<T>) );

                        cell_particle_array_sizes[b] *= 2;

                        cell_particle_aos[b] = (particle_aos<T> *)realloc(cell_particle_aos[b], cell_particle_array_sizes[b]);
                    }

                    if (new_cell_particle != NULL)  (*new_cell_particle)[b] = cell_particle_aos[b];
                }
            }

            void resize_nodes_arrays (int elements, uint64_t block_id)
            {
                while ( node_index_array_sizes[block_id] < ((size_t) elements * sizeof(uint64_t)) )
                {
                    node_index_array_sizes[block_id] *= 2;
                    node_flow_array_sizes[block_id]  *= 2;

                    all_interp_node_indexes[block_id]     = (uint64_t*)    realloc(all_interp_node_indexes[block_id],     node_index_array_sizes[block_id]);
                    all_interp_node_flow_fields[block_id] = (flow_aos<T> *)realloc(all_interp_node_flow_fields[block_id], node_flow_array_sizes[block_id]);
                }
            }

            size_t get_array_memory_usage ()
            {
                uint64_t total_node_index_array_size          = 0;
                uint64_t total_node_flow_array_size           = 0;
                uint64_t total_cell_particle_index_array_size = 0;
                uint64_t total_cell_particle_array_size       = 0;

                for (uint64_t b = 0; b < mesh->num_blocks; b++)  
                {
                    total_node_index_array_size  += node_index_array_sizes[b];
                    total_node_flow_array_size   += node_flow_array_sizes[b];

                    total_cell_particle_index_array_size += cell_particle_index_array_sizes[b];
                    total_cell_particle_array_size       += cell_particle_array_sizes[b];
                }

                // if (mpi_config->particle_flow_rank == 0)
                // {
                //     printf("total_node_index_array_size  %.2f\n",         total_node_index_array_size           / 1.e9);
                //     printf("total_node_flow_array_size  %.2f\n",          total_node_flow_array_size            / 1.e9);
                //     printf("total_cell_particle_index_array_size %.2f\n", total_cell_particle_index_array_size  / 1.e9);
                //     printf("total_cell_particle_array_size %.2f\n",       total_cell_particle_array_size        / 1.e9);

                // }
                return  total_node_index_array_size  + total_node_flow_array_size  + total_cell_particle_index_array_size + total_cell_particle_array_size ;

            }

            size_t get_stl_memory_usage ()
            {
                uint64_t total_neighbours_sets_size            = 0;
                uint64_t total_cell_particle_field_map_size    = 0;

                uint64_t total_particles_size                  = particles.size() * sizeof(Particle<T>);
                uint64_t total_node_to_field_address_map_size  = node_to_field_address_map.size() * sizeof(flow_aos<T> *);

                for (uint64_t b = 0; b < mesh->num_blocks; b++)  
                {
                    total_neighbours_sets_size            += neighbours_sets[b].size() * sizeof(uint64_t);
                    total_cell_particle_field_map_size    += cell_particle_field_map[b].size() * sizeof(uint64_t);
                }

                // if (mpi_config->particle_flow_rank == 0)
                // {
                //     printf("total_particles_size %.2f\n",                 total_particles_size           / 1.e9);
                //     printf("total_node_to_field_address_map_size %.2f\n", total_node_to_field_address_map_size          / 1.e9);
                //     printf("total_neighbours_sets_size %.2f\n",           total_neighbours_sets_size           / 1.e9);
                //     printf("total_cell_particle_field_map_size %.2f\n",   total_cell_particle_field_map_size  / 1.e9);

                // }

                return total_neighbours_sets_size + total_cell_particle_field_map_size + total_particles_size + total_node_to_field_address_map_size;
            }

            void output_data(uint64_t timestep);

            void print_logger_stats(uint64_t timesteps, double runtime);

            void update_flow_field(); // Synchronize point with flow solver
            
            void particle_release();

            void solve_spray_equations();
            
            void update_particle_positions();

            void update_spray_source_terms();

            void map_source_terms_to_grid();

            void interpolate_nodal_data();

            void timestep();



    }; // class ParticleSolver

}   // namespace minicombust::particles 
