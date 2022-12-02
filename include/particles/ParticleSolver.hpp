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
           
            vector<Particle<T>>                       particles;
            unordered_map<uint64_t, particle_aos<T>>  cell_particle_field_map;
            unordered_map<uint64_t, uint64_t>         node_to_position_map;
            unordered_set<uint64_t>                   neighbours_set;
            ParticleDistribution<T>                  *particle_dist;

            Mesh<T> *mesh;
            
            Particle_Logger logger;
            
            PerformanceLogger<T> performance_logger;

            T flow_field;

            T domega_Z_dt; // For mixture fraction equation
            T S_i_d;       // For momentum equation
            T Q_d;         // For energy equation

            particle_aos<T> *cell_particle_aos;
            flow_aos<T>     *cell_flow_aos;
            flow_aos<T>     *cell_flow_grad_aos;

            uint64_t  neighbours_size;
            uint64_t  rank_neighbours_size;
            uint64_t *cell_indexes;

            
            uint64_t max_cell_storage;
            uint64_t max_point_storage;

            uint64_t    *all_interp_node_indexes;
            flow_aos<T> *all_interp_node_flow_fields;

            int *rank_nodal_sizes;
            int *rank_nodal_disps;

            size_t   cell_index_array_size;
            size_t   point_index_array_size;
            size_t   cell_flow_array_size;
            size_t   cell_particle_array_size;
            size_t   point_flow_array_size;

        public:
            MPI_Config *mpi_config;


            
            template<typename M>
            ParticleSolver(MPI_Config *mpi_config, uint64_t ntimesteps, T delta, ParticleDistribution<T> *particle_dist, Mesh<M> *mesh, uint64_t reserve_particles_size) : 
                           delta(delta), num_timesteps(ntimesteps), reserve_particles_size(reserve_particles_size), particle_dist(particle_dist), mesh(mesh), mpi_config(mpi_config)
            {
                const size_t particles_array_size = reserve_particles_size * sizeof(Particle<T>);
                
                float fraction    = 0.125;
                max_cell_storage  = fraction * mesh->mesh_size;
                max_point_storage = 2 * fraction * mesh->points_size;

                cell_index_array_size       = max_cell_storage * sizeof(uint64_t);
                cell_flow_array_size        = max_cell_storage * sizeof(flow_aos<T>);
                cell_particle_array_size    = max_cell_storage * sizeof(particle_aos<T>);

                point_index_array_size     = max_point_storage * sizeof(uint64_t); 
                point_flow_array_size      = max_point_storage * sizeof(flow_aos<T>); 


                cell_indexes                      =        (uint64_t *)malloc(cell_index_array_size);
                cell_particle_aos                 = (particle_aos<T> *)malloc(cell_particle_array_size);
                cell_flow_aos                     =     (flow_aos<T> *)malloc(cell_flow_array_size);
                cell_flow_grad_aos                =     (flow_aos<T> *)malloc(cell_flow_array_size);

                all_interp_node_indexes           = (uint64_t *)   malloc(point_index_array_size);
                all_interp_node_flow_fields       = (flow_aos<T> *)malloc(point_flow_array_size);

                rank_nodal_sizes = (int *)malloc(mpi_config->particle_flow_world_size * sizeof(int));
                rank_nodal_disps = (int *)malloc(mpi_config->particle_flow_world_size * sizeof(int));

                particles.reserve(reserve_particles_size);
                
                // TODO: Play with these for performance
                // cell_particle_field_map.reserve(mesh->mesh_size / 10);

                if (mpi_config->particle_flow_rank == 0)
                {
                    // TODO: Take into account decay rate of particles, shrink size of array. Dynamic memory resize?
                    printf("Particle solver storage requirements (per process):\n");
                    printf("\tReserving particles array, particles                         (%.2f MB)  particles_array_size %" PRIu64 "\n",   (float)(particles_array_size)/1000000.0, reserve_particles_size);
                    // printf("\tAllocating nodal flow source terms array                     (%.2f MB)\n",   ((float)nodal_array_size)/1000000.);
                    printf("\tAllocating cell flow source terms array                      (%.2f MB)\n",   ((float)cell_flow_array_size)/1000000.);
                    printf("\tAllocating cell flow source terms array                      (%.2f MB)\n",   ((float)cell_particle_array_size)/1000000.);
                    printf("\tAllocating cell grad flow source terms array                 (%.2f MB)\n",   ((float)cell_flow_array_size)/1000000.);
                    printf("\tAllocating neighbour cell array                              (%.2f MB)\n",   ((float)cell_index_array_size)/1000000.);
                    printf("\tAllocating interpolation node indexes                        (%.2f MB)\n",   ((float)point_index_array_size)/1000000.);
                    printf("\tAllocating interpolation node flow terms                     (%.2f MB)\n",   ((float)point_flow_array_size)/1000000.);

                    // const size_t total_size = particles_array_size + nodal_array_size + 2*cell_flow_array_size + cell_index_array_size;
                    const size_t total_size = particles_array_size + 2*cell_flow_array_size + cell_index_array_size + point_index_array_size + point_flow_array_size + cell_particle_array_size;
                    printf("\tAllocated particle solver. Total size                        (%.2f MB)\n",   ((float)total_size)/1000000.0);
                    printf("\tAllocated particle solver. Total size (per world)            (%.2f MB)\n\n", ((float)total_size * mpi_config->particle_flow_world_size)/1000000.0);
                }


                memset(&logger,           0, sizeof(Particle_Logger));

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);

            }

            void resize_cell_indexes (uint64_t elements, uint64_t **new_cell_indexes)
            {
                while ( cell_index_array_size < ((size_t) elements * sizeof(uint64_t)) )
                {
                    printf("Rank %d resizing particle %ld to  %ld\n", mpi_config->rank, cell_index_array_size / sizeof(uint64_t), cell_index_array_size*2 / sizeof(uint64_t));
                    cell_index_array_size *= 2;

                    cell_indexes = (uint64_t*)realloc(cell_indexes,       cell_index_array_size);
                }
                
                if (new_cell_indexes != NULL)  *new_cell_indexes = cell_indexes;
            }

            void resize_cell_flow (uint64_t elements)
            {
                resize_cell_indexes(elements, NULL);
                while ( cell_flow_array_size < ((size_t) elements * sizeof(flow_aos<T>)) )
                {
                    cell_flow_array_size *= 2;

                    cell_flow_aos      = (flow_aos<T> *)realloc(cell_flow_aos,      cell_flow_array_size);
                    cell_flow_grad_aos = (flow_aos<T> *)realloc(cell_flow_grad_aos, cell_flow_array_size);
                }
            }

            void resize_cell_particle (uint64_t elements, uint64_t **new_cell_indexes, particle_aos<T> **new_cell_particle)
            {
                resize_cell_indexes(elements, new_cell_indexes);
                while ( cell_particle_array_size < ((size_t) elements * sizeof(particle_aos<T>)) )
                {
                    cell_particle_array_size *= 2;

                    cell_particle_aos = (particle_aos<T> *)realloc(cell_particle_aos,  cell_particle_array_size);
                }

                if (new_cell_particle != NULL)  *new_cell_particle = cell_particle_aos;
            }

            void resize_nodes_arrays (uint64_t elements)
            {
                while ( max_point_storage < (uint64_t) elements )
                {
                    printf("Resizing points!!\n");
                    max_point_storage      *= 2;
                    point_index_array_size *= 2;
                    point_flow_array_size  *= 2;

                    all_interp_node_indexes     = (uint64_t*)    realloc(all_interp_node_indexes,     point_index_array_size);
                    all_interp_node_flow_fields = (flow_aos<T> *)realloc(all_interp_node_flow_fields, point_flow_array_size);
                }
            }

            size_t get_array_memory_usage ()
            {
                // if (mpi_config->particle_flow_rank == 0)
                // {
                //     printf("cell_index_array_size %.2f\n",    cell_index_array_size    / 1.e9);
                //     printf("cell_particle_array_size %.2f\n", cell_particle_array_size / 1.e9);
                //     printf("cell_flow_array_size %.2f\n",     cell_flow_array_size     / 1.e9);
                //     printf("point_index_array_size %.2f\n",   point_index_array_size   / 1.e9);
                //     printf("point_flow_array_size %.2f\n",    point_flow_array_size    / 1.e9);

                // }

                return cell_index_array_size + cell_particle_array_size + 2 * cell_flow_array_size + point_index_array_size + point_flow_array_size;
            }

            size_t get_stl_memory_usage ()
            {
                return particles.size()*sizeof(Particle<T>) + cell_particle_field_map.size()*sizeof(particle_aos<T>) + node_to_position_map.size()*sizeof(uint64_t) + neighbours_set.size()*sizeof(neighbours_set);
            }

            void output_data(uint64_t timestep);

            void print_logger_stats(uint64_t timesteps, double runtime);

            void update_flow_field(bool send_particle); // Synchronize point with flow solver
            
            void particle_release();

            void solve_spray_equations();
            
            void update_particle_positions();

            void update_spray_source_terms();

            void map_source_terms_to_grid();

            void interpolate_nodal_data();

            void timestep();



    }; // class ParticleSolver

}   // namespace minicombust::particles 
