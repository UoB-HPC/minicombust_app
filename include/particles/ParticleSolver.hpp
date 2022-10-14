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
            unordered_set<uint64_t>                   neighbours_set;
            ParticleDistribution<T>                  *particle_dist;

            Mesh<T> *mesh;
            
            Particle_Logger logger;
            
            PerformanceLogger<T> performance_logger;

            T flow_field;

            T domega_Z_dt; // For mixture fraction equation
            T S_i_d;       // For momentum equation
            T Q_d;         // For energy equation

            // vec_soa<T>   nodal_gas_velocity_soa;
            flow_aos<T>     *nodal_flow_aos;
            flow_aos<T>     *cell_flow_aos;
            flow_aos<T>     *cell_flow_grad_aos;

            uint64_t  neighbours_size;
            uint64_t *neighbour_indexes;


        public:
            MPI_Config *mpi_config;

            template<typename M>
            ParticleSolver(MPI_Config *mpi_config, uint64_t ntimesteps, T delta, ParticleDistribution<T> *particle_dist, Mesh<M> *mesh, uint64_t n, uint64_t reserve_particles_size) : mpi_config(mpi_config), delta(delta), num_timesteps(n), particle_dist(particle_dist), mesh(mesh), reserve_particles_size(reserve_particles_size)
            {
                neighbours_size = mesh->mesh_size;
                
                const size_t cell_index_array_size   = mesh->mesh_size * sizeof(uint64_t);
                const size_t cell_array_size         = mesh->mesh_size * sizeof(flow_aos<T>);
                const size_t nodal_array_size        = mesh->points_size * sizeof(flow_aos<T>);
                const size_t particles_array_size    = reserve_particles_size * sizeof(Particle<T>);

                if (mpi_config->solver_type == PARTICLE)
                {
                    // nodal_gas_velocity_soa            = allocate_vec_soa<T>(mesh->points_size);
                    nodal_flow_aos                    = (flow_aos<T> *)malloc(nodal_array_size);
                    cell_flow_aos                     = (flow_aos<T> *)malloc(cell_array_size);
                    cell_flow_grad_aos                = (flow_aos<T> *)malloc(cell_array_size);

                    neighbour_indexes                 = (uint64_t*)malloc(cell_index_array_size);
                    particles.reserve(reserve_particles_size);

                }
                

                
                // TODO: Play with these for performance
                // neighbours_set.reserve(mesh->mesh_size / x);
                // cell_particle_field_map.reserve(mesh->mesh_size / 10);

                if (mpi_config->rank == 0)
                {
                    // TODO: Take into account decay rate of particles, shrink size of array. Dynamic memory resize?
                    printf("Particle solver storage requirements (per process):\n");
                    printf("\tReserving particles array, particles                         (%.2f MB)  particles_array_size %" PRIu64 "\n",   (float)(particles_array_size)/1000000.0, reserve_particles_size);
                    printf("\tAllocating nodal flow source terms array                     (%.2f MB)\n",   ((float)nodal_array_size)/1000000.);
                    printf("\tAllocating cell flow source terms array                      (%.2f MB)\n",   ((float)cell_array_size)/1000000.);
                    printf("\tAllocating cell grad flow source terms array                 (%.2f MB)\n",   ((float)cell_array_size)/1000000.);
                    printf("\tAllocating neighbour cell array                              (%.2f MB)\n",   ((float)cell_index_array_size)/1000000.);

                    const size_t total_size = particles_array_size + nodal_array_size + 2*cell_array_size + cell_index_array_size;
                    printf("\tAllocated particle solver. Total size                        (%.2f MB)\n",   ((float)total_size)/1000000.0);
                    printf("\tAllocated particle solver. Total size (per world)            (%.2f MB)\n\n", ((float)total_size * mpi_config->particle_flow_world_size)/1000000.0);
                }


                memset(&logger,           0, sizeof(Particle_Logger));

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);

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
