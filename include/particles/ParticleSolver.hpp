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
            MPI_Config *mpi_config;

            T delta;

            const uint64_t num_timesteps;
            const uint64_t reserve_particles_size;
           
            vector<Particle<T>>      particles;
            unordered_set<uint64_t>  cell_set;
            ParticleDistribution<T> *particle_dist;

            Mesh<T> *mesh;
            
            particle_logger logger;
            
            PerformanceLogger<T> performance_logger;

            T flow_field;

            T domega_Z_dt; // For mixture fraction equation
            T S_i_d;       // For momentum equation
            T Q_d;         // For energy equation

            vec_soa<T>   nodal_gas_velocity_soa;
            flow_aos<T> *nodal_flow_aos;


        public:

            template<typename M>
            ParticleSolver(MPI_Config *mpi_config, uint64_t ntimesteps, T delta, ParticleDistribution<T> *particle_dist, Mesh<M> *mesh, uint64_t n, uint64_t reserve_particles_size) : mpi_config(mpi_config), delta(delta), num_timesteps(n), particle_dist(particle_dist), mesh(mesh), reserve_particles_size(reserve_particles_size)
            {
                const size_t source_vector_array_size = mesh->points_size * sizeof(vec<T>);
                const size_t source_scalar_array_size = mesh->points_size * sizeof(T);
                const size_t particles_array_size    = reserve_particles_size * sizeof(Particle<T>);

                nodal_gas_velocity_soa            = allocate_vec_soa<T>(mesh->points_size);
                nodal_flow_aos                    = (flow_aos<T> *)malloc(mesh->points_size * sizeof(flow_aos<T>));

                particles.reserve(reserve_particles_size);

                if (mpi_config->rank == 0)
                {
                    // TODO: Take into account decay rate of particles, shrink size of array. Dynamic memory resize?
                    printf("Particle solver storage requirements (per process):\n");
                    printf("\tReserving particles array, particles                        (%.2f MB)  particles_array_size %" PRIu64 "\n",   (float)(particles_array_size)/1000000.0, reserve_particles_size);
                    printf("\tAllocating nodal_gas_velocity array                         (%.2f MB)\n",  ((float)source_vector_array_size)/1000000.);
                    printf("\tAllocating nodal_gas_pressure array                         (%.2f MB)\n",  ((float)source_scalar_array_size)/1000000.);
                    printf("\tAllocating nodal_gas_temperature array                      (%.2f MB)\n",  ((float)source_scalar_array_size)/1000000.);

                    const size_t total_size = particles_array_size + source_vector_array_size + source_scalar_array_size + source_scalar_array_size;
                    printf("\tAllocated particle solver. Total size                       (%.2f MB)\n",   ((float)total_size)/1000000.0);
                    printf("\tAllocated particle solver. Total size (per world)           (%.2f MB)\n\n", ((float)total_size * mpi_config->world_size)/1000000.0);
                }


                memset(&logger,           0, sizeof(particle_logger));

                

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);

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
