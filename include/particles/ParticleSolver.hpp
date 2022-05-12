#pragma once

#include <map>
#include <memory.h>
#include <vector>

#include "utils/utils.hpp"
#include "particles/Particle.hpp"
#include "particles/ParticleDistribution.hpp"
#include "performance/PerformanceLogger.hpp"

#ifdef PAPI
using namespace minicombust::performance; 
#endif

namespace minicombust::particles 
{

    using namespace std; 

    template<class T>
    class ParticleSolver 
    {
        private:

            T delta;

            const uint64_t num_timesteps;

           
            vector<Particle<T>>      particles;
            unordered_set<uint64_t>  cell_set;
            ParticleDistribution<T> *particle_dist;

            Mesh<T> *mesh;
            
            particle_logger logger;
            
            #ifdef PAPI
            PerformanceLogger<T> performance_logger;
            #endif

            T flow_field;

            T domega_Z_dt; // For mixture fraction equation
            T S_i_d;       // For momentum equation
            T Q_d;         // For energy equation

            vec_soa<T>   nodal_gas_velocity_soa;
            flow_aos<T> *nodal_flow_aos;


        public:

            template<typename M>
            ParticleSolver(uint64_t ntimesteps, T delta, ParticleDistribution<T> *particle_dist, Mesh<M> *mesh, uint64_t n) : delta(delta), particle_dist(particle_dist), mesh(mesh), num_timesteps(n)
            {
                const size_t source_vector_array_size = mesh->points_size * sizeof(vec<T>);
                const size_t source_scalar_array_size = mesh->points_size * sizeof(T);
                const size_t particles_array_size    = mesh->max_cell_particles * sizeof(Particle<T>);

                nodal_gas_velocity_soa            = allocate_vec_soa<T>(mesh->points_size);
                nodal_flow_aos                    = (flow_aos<T> *)malloc(mesh->points_size * sizeof(flow_aos<T>));

                particles.reserve(mesh->max_cell_particles);


                // TODO: Take into account decay rate of particles, shrink size of array. Dynamic memory resize?
                printf("Particle solver storage requirements:\n");
                printf("\tReserving particles array, particles                        (%.2f MB)  particles_array_size %" PRIu64 "\n",   (float)(particles_array_size)/1000000.0, mesh->max_cell_particles);
                printf("\tAllocating nodal_gas_velocity array                         (%.2f MB)\n",  ((float)source_vector_array_size)/1000000.);
                printf("\tAllocating nodal_gas_pressure array                         (%.2f MB)\n",  ((float)source_scalar_array_size)/1000000.);
                printf("\tAllocating nodal_gas_temperature array                      (%.2f MB)\n",  ((float)source_scalar_array_size)/1000000.);

                const size_t total_size = particles_array_size + source_vector_array_size + source_scalar_array_size + source_scalar_array_size;
                printf("\tAllocated particle solver. Total size                       (%.2f MB)\n\n", ((float)total_size)/1000000.0);

                memset(&logger,           0, sizeof(particle_logger));

                

                #ifdef PAPI
                performance_logger.init_papi();
                performance_logger.load_papi_events();
                #endif

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
