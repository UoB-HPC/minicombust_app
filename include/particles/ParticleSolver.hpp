#pragma once

#include <map>
#include <memory.h>

#include "utils/utils.hpp"
#include "particles/Particle.hpp"
#include "particles/ParticleDistribution.hpp"

namespace minicombust::particles 
{

    using namespace std; 

    template<class T>
    class ParticleSolver 
    {
        private:

            T delta;

           
            Particle<T> *particles;
            ParticleDistribution<T> *particle_dist;

            Mesh<T> *mesh;
            
            particle_logger logger;
            
            T flow_field;

            T domega_Z_dt; // For mixture fraction equation
            T S_i_d;       // For momentum equation
            T Q_d;         // For energy equation

            uint8_t *nodal_counter;

            vec<T> *nodal_gas_velocity;
            T      *nodal_gas_pressure;
            T      *nodal_gas_temperature;

        public:

            uint64_t current_particle0;
            uint64_t current_particle1;

            template<typename M>
            ParticleSolver(uint64_t ntimesteps, T delta, ParticleDistribution<T> *particle_dist, Mesh<M> *mesh) : delta(delta), particle_dist(particle_dist), mesh(mesh)
            {
                const size_t particles_array_size     = mesh->max_cell_particles * mesh->mesh_size * sizeof(Particle<T>);
                const size_t source_vector_array_size = mesh->points_size * sizeof(vec<T>);
                const size_t source_scalar_array_size = mesh->points_size * sizeof(T);
                const size_t source_uint8t_array_size = mesh->points_size * sizeof(uint8_t);


                particles                         = (Particle<T> *)malloc(particles_array_size);
                nodal_gas_velocity                = (vec<T> *)     malloc(source_vector_array_size);
                nodal_gas_pressure                = (T *)          malloc(source_scalar_array_size);
                nodal_gas_temperature             = (T *)          malloc(source_scalar_array_size);

                nodal_counter                     = (uint8_t *)    malloc(source_uint8t_array_size);


                // TODO: Take into account decay rate of particles, shrink size of array. Dynamic memory resize?
                printf("Particle solver storage requirements:\n");
                printf("\tAllocating particles array, particles                       (%.2f MB)  particles_array_size %llu\n",   (float)(particles_array_size)/1000000.0, mesh->max_cell_particles * mesh->mesh_size);
                printf("\tAllocating nodal_gas_velocity array                         (%.2f MB)\n",  ((float)source_vector_array_size)/1000000.);
                printf("\tAllocating nodal_gas_pressure array                         (%.2f MB)\n",  ((float)source_scalar_array_size)/1000000.);
                printf("\tAllocating nodal_gas_temperature array                      (%.2f MB)\n",  ((float)source_scalar_array_size)/1000000.);
                printf("\tAllocating nodal_counter array,                             (%.2f MB)\n",  ((float)source_uint8t_array_size)/1000000.);
                const size_t total_size = particles_array_size + source_vector_array_size + source_scalar_array_size + source_scalar_array_size + source_uint8t_array_size;
                printf("\tAllocated particle solver. Total size                       (%.2f MB)\n\n", ((float)total_size)/1000000.0);

                memset(&logger,           0, sizeof(particle_logger));

                memset(nodal_counter, 0, mesh->points_size*sizeof(uint8_t));
                for (int n = 0; n < mesh->points_size; n++)
                {
                    nodal_gas_velocity[n]     = {0.0, 0.0, 0.0};
                    nodal_gas_pressure[n]     = 0.0;
                    nodal_gas_temperature[n]  = 0.0;
                }
            }

            void output_data(int timestep);

            void print_logger_stats(int timesteps, double runtime);

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
