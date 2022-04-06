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

            uint64_t current_particle;
            Particle<T> *particles;
            ParticleDistribution<T> *particle_dist;

            Mesh<T> *mesh;
            
            particle_logger logger;
            
            T flow_field;

            T dm_d_dt;     // For mass equation
            T domega_Z_dt; // For mixture fraction equation
            T S_i_d;       // For momentum equation
            T Q_d;         // For energy equation


        public:

            template<typename M>
            ParticleSolver(uint64_t ntimesteps, ParticleDistribution<T> *particle_dist, Mesh<M> *mesh) : particle_dist(particle_dist), mesh(mesh)
            {
                // TODO: Take into account decay rate of particles, shrink size of array. Dynamic memory resize?
                printf("Particle storage requirements:\n\tAllocating particles array, %llu particles (%.2f MB)\n", ntimesteps * particle_dist->particles_per_timestep, 
                                                                              (float)(ntimesteps * particle_dist->particles_per_timestep * sizeof(Particle<T>))/1000000.0);
                particles = (Particle<T> *)malloc(ntimesteps * particle_dist->particles_per_timestep * sizeof(Particle<T>));

                memset(&logger, 0, sizeof(particle_logger));
            }

            void output_data(int timestep);

            void print_logger_stats(int timesteps, double runtime);

            void update_flow_field(); // Synchronize point with flow solver
            
            void particle_release();

            void solve_spray_equations();
            
            void update_particle_positions();

            void update_spray_source_terms();

            void map_source_terms_to_grid();

            void interpolate_data();

            void timestep();



    }; // class ParticleSolver

}   // namespace minicombust::particles 
