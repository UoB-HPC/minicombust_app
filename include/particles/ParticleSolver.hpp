#pragma once

#include <map>

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

            Mesh<T> *global_mesh;

            

            
            T flow_field;

            T dm_d_dt;     // For mass equation
            T domega_Z_dt; // For mixture fraction equation
            T S_i_d;       // For momentum equation
            T Q_d;         // For energy equation

        public:

            template<typename M>
            ParticleSolver(uint64_t ntimesteps, ParticleDistribution<T> *particle_dist, Mesh<M> *boundary_mesh, Mesh<M> *global_mesh) : particle_dist(particle_dist), global_mesh(global_mesh)
            {
                // TODO: Take into account decay rate of particles, shrink size of array. Dynamic memory resize?
                printf("Allocating particles array, %llu particles (%.2f MB)\n", ntimesteps * particle_dist->particles_per_timestep, 
                                                                              (float)(ntimesteps * particle_dist->particles_per_timestep * sizeof(Particle<T>))/1000000.0);
                particles = (Particle<T> *)malloc(ntimesteps * particle_dist->particles_per_timestep * sizeof(Particle<T>));
            }

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