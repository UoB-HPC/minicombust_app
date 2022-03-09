#pragma once

// #include <random>

#include "geometry/Mesh.hpp"
#include "particles/Particle.hpp"
#include "utils/utils.hpp"

using namespace minicombust::utils;

namespace minicombust::particles 
{
    template<class T>
    class Distribution 
    {
        // private:

        public:
            virtual T get_value() = 0;

        protected:
            Distribution() { }


            
            
    }; // class Distribution

    template<class T>
    class NormalDistribution : public Distribution<T>
    {
        private:
            T mean;
            T std_dev;
            T lower;
            T upper;

        public:
             NormalDistribution(T mean, T std_dev, T lower, T upper) : mean(mean), std_dev(std_dev), lower(lower), upper(upper)
             { }

            // TODO: Add normal distribution random functionality
            T get_value() override {
                return mean;
            }

    }; // class NormalDistribution

    template<class T>
    class UniformDistribution : public Distribution<T>
    {
        private:
            T lower;
            T upper;

        public:
             UniformDistribution(T lower, T upper) : lower(lower), upper(upper)
             { }

            T get_value() override {
                T r = (T)rand() / RAND_MAX;
                return lower + (r * (upper - lower));
            }

    }; // class UniformDistribution

    template<class T>
    class FixedDistribution : public Distribution<T>
    {
        private:
            T fixed_val;

        public:
             FixedDistribution(T fixed_val) : fixed_val(fixed_val)
             { }

            T get_value() override {
                return fixed_val;
            }

    }; // class FixedDistribution
 

    template<class T>
    class ParticleDistribution 
    {
        // private:

        public:
            // TODO: Change this to particles per unit of time?
            uint64_t particles_per_timestep;
            Mesh<T> *mesh;

            Distribution<vec<T>> *start_pos;
            Distribution<T> *rate;            
            Distribution<T> *angle_xy;        
            Distribution<T> *angle_rot;  
            Distribution<vec<T>> *velocity;
            Distribution<vec<T>> *acceleration;
            Distribution<vec<T>> *jerk;
            Distribution<T> *decay_rate;      
            Distribution<T> *decay_threshold; 

            // TODO: Read particle distribution from file

            // Generate fixed distribution
            ParticleDistribution (uint64_t particles_per_timestep, Mesh<T> *mesh, vec<T> start, T rate_mean, T angle_xy_mean, T angle_rot_mean, vec<T> vel_mean,
                                  vec<T> acc_mean, vec<T> jerk_mean, T decay_rate_mean, T decay_threshold_mean) :
                                  particles_per_timestep(particles_per_timestep), mesh(mesh)
            {   
                start_pos        = new FixedDistribution<vec<T>>(start);
                rate             = new FixedDistribution<T>(rate_mean);           
                angle_xy         = new FixedDistribution<T>(angle_xy_mean);   
                angle_rot        = new FixedDistribution<T>(angle_rot_mean);  
                velocity         = new FixedDistribution<vec<T>>(vel_mean);
                acceleration     = new FixedDistribution<vec<T>>(acc_mean);
                jerk             = new FixedDistribution<vec<T>>(jerk_mean);
                decay_rate       = new FixedDistribution<T>(decay_rate_mean);  
                decay_threshold  = new FixedDistribution<T>(decay_threshold_mean);  
            }

            void emit_particles(Particle<T> *particles)
            {
                for (int p = 0; p < particles_per_timestep; p++)
                {
                    // TODO: Calc starting cell positions here.
                    particles[p] = Particle<T>(start_pos->get_value(), velocity->get_value(), acceleration->get_value(), 0);
                }
            }
                                  

    }; // class ParticleDistribution
 
}   // namespace minicombust::particles 