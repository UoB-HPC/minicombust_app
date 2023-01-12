#pragma once

// #include <random>

#include "geometry/Mesh.hpp"
#include "particles/Particle.hpp"
#include "utils/utils.hpp"

using namespace std;
using namespace minicombust::utils;

namespace minicombust::particles 
{

    enum ProbabilityDistribution { FIXED = 0, NORMAL = 1, UNIFORM = 2};


    template<class T>
    class Distribution 
    {
        // private:

        public:
            virtual T get_value() = 0;
            virtual T get_scaled_value() = 0;

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

            T get_scaled_value() override {
                return mean;
            }

    }; // class NormalDistribution

    template<class T>
    class UniformDistribution : public Distribution<T>
    {
        private:
            T lower;
            T upper;
            T mean;
            T unit_mean;
            
            double mag;



        public:
             UniformDistribution(T lower, T upper) : lower(lower), upper(upper)
             { 
                mean      = (upper + lower) / 2.;
                mag       = magnitude(mean);
                unit_mean = mean / mag;
             }

            inline T get_value() override {
                T r;
                if constexpr(std::is_same_v<T, double>)
                {
                    r = static_cast<double>(rand())/RAND_MAX;
                }
                else
                {
                    r = T { static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX } ;
                }

                return lower + (r * (upper - lower));
            }

            inline T get_scaled_value() override {
                T r;
                if constexpr(std::is_same_v<T, double>)
                {
                    r = static_cast<double>(rand())/RAND_MAX;
                    r = lower + (r * (upper - lower));
                }
                else
                {
                    // Create random unit vector on sphere surface
                    T rnd_unit = T { static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX } ;
                    rnd_unit = - 1. + (rnd_unit * 2.);

                    T rnd_perpendicular = rnd_unit - dot_product(rnd_unit, unit_mean) * unit_mean; 
                    r = mean + rnd_perpendicular * 150.;

                    r = mag * r / magnitude(r);

                    // cout << print_vec(mean) << "      " << print_vec(rnd_perpendicular) << "     ";
                }



                return r;
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

            T get_scaled_value() override {
                return fixed_val;
            }


    }; // class FixedDistribution
 

    template<class T>
    class ParticleDistribution 
    {
        // private:

        public:
            // TODO: Change this to particles per unit of time?
            uint64_t wave_particles_per_timestep;
            uint64_t even_particles_per_timestep;
            uint64_t remainder_particles;

            MPI_Config *mpi_config;

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
            Distribution<T> *temperature; 



            // TODO: Read particle distribution from file

            // Generate fixed distribution
            ParticleDistribution (uint64_t wave_particles_per_timestep, uint64_t even_particles_per_timestep, uint64_t remainder_particles, MPI_Config *mpi_config, Mesh<T> *mesh, vec<T> start, T rate_mean, T angle_xy_mean, T angle_rot_mean, vec<T> vel_mean,
                                  vec<T> acc_mean, vec<T> jerk_mean, T decay_rate_mean, T decay_threshold_mean, T temp, ProbabilityDistribution dist) :
                                  wave_particles_per_timestep(wave_particles_per_timestep), even_particles_per_timestep(even_particles_per_timestep), remainder_particles(remainder_particles), mpi_config(mpi_config), mesh(mesh)
            {   
                // srand (time(NULL) + mpi_config->rank);
                srand (0 + mpi_config->rank);
                if (dist == UNIFORM)
                {
                    T var = 0.2;
                    start_pos        = new UniformDistribution<vec<T>>(start - start*(0.05),       start    + start*0.05);
                    velocity         = new UniformDistribution<vec<T>>(vel_mean - vel_mean*0.5, vel_mean + vel_mean*0.5);
                    acceleration     = new UniformDistribution<vec<T>>(acc_mean - acc_mean*(var), acc_mean + acc_mean*var);
                    temperature      = new UniformDistribution<T>(temp - temp*0.05, temp + temp*0.05);
                }
                else 
                {
                    start_pos        = new FixedDistribution<vec<T>>(start);
                    acceleration     = new FixedDistribution<vec<T>>(acc_mean);
                    velocity         = new FixedDistribution<vec<T>>(vel_mean);
                }
                rate             = new FixedDistribution<T>(rate_mean);           
                angle_xy         = new FixedDistribution<T>(angle_xy_mean);   
                angle_rot        = new FixedDistribution<T>(angle_rot_mean);  
                jerk             = new FixedDistribution<vec<T>>(jerk_mean);
                decay_rate       = new FixedDistribution<T>(decay_rate_mean);  
                decay_threshold  = new FixedDistribution<T>(decay_threshold_mean);  
            }

            inline void emit_particles_waves(vector<Particle<T>>& particles, vector<unordered_map<uint64_t, uint64_t>>& cell_particle_field_map,  uint64_t **indexes, particle_aos<T> **indexed_fields, Particle_Logger *logger)
            {
                particle_aos<T> zero_field = (particle_aos<T>){(vec<T>){0.0, 0.0, 0.0}, 0.0, 0.0};
                uint64_t start_cell = mesh->mesh_size * 0.49;
                
                static int timestep_count = 0;
                timestep_count++;

                uint64_t elements [mesh->num_blocks] = {0};


                if ( (timestep_count++ % mpi_config->particle_flow_world_size) == mpi_config->particle_flow_rank )
                {
                    for (uint64_t p = 0; p < wave_particles_per_timestep; p++)
                    {
                        const Particle<T> particle = Particle<T>(mesh, start_pos->get_value(), velocity->get_scaled_value(), acceleration->get_value(), temperature->get_value(), start_cell, logger);

                        if (particle.decayed) 
                        {
                            p -= 1;
                            logger->decayed_particles --;
                            continue;
                        }

                        start_cell = particle.cell; 
                        particles.push_back(particle);
                        
                        const uint64_t block_id = mesh->get_block_id(particle.cell);
                        const uint64_t index    = cell_particle_field_map[block_id].size();

                        elements[block_id] = cell_particle_field_map[block_id].size() + 1;

                        if ( !cell_particle_field_map[block_id].contains(particle.cell) )
                        {
                            cell_particle_field_map[block_id][particle.cell] = index;

                            resize_fn(elements, &indexes, &indexed_fields);
                            
                            indexes[block_id][index]                 = particle.cell;
                            indexed_fields[block_id][index]          = zero_field;
                        }
                    }
                }

                logger->num_particles      += wave_particles_per_timestep ;
                logger->emitted_particles  += wave_particles_per_timestep ;
            }

            inline void emit_particles_evenly(vector<Particle<T>>& particles, vector<unordered_map<uint64_t, uint64_t>>& cell_particle_field_map,  uint64_t **indexes, particle_aos<T> **indexed_fields, function<void(uint64_t*, uint64_t ***, particle_aos<T> ***)> resize_fn, Particle_Logger *logger)
            {
                particle_aos<T> zero_field = (particle_aos<T>){(vec<T>){0.0, 0.0, 0.0}, 0.0, 0.0};
                uint64_t start_cell = mesh->mesh_size * 0.49;
                
                static int timestep_count = 0;
                timestep_count++;
                uint64_t remainder = ((mpi_config->particle_flow_rank + timestep_count*remainder_particles) % mpi_config->particle_flow_world_size) < remainder_particles;


                uint64_t elements [mesh->num_blocks] = {0};

                for (uint64_t p = 0; p < even_particles_per_timestep + remainder; p++)
                {
                    // printf("Rank %d trying new particle %lu\n", mpi_config->rank, p);
                    const Particle<T> particle = Particle<T>(mesh, start_pos->get_value(), velocity->get_scaled_value(), acceleration->get_value(), temperature->get_value(), start_cell, logger);
                    // printf("Rank %d trying new particle %lu decayed %d\n", mpi_config->rank, p, particle.decayed);

                    if (particle.decayed) 
                    {
                        p -= 1;
                        logger->decayed_particles--;
                        continue;
                    }

                    start_cell = particle.cell; 
                    particles.push_back(particle);
                    

                    const uint64_t block_id = mesh->get_block_id(particle.cell);

                    elements[block_id] = cell_particle_field_map[block_id].size() + 1;

                    if ( !cell_particle_field_map[block_id].contains(particle.cell) )
                    {
                        const uint64_t index    = cell_particle_field_map[block_id].size();

                        resize_fn(elements, &indexes, &indexed_fields);
                        
                        // printf("Rank %d block %lu particle is in cell %lu\n", mpi_config->rank, block_id, particle.cell);
                        
                        indexes[block_id][index]                 = particle.cell;
                        indexed_fields[block_id][index]          = zero_field;

                        cell_particle_field_map[block_id][particle.cell] = index;
                    }
                }
                


                logger->num_particles      += even_particles_per_timestep + remainder;
                logger->emitted_particles  += even_particles_per_timestep + remainder;
            }

    }; // class ParticleDistribution
 
}   // namespace minicombust::particles 
