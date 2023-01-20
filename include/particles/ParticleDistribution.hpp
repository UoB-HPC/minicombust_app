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
        private:
            vec<T> to_cartesian ( vec<T> cylindrical_vec )
            {
                vec<T> cartesian_vec;
                cartesian_vec.y = cylindrical_vec.x * cos(cylindrical_vec.y);
                cartesian_vec.z = cylindrical_vec.x * sin(cylindrical_vec.y);
                cartesian_vec.x = cylindrical_vec.z;

                return cartesian_vec;
            }

        public:
            bool cylindrical = false;

            // TODO: Change this to particles per unit of time?
            uint64_t wave_particles_per_timestep;
            uint64_t even_particles_per_timestep;
            uint64_t remainder_particles;

            MPI_Config *mpi_config;

            Mesh<T> *mesh;

            vec<T> injector_position;


            Distribution<vec<T>> *start_pos;
            Distribution<vec<T>> *velocity;
            Distribution<vec<T>> *acceleration;
            Distribution<T>      *temperature; 

            Distribution<vec<T>> *cyclindrical_position;
            Distribution<vec<T>> *cyclindrical_velocity;
            

            // Generate fixed distribution
            ParticleDistribution (uint64_t wave_particles_per_timestep, uint64_t even_particles_per_timestep, uint64_t remainder_particles, MPI_Config *mpi_config, Mesh<T> *mesh, vec<T> start, vec<T> vel_mean, vec<T> acc_mean, T temp, ProbabilityDistribution dist) :
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
            }

            // Generate fixed distribution
            ParticleDistribution (uint64_t wave_particles_per_timestep, uint64_t even_particles_per_timestep, uint64_t remainder_particles, MPI_Config *mpi_config, Mesh<T> *mesh, vec<T> injector_position, double inner_injector_radius, double outer_injector_radius, vec<T> cyclindrical_velocity_mean, T temp, ProbabilityDistribution dist) :
                                  wave_particles_per_timestep(wave_particles_per_timestep), even_particles_per_timestep(even_particles_per_timestep), remainder_particles(remainder_particles), mpi_config(mpi_config), mesh(mesh), injector_position(injector_position)
            {   
                // srand (time(NULL) + mpi_config->rank);
                srand (0 + mpi_config->rank);
                cylindrical = true;

                if (dist == UNIFORM)
                {
                    vec<T> cyclindrical_velocity_lower = cyclindrical_velocity_mean - 0.05 * cyclindrical_velocity_mean;
                    vec<T> cyclindrical_velocity_upper = cyclindrical_velocity_mean + 0.05 * cyclindrical_velocity_mean;
                    cyclindrical_velocity_lower.y = 0;
                    cyclindrical_velocity_upper.y = 2 * M_PI;


                    cyclindrical_velocity_lower.y = (mpi_config->particle_flow_rank + 0) * (2 * M_PI / (mpi_config->particle_flow_world_size + 1));
                    cyclindrical_velocity_upper.y = (mpi_config->particle_flow_rank + 1) * (2 * M_PI / (mpi_config->particle_flow_world_size + 1));


                    vec<T> cyclindrical_position_lower = { inner_injector_radius,      0.0, -0.0001 }; 
                    vec<T> cyclindrical_position_upper = { outer_injector_radius, 2 * M_PI, +0.0001 }; 

                    cyclindrical_velocity = new UniformDistribution<vec<T>>( cyclindrical_velocity_lower, cyclindrical_velocity_upper );
                    cyclindrical_position = new UniformDistribution<vec<T>>( cyclindrical_position_lower, cyclindrical_position_upper );

                    acceleration     = new UniformDistribution<vec<T>>(vec<T> {0.0, 0.0, 0.0}, vec<T> {0.0, 0.0, 0.0});
                    temperature      = new UniformDistribution<T>(temp - temp*0.05, temp + temp*0.05);
                }
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

            inline void emit_particles_evenly(vector<Particle<T>>& particles, vector<unordered_map<uint64_t, uint64_t>>& cell_particle_field_map, unordered_map<uint64_t, flow_aos<T> *>& node_to_field_address_map,  uint64_t **indexes, particle_aos<T> **indexed_fields, function<void(uint64_t*, uint64_t ***, particle_aos<T> ***)> resize_fn, Particle_Logger *logger)
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
                    const Particle<T> particle = (!cylindrical) ? Particle<T>(mesh, start_pos->get_value(),                                               velocity->get_scaled_value(),                     acceleration->get_value(), temperature->get_value(), start_cell, logger) :
                                                                  Particle<T>(mesh, injector_position + to_cartesian(cyclindrical_position->get_value()), to_cartesian(cyclindrical_velocity->get_value()), acceleration->get_value(), temperature->get_value(), start_cell, logger);
                    // printf("Rank %d trying new particle %lu decayed %d\n", mpi_config->rank, p, particle.decayed);

                    // cout << "Particle created at position " << print_vec(particle.x1) << " with velocity " << print_vec(particle.v1) << " with acc " << print_vec(particle.a1) << " decayed " << particle.decayed << " cell " << " temp " << particle.temp << particle.cell << endl;

                    if (particle.decayed) 
                    {
                        p -= 1;
                        logger->decayed_particles--;
                        continue;
                    }

                    start_cell = particle.cell; 
                    particles.push_back(particle);

                    const uint64_t block_id = mesh->get_block_id(particle.cell);

                    if ( !cell_particle_field_map[block_id].contains(particle.cell) )
                    {
                        elements[block_id]   = cell_particle_field_map[block_id].size() + 1;
                        const uint64_t index = cell_particle_field_map[block_id].size();

                        resize_fn(elements, &indexes, &indexed_fields);
                        
                        // printf("Rank %d block %lu particle is in cell %lu\n", mpi_config->rank, block_id, particle.cell);
                        
                        indexes[block_id][index]                 = particle.cell;
                        indexed_fields[block_id][index]          = zero_field;

                        cell_particle_field_map[block_id][particle.cell] = index;

                        #pragma ivdep
                        for (uint64_t n = 0; n < mesh->cell_size; n++)
                        {
                            const uint64_t node_id = mesh->cells[(particle.cell - mesh->shmem_cell_disp) * mesh->cell_size + n];
                            
                            if (!node_to_field_address_map.contains(node_id))
                                node_to_field_address_map[node_id] = nullptr;
                        }
                    }
                }

                logger->num_particles      += even_particles_per_timestep + remainder;
                logger->emitted_particles  += even_particles_per_timestep + remainder;
            }

    }; // class ParticleDistribution
 
}   // namespace minicombust::particles 
