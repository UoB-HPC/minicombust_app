#include "examples/particle_examples.hpp"

#include <cstdint>

using namespace minicombust::particles;
using namespace minicombust::utils;

ParticleDistribution<double> *load_particle_distribution(uint64_t particles_per_timestep, uint64_t local_particles_per_timestep, uint64_t remainder_particles, MPI_Config *mpi_config, Mesh<double> *mesh)
{
    vec<double> position     = {0.005, 0.025, 0.025};
    vec<double> velocity     = {103.,    5.,   5.};
    vec<double> acceleration = {   0.,   0.,   0.};
    double temp              = 300.;

    ParticleDistribution<double> *particle_dist = new ParticleDistribution<double>(particles_per_timestep, local_particles_per_timestep, remainder_particles, mpi_config, mesh, position, velocity, acceleration, temp, UNIFORM);
    return particle_dist;
}

ParticleDistribution<double> *load_injector_particle_distribution(uint64_t particles_per_timestep, uint64_t local_particles_per_timestep, uint64_t remainder_particles, MPI_Config *mpi_config, vec<double> box_dim, Mesh<double> *mesh)
{
    double      inner_injector_radius = 0.05*box_dim.x;
    double      outer_injector_radius = 0.06*box_dim.x;
    vec<double> injector_position     = {box_dim.x/10.0, box_dim.y/2.0, box_dim.z/2.0};

    vec<double> cyclindrical_velocity_mean = {7.5*box_dim.x, M_PI, 75*box_dim.y};

    double temp = 300.;

    ParticleDistribution<double> *particle_dist = new ParticleDistribution<double>(particles_per_timestep, local_particles_per_timestep, remainder_particles, mpi_config, mesh, injector_position, inner_injector_radius, outer_injector_radius, cyclindrical_velocity_mean, temp, UNIFORM);
    return particle_dist;
}
