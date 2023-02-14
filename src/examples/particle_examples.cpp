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

ParticleDistribution<double> *load_injector_particle_distribution(uint64_t particles_per_timestep, uint64_t local_particles_per_timestep, uint64_t remainder_particles, MPI_Config *mpi_config, Mesh<double> *mesh)
{
    double      inner_injector_radius = 0.0016;
    double      outer_injector_radius = 0.002;
    vec<double> injector_position     = {0.005, 0.025, 0.025};

    vec<double> cyclindrical_velocity_mean = {120, M_PI, 100};

    double temp = 300.;

    ParticleDistribution<double> *particle_dist = new ParticleDistribution<double>(particles_per_timestep, local_particles_per_timestep, remainder_particles, mpi_config, mesh, injector_position, inner_injector_radius, outer_injector_radius, cyclindrical_velocity_mean, temp, UNIFORM);
    return particle_dist;
}