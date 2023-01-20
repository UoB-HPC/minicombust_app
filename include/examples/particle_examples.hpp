#pragma once

#include "particles/ParticleDistribution.hpp"
#include "geometry/Mesh.hpp"
#include "utils/utils.hpp"

using namespace minicombust::particles;

ParticleDistribution<double> *load_particle_distribution(uint64_t particles_per_timestep, uint64_t local_particles_per_timestep, uint64_t remainder_particles, MPI_Config *mpi_config, Mesh<double> *mesh);
ParticleDistribution<double> *load_injector_particle_distribution(uint64_t particles_per_timestep, uint64_t local_particles_per_timestep, uint64_t remainder_particles, MPI_Config *mpi_config, Mesh<double> *mesh);