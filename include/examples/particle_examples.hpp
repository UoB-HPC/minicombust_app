#pragma once

#include "particles/ParticleDistribution.hpp"
#include "geometry/Mesh.hpp"
#include "utils/utils.hpp"

using namespace minicombust::particles;

ParticleDistribution<double> *load_particle_distribution(uint64_t particles_per_timestep, Mesh<double> *mesh);