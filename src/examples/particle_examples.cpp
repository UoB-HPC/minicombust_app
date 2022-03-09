#include "examples/particle_examples.hpp"

#include <cstdint>

using namespace minicombust::particles;
using namespace minicombust::utils;

ParticleDistribution<double> *load_particle_distribution(uint64_t particles_per_timestep, Mesh<double> *mesh)
{
    vec<double> position     = {5.0, 5.0, 5.0};
    vec<double> velocity     = {20.0, 6., 0.};
    vec<double> acceleration = {0.0, 0.0, 0.0};
    vec<double> jerk         = {0.1, 0.1, 0.1};
    double rate              = 0.1;
    double xy_angle          = 0.1;   
    double angle_rotation    = 0.1;
    double decay_rate        = 0.1;
    double decay_threshold   = 0.1;

    ParticleDistribution<double> *particle_dist = new ParticleDistribution<double>(particles_per_timestep, mesh, position, rate, xy_angle, angle_rotation, velocity,
                                                                                   acceleration, jerk, decay_rate, decay_threshold);
    return particle_dist;
}