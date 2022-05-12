#include "examples/particle_examples.hpp"

#include <cstdint>

using namespace minicombust::particles;
using namespace minicombust::utils;

ParticleDistribution<double> *load_particle_distribution(uint64_t particles_per_timestep, Mesh<double> *mesh)
{
    vec<double> position     = {0.01, 0.01, 0.01};
    vec<double> velocity     = {103., 103., 103.};
    vec<double> acceleration = {   0.,   0.,   0.};
    vec<double> jerk         = {0.01, 0.01, 0.01};
    double rate              = 0.1;
    double xy_angle          = 0.1;   
    double angle_rotation    = 0.1;
    double decay_rate        = 0.1;
    double decay_threshold   = 0.1;
    double temp              = 300.;

    ParticleDistribution<double> *particle_dist = new ParticleDistribution<double>(particles_per_timestep, mesh, position, rate, xy_angle, angle_rotation, velocity,
                                                                                   acceleration, jerk, decay_rate, decay_threshold, temp, UNIFORM);
    return particle_dist;
}