#pragma once

#include "geometry/Mesh.hpp"
#include "particles/Particle.hpp"
#include "utils/utils.hpp"

using namespace minicombust::geometry;
using namespace minicombust::utils;


bool check_particle_posistion(Mesh<double> *mesh, vec<double> start, vec<double> velocity);

void run_particle_tests();