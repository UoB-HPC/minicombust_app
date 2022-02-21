#pragma once

#include "geometry/Mesh.hpp"
#include "utils/utils.hpp"

using namespace minicombust::geometry;


Mesh<double> *load_boundary_box_mesh(double box_size);

// <typename T>
Mesh<double> *load_global_mesh(double mesh_dim, int elements_per_dim);