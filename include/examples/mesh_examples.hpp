#pragma once

#include "geometry/Mesh.hpp"
#include "utils/utils.hpp"

using namespace minicombust::geometry;


Mesh<double> *load_boundary_box_mesh(double box_size);

Mesh<double> *load_mesh(MPI_Config *mpi_config, double mesh_dim, uint64_t elements_per_dim);