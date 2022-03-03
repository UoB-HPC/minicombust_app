#pragma once

#include <map>

#include "utils/utils.hpp"

using namespace minicombust::utils; 


namespace minicombust::geometry 
{   
    template<class T>
    class Mesh 
    {
        private:

          // Calculate the centre point in each cell
          // Computed as the average of all vertex positions
          void calculate_cell_centres(void) {

              for (uint64_t cells = 0; cells < mesh_size; ++cells) {
                  mesh_cells_centres[cells] = vec<T>{0.0, 0.0, 0.0};
                  for (uint32_t i = 0; i < cells_vertex_count; ++i) {
                      mesh_cells_centres[cells] += *mesh_cells[cells][i];
                  }
                  mesh_cells_centres[cells] /= static_cast<T>(cells_vertex_count);
              }
          }
 
        public:
            const uint32_t cells_vertex_count = 3; // Generic, 3 = triangle etc

            uint64_t mesh_points_size;    // Number of points in the mesh
            uint64_t mesh_size;           // Number of polygons in the mesh
            uint64_t cell_size;           // Number of points in the cell

            vec<T> *mesh_points;          // Mesh Points
            vec<T> ***mesh_cells;         // Array of [cells_vertex_count*mesh_point pointers

            uint64_t *particles_per_cell; // Number of particles in each cell

            vec<T> *mesh_cells_centres;   // Cell centres

            Mesh(uint64_t points_size, uint64_t mesh_size, uint64_t cell_size, vec<T> *points, vec<T> ***cells) : mesh_points_size(points_size), mesh_size(mesh_size), cell_size(cell_size),
                                                                                                                  mesh_points(points), mesh_cells(cells)
            {
                // Allocate space for and calculate cell centre co-ordinates
                const size_t mesh_cell_centre_size = mesh_size * sizeof(vec<T>);
                mesh_cells_centres = (vec<T> *)malloc(mesh_cell_centre_size);
                printf("Allocating %llu mesh cell centre points (%.2f MB)\n", mesh_size, (float)(mesh_cell_centre_size)/1000000.0);
                calculate_cell_centres();

                const size_t particles_per_cell_size = mesh_size * sizeof(uint64_t);
                particles_per_cell = (uint64_t *)malloc(particles_per_cell_size);
                printf("Allocating array of particles per cell (%.2f MB)\n", (float)(particles_per_cell_size)/1000000.0);

                const size_t points_array_size = mesh_points_size*sizeof(vec<double>);
                const size_t cells_array_size  = mesh_size*8*sizeof(vec<double> *);
                printf("Allocating %llu vertexes (%.2f MB)\n", mesh_points_size, (float)(points_array_size)/1000000.0);
                printf("Allocating %llu cells (%.2f MB)\n",    mesh_size,        (float)(cells_array_size)/1000000.0);

                const size_t total_size = mesh_cell_centre_size + particles_per_cell_size + points_array_size + cells_array_size;

                printf("Allocated mesh. Total size (%.2f MB)\n\n", (float)total_size/1000000.0);
            }

            void clear_particles_per_cell_array(void)
            {
                memset(particles_per_cell, 0, mesh_size * sizeof(uint64_t));
            }

    }; // class Mesh

}   // namespace minicombust::particles 
