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

            uint64_t mesh_points_size;  // Number of points in the mesh
            uint64_t mesh_size;         // Number of polygons in the mesh
            uint64_t cell_size;         // Number of points in the cell

            vec<T> *mesh_points;        // Mesh Points
            vec<T> ***mesh_cells;       // Array of [cells_vertex_count*mesh_point pointers

            vec<T> *mesh_cells_centres; // Cell centres

            Mesh(uint64_t points_size, uint64_t mesh_size, uint64_t cell_size, vec<T> *points, vec<T> ***cells) : mesh_points_size(points_size), mesh_size(mesh_size), cell_size(cell_size),
                                                                                                                  mesh_points(points), mesh_cells(cells)
            {
                // Allocate space for and calculate cell centre co-ordinates
                mesh_cells_centres = (vec<T> *)malloc(mesh_size * sizeof(vec<T>));
                printf("Allocating mesh cell centres array, %llu points (%.2f MB)\n\n", mesh_size,
                                                                              (float)(mesh_size * sizeof(vec<T>))/1000000.0);
                calculate_cell_centres();
            }

    }; // class Mesh

}   // namespace minicombust::particles 
