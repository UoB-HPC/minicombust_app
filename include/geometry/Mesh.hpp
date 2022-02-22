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

              for (uint64_t obj = 0; obj < mesh_obj_size; ++obj) {
                  mesh_obj_centres[obj] = vec<T>{0.0, 0.0, 0.0};
                  for (uint32_t i = 0; i < obj_vertex_count; ++i) {
                      mesh_obj_centres[obj] += *mesh_obj_vertexes[obj][i];
                  }
                  mesh_obj_centres[obj] /= static_cast<T>(obj_vertex_count);
              }
          }
 
        public:
            const uint32_t obj_vertex_count = 3; // Generic, 3 = triangle etc

            uint64_t mesh_points_size;  // Number of points in the mesh
            uint64_t mesh_obj_size;     // Number of polygons in the mesh

            vec<T> *mesh_points;          // Mesh Points
            vec<T> ***mesh_obj_vertexes;   // Array of [obj_vertex_count*mesh_point pointers]

            vec<T> *mesh_obj_centres;      // Cell centres

            Mesh(uint64_t points_size, uint64_t objects_size) : mesh_points_size(points_size), mesh_obj_size(objects_size)
            {
                mesh_points = (vec<T> *)malloc(mesh_points_size * (sizeof(vec<T>)));

                mesh_obj_vertexes = (vec<T> **)malloc(obj_vertex_count * mesh_obj_size * (sizeof(vec<T> *)));

                // Allocate space for and calculate cell centre co-ordinates
                mesh_obj_centres = (vec<T> *)malloc(mesh_obj_size * sizeof(vec<T>));
                calculate_cell_centres();
            }

            Mesh(uint64_t points_size, uint64_t objects_size, vec<T> *points, vec<T> ***objects) : mesh_points_size(points_size), mesh_obj_size(objects_size),
                                                                                                  mesh_points(points), mesh_obj_vertexes(objects)
            {
                // Allocate space for and calculate cell centre co-ordinates
                mesh_obj_centres = (vec<T> *)malloc(mesh_obj_size * sizeof(vec<T>));
                calculate_cell_centres();
            }

    }; // class Mesh

}   // namespace minicombust::particles 
