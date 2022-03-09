#pragma once

#include <map>

#include "utils/utils.hpp"

using namespace minicombust::utils; 



namespace minicombust::geometry 
{   

    enum FACE_DIRECTIONS { FRONT_FACE = 0, BACK_FACE = 1, LEFT_FACE = 2, RIGHT_FACE = 3, DOWN_FACE = 4, UP_FACE = 5};
    enum CUBE_VERTEXES { A_VERTEX = 0, B_VERTEX = 1, C_VERTEX = 2, D_VERTEX = 3, E_VERTEX = 4, F_VERTEX = 5, G_VERTEX = 6, H_VERTEX = 7};

    const uint64_t CUBE_FACE_VERTEX_MAP[6][4] = 
    {
        {A_VERTEX, B_VERTEX, C_VERTEX, D_VERTEX}, // FRONT FACE
        {E_VERTEX, F_VERTEX, G_VERTEX, H_VERTEX}, // BACK FACE
        {A_VERTEX, C_VERTEX, E_VERTEX, G_VERTEX}, // LEFT FACE
        {B_VERTEX, D_VERTEX, F_VERTEX, H_VERTEX}, // RIGHT FACE
        {A_VERTEX, B_VERTEX, E_VERTEX, F_VERTEX}, // DOWN FACE
        {C_VERTEX, D_VERTEX, G_VERTEX, H_VERTEX}, // UP FACE
    };

    template<class T>
    class Face
    {
        private:


        public:
            vec<T> **cell0;
            vec<T> **cell1;

            // TODO: Allow some way of toggling, if cell asks for face, get other face etc

            Face(vec<T> **cell0, vec<T> **cell1) : cell0(cell0), cell1(cell1)
            { }
            
    }; // class Face

    template<class T>
    class Mesh 
    {
        private:

          // Calculate the centre point in each cell
          // Computed as the average of all vertex positions
          void calculate_cell_centres(void) {

              for (uint64_t c = 0; c < mesh_size; ++c) {
                  mesh_cells_centres[c] = vec<T>{0.0, 0.0, 0.0};
                  for (uint32_t i = 0; i < cells_vertex_count; ++i) {
                      mesh_cells_centres[c] += *cells[c][i];
                  }
                  mesh_cells_centres[c] /= static_cast<T>(cells_vertex_count);
              }
          }
 
        public:
            const uint32_t cells_vertex_count = 3; // Generic, 3 = triangle etc

            uint64_t mesh_points_size;    // Number of points in the mesh
            uint64_t mesh_size;           // Number of polygons in the mesh
            uint64_t cell_size;           // Number of points in the cell
            uint64_t faces_size;          // Number of faces in the cell

            vec<T> *mesh_points;          // Mesh Points
            vec<T> ***cells;              // Array of cell_size*mesh_point vertex pointers

            Face<T> ***mesh_faces;         // Array of faces_size*mesh_point face pointers

            uint64_t *particles_per_point; // Number of particles in each cell

            vec<T> *mesh_cells_centres;   // Cell centres

            Mesh(uint64_t points_size, uint64_t mesh_size, uint64_t cell_size, uint64_t faces_size, vec<T> *points, vec<T> ***cells, Face<T> ***faces) : mesh_points_size(points_size), mesh_size(mesh_size), cell_size(cell_size), faces_size(faces_size),
                                                                                                                                                        mesh_points(points), cells(cells), mesh_faces(faces)
            {
                // Allocate space for and calculate cell centre co-ordinates
                const size_t mesh_cell_centre_size = mesh_size * sizeof(vec<T>);
                mesh_cells_centres = (vec<T> *)malloc(mesh_cell_centre_size);
                printf("Allocating %llu mesh cell centre points (%.2f MB)\n", mesh_size, (float)(mesh_cell_centre_size)/1000000.0);
                calculate_cell_centres();

                const size_t particles_per_point_size = mesh_points_size * sizeof(uint64_t);
                particles_per_point = (uint64_t *)malloc(particles_per_point_size);
                printf("Allocating array of particles per cell (%.2f MB)\n", (float)(particles_per_point_size)/1000000.0);

                const size_t points_array_size = mesh_points_size*sizeof(vec<double>);
                const size_t cells_array_size  = mesh_size*cell_size*sizeof(vec<double> *);
                const size_t faces_array_size  = mesh_size*faces_size*sizeof(Face<T>);
                printf("Allocating %llu vertexes (%.2f MB)\n", mesh_points_size,      (float)(points_array_size)/1000000.0);
                printf("Allocating %llu cells (%.2f MB)\n",    mesh_size,             (float)(cells_array_size)/1000000.0);
                printf("Allocating %llu faces (%.2f MB)\n",    mesh_size*faces_size,  (float)(faces_array_size)/1000000.0);

                const size_t total_size = mesh_cell_centre_size + particles_per_point_size + points_array_size + cells_array_size;

                printf("Allocated mesh. Total size (%.2f MB)\n\n", (float)total_size/1000000.0);
            }

            void clear_particles_per_point_array(void)
            {
                memset(particles_per_point, 0, mesh_points_size * sizeof(uint64_t));
            }

    }; // class Mesh

}   // namespace minicombust::particles 
