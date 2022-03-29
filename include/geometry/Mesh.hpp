#pragma once

#include <map>

#include "utils/utils.hpp"

using namespace minicombust::utils; 

namespace minicombust::geometry 
{   
    static const uint64_t MESH_BOUNDARY = UINT64_MAX;

    enum FACE_DIRECTIONS { FRONT_FACE = 0, BACK_FACE = 1, LEFT_FACE = 2, RIGHT_FACE = 3, DOWN_FACE = 4, UP_FACE = 5};
    enum CUBE_VERTEXES { A_VERTEX = 0, B_VERTEX = 1, C_VERTEX = 2, D_VERTEX = 3, E_VERTEX = 4, F_VERTEX = 5, G_VERTEX = 6, H_VERTEX = 7};

    static const uint64_t CUBE_FACE_VERTEX_MAP[6][4] = 
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
            uint64_t cell0;
            uint64_t cell1;

            Face(uint64_t cell0, uint64_t cell1) : cell0(cell0), cell1(cell1)
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
                  cell_centres[c] = vec<T>{0.0, 0.0, 0.0};
                  for (uint32_t i = 0; i < cells_vertex_count; ++i) {
                      cell_centres[c] += points[cells[c*cell_size + i]];
                  }
                  cell_centres[c] /= static_cast<T>(cells_vertex_count);
              }
          }
 
        public:
            const uint32_t cells_vertex_count = 3; // Generic, 3 = triangle etc

            uint64_t points_size;         // Number of points in the mesh
            uint64_t mesh_size;           // Number of polygons in the mesh
            uint64_t cell_size;           // Number of points in the cell
            uint64_t faces_size;          // Number of unique faces in the mesh
            uint64_t faces_per_cell;      // Number of faces in a cell
            
            
            vec<T> cell_size_vector;      // Cell size


            vec<T> *points;               // Mesh points    = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, ...}:

            uint64_t *cells;              // Cells          = {{0, 1, 2, 300, 40, 36, 7, 2}, {1, 2, 4, 300}, ...};
            Face<T> *faces;               // Faces          = {{0, BOUNDARY}, {0, BOUNDARY}, {0, BOUNDARY}, {0, 1}, ...}; 
            vec<T> *cell_centres;         // Cell centres   = {{0.5, 3.0, 4.0}, {2.5, 3.0, 4.0}, ...};
            uint64_t *cell_neighbours;    // Cell faces     = {{0, 1, 2, 3, 4, 5}, {6, 1, 7, 3, 8, 5}}
 

            uint64_t *particles_per_point; // Number of particles in each cell


            Mesh(uint64_t points_size, uint64_t mesh_size, uint64_t cell_size, uint64_t faces_size, uint64_t faces_per_cell, vec<T> *points, uint64_t *cells, Face<T> *faces, uint64_t *cell_neighbours) 
            : points_size(points_size), mesh_size(mesh_size), cell_size(cell_size), faces_size(faces_size), faces_per_cell(faces_per_cell), points(points), cells(cells), faces(faces), cell_neighbours(cell_neighbours)
            {
                // Allocate space for and calculate cell centre co-ordinates
                const size_t mesh_cell_centre_size = mesh_size * sizeof(vec<T>);
                cell_centres = (vec<T> *)malloc(mesh_cell_centre_size);
                printf("\nMesh storage requirements:\n\tAllocating %llu mesh cell centre points (%.2f MB)\n", mesh_size, (float)(mesh_cell_centre_size)/1000000.0);
                calculate_cell_centres();

                const size_t particles_per_point_size = points_size * sizeof(uint64_t);
                particles_per_point = (uint64_t *)malloc(particles_per_point_size);
                printf("\tAllocating array of particles per cell (%.2f MB)\n", (float)(particles_per_point_size)/1000000.0);

                const size_t points_array_size           = points_size*sizeof(vec<double>);
                const size_t cells_array_size            = mesh_size*cell_size*sizeof(uint64_t);
                const size_t faces_array_size            = faces_size*sizeof(Face<T>);
                const size_t cell_neighbours_array_size  = mesh_size*faces_per_cell*sizeof(uint64_t);
                printf("\tAllocating %llu vertexes (%.2f MB)\n",          points_size,               (float)(points_array_size)/1000000.0);
                printf("\tAllocating %llu cells (%.2f MB)\n",             mesh_size,                 (float)(cells_array_size)/1000000.0);
                printf("\tAllocating %llu faces (%.2f MB)\n",             faces_size,                (float)(faces_array_size)/1000000.0);
                printf("\tAllocating %llu cell neighbour indexes (%.2f MB)\n", mesh_size*faces_per_cell,  (float)(cell_neighbours_array_size)/1000000.0);

                const size_t total_size = mesh_cell_centre_size + particles_per_point_size + points_array_size + cells_array_size + faces_array_size + cell_neighbours_array_size;

                cell_size_vector = points[cells[H_VERTEX]] - points[cells[A_VERTEX]];

                printf("\tAllocated mesh. Total size (%.2f MB)\n\n", (float)total_size/1000000.0);
            }

            void clear_particles_per_point_array(void)
            {
                memset(particles_per_point, 0, points_size * sizeof(uint64_t));
            }


            string get_face_string(uint64_t face)
            {
                switch(face)  
                {
                    case FRONT_FACE:
                        return "FRONT";
                    case BACK_FACE:
                        return "BACK";
                    case LEFT_FACE:
                        return "LEFT";
                    case RIGHT_FACE:
                        return "RIGHT";
                    case DOWN_FACE:
                        return "DOWN";
                    case UP_FACE:
                        return "UP";
                default:
                        return "INVALID FACE"; 
                }
            }
            string get_vertex_string(uint64_t vertex)
            {
                switch(vertex)  
                {
                    case A_VERTEX:
                        return "A";
                    case B_VERTEX:
                        return "B";
                    case C_VERTEX:
                        return "C";
                    case D_VERTEX:
                        return "D";
                    case E_VERTEX:
                        return "E";
                    case F_VERTEX:
                        return "F";
                    case G_VERTEX:
                        return "G";
                    case H_VERTEX:
                        return "H";
                default:
                        return "INVALID VERTEX"; 
                }
            }

    }; // class Mesh

}   // namespace minicombust::particles 
