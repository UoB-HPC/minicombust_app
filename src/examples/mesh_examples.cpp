#include "examples/mesh_examples.hpp"

#include <cstdint>
#include <string>  
#include <memory.h>

using namespace minicombust::geometry;
using namespace minicombust::utils;

using namespace std;


Mesh<double> *load_mesh(double mesh_dim, uint64_t elements_per_dim)
{
    const uint64_t cell_size  = 8; // Cube
    const uint64_t faces_per_cell = 6; // Cube

    
    const double mesh_x_dim = mesh_dim; 
    const double mesh_y_dim = mesh_dim;   
    const double mesh_z_dim = mesh_dim;

    const uint64_t elements_per_x_dim = elements_per_dim; // Number of cubes for each dimension
    const uint64_t elements_per_y_dim = elements_per_dim;
    const uint64_t elements_per_z_dim = elements_per_dim;

    const int z_points = elements_per_z_dim + 1;
    const int y_points = elements_per_y_dim + 1;
    const int x_points = elements_per_x_dim + 1;

    const double element_x_dim = mesh_x_dim / elements_per_x_dim;
    const double element_y_dim = mesh_y_dim / elements_per_y_dim;
    const double element_z_dim = mesh_z_dim / elements_per_z_dim;
 
    const double num_points       = z_points * y_points * x_points;   
    const double num_cubes        = elements_per_x_dim * elements_per_y_dim * elements_per_z_dim;


    // Create array of all the points we'll need.
    vec<double> *points = new vec<double>[num_points];
    vec<double> current_point = {0, 0, 0};
    for (int z = 0; z < z_points; z++)
    {
        current_point.y = 0;
        for (int y = 0; y < y_points; y++)
        {
            current_point.x = 0;
            for (int x = 0; x < x_points; x++)
            {
                int index = z*x_points*y_points + y*x_points + x;
                memcpy(points + index, &current_point, sizeof(vec<double>));
                current_point = current_point + vec<double>{element_x_dim, 0, 0};
            }
            current_point = current_point + vec<double>{0, element_y_dim, 0};
        }
        current_point = current_point + vec<double>{0, 0, element_z_dim};
    }


    // Create array of cube cells
    uint64_t *cubes = (uint64_t *)malloc(num_cubes*cell_size*sizeof(uint64_t));

    for (int z = 0; z < elements_per_z_dim; z++)
    {
        for (int y = 0; y < elements_per_y_dim; y++)
        {
            for (int x = 0; x < elements_per_x_dim; x++)
            {
                int index         = z * x_points * y_points + y * x_points + x;
                int cube_index    = z*elements_per_x_dim*elements_per_y_dim + y*elements_per_x_dim + x;

                cubes[cube_index*cell_size + A_VERTEX] = index;
                cubes[cube_index*cell_size + B_VERTEX] = index + 1;
                cubes[cube_index*cell_size + C_VERTEX] = index + x_points;
                cubes[cube_index*cell_size + D_VERTEX] = index + x_points + 1;
                cubes[cube_index*cell_size + E_VERTEX] = index + x_points*y_points;
                cubes[cube_index*cell_size + F_VERTEX] = index + x_points*y_points + 1;
                cubes[cube_index*cell_size + G_VERTEX] = index + x_points*y_points + x_points;
                cubes[cube_index*cell_size + H_VERTEX] = index + x_points*y_points + x_points+1;
            }
        }
    }


    
    // Create array of faces, each face is a pointer to two neighbouring cells.
    const uint64_t faces_size = elements_per_x_dim*elements_per_y_dim*z_points + elements_per_y_dim*elements_per_z_dim*x_points + elements_per_z_dim*elements_per_x_dim*y_points;
    Face<double> *faces       = (Face<double> *)malloc(faces_size*sizeof(Face<double>));                      // Faces         = {{0, BOUNDARY}, {0, BOUNDARY}, {0, BOUNDARY}, {0, 1}, ...}; 

    uint64_t *cell_neighbours = (uint64_t*)malloc(num_cubes*faces_per_cell*sizeof(uint64_t));                 // Cell faces    = {{0, 1, 2, 3, 4, 5}, {6, 1, 7, 3, 8, 5}}


    uint64_t faces_count = 0;
    for (uint64_t c = 0; c < num_cubes; c++)
    {
        // Assuming we iterate through the cells, front -> back, left -> right, and down -> up, back/right/up faces of the cells must be created.
        // Front/left/down faces are only created if they are on the edge of a grid, and therefore, no cells have created them.

        // FRONT
        if ( c < elements_per_x_dim*elements_per_y_dim )  {
            faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
            cell_neighbours[c*faces_per_cell + FRONT_FACE] = MESH_BOUNDARY;
        }                             
        else
        {
            cell_neighbours[c*faces_per_cell + FRONT_FACE] = c - elements_per_x_dim*elements_per_y_dim;
        }

        // BACK
        if ( c >= (elements_per_z_dim-1)*elements_per_x_dim*elements_per_y_dim )  {
            faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
            cell_neighbours[c*faces_per_cell + BACK_FACE]  = MESH_BOUNDARY;
        }                             
        else
        {
            faces[faces_count++]                           = Face<double>(c, c + elements_per_x_dim*elements_per_y_dim);
            cell_neighbours[c*faces_per_cell + BACK_FACE]  = c + elements_per_x_dim*elements_per_y_dim;
        }



        // LEFT
        if ( (c % elements_per_x_dim) == 0 )  {
            faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
            cell_neighbours[c*faces_per_cell + LEFT_FACE]  = MESH_BOUNDARY;
        }                             
        else
        {
            cell_neighbours[c*faces_per_cell + LEFT_FACE]  = c - 1;
        }

        // RIGHT
        if ( ((c+1) % elements_per_x_dim) == 0 )  {
            faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
            cell_neighbours[c*faces_per_cell + RIGHT_FACE] = MESH_BOUNDARY;
        }                             
        else
        {
            faces[faces_count++]                           = Face<double>(c, c + 1);
            cell_neighbours[c*faces_per_cell + RIGHT_FACE] = c + 1;
        }



        // DOWN
        if ( (c % (elements_per_x_dim*elements_per_y_dim)) < elements_per_x_dim )  {
            faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
            cell_neighbours[c*faces_per_cell + DOWN_FACE]  = MESH_BOUNDARY;
        }                             
        else
        {
            cell_neighbours[c*faces_per_cell + DOWN_FACE]  = c - elements_per_x_dim;
        }

        // UP
        if ( ((c+elements_per_x_dim) % (elements_per_x_dim*elements_per_y_dim)) < elements_per_x_dim )  {
            faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
            cell_neighbours[c*faces_per_cell + UP_FACE]    = MESH_BOUNDARY;
        }                             
        else
        {
            faces[faces_count++]                           = Face<double>(c, c + elements_per_x_dim);
            cell_neighbours[c*faces_per_cell + UP_FACE]    = c + elements_per_x_dim;
        }
    }

    Mesh<double> *mesh = new Mesh<double>(num_points, num_cubes, cell_size, faces_size, faces_per_cell, points, cubes, faces, cell_neighbours);

    return mesh;
}
