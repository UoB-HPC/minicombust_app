#include "examples/mesh_examples.hpp"

#include <cstdint>
#include <string>  


using namespace minicombust::geometry;
using namespace minicombust::utils;

using namespace std;

Mesh<double> *load_boundary_box_mesh(double box_size)
{
    const uint64_t cell_size = 3; // Triangle
    const double size  = box_size;

    const int num_points    = 8;
    const int num_triangles = 6 * 2;

    // Box vertices
    vec<double> A = {0,     0,    0};
    vec<double> B = {0,     size, 0};
    vec<double> C = {size,  size, 0};
    vec<double> D = {size,   0,   0};

    vec<double> E = {0,     0,    size};
    vec<double> F = {0,     size, size};
    vec<double> G = {size,  size, size};
    vec<double> H = {size,  0,    size};

    vec<double> *points              = new vec<double>[8]{A, B, C, D, E, F, G, H};

    // Front face
    vec<double> **triangle_1  = new vec<double>*[3]{&A, &B, &C};
    vec<double> **triangle_2  = new vec<double>*[3]{&A, &D, &C};

    // Back face
    vec<double> **triangle_3  = new vec<double>*[3]{&E, &F, &G};
    vec<double> **triangle_4  = new vec<double>*[3]{&E, &H, &G};

    // Left face
    vec<double> **triangle_5  = new vec<double>*[3]{&A, &B, &F};
    vec<double> **triangle_6  = new vec<double>*[3]{&A, &E, &F};

    // Right face
    vec<double> **triangle_7  = new vec<double>*[3]{&D, &C, &G};
    vec<double> **triangle_8  = new vec<double>*[3]{&D, &H, &G};

    // doubleop face
    vec<double> **triangle_9  = new vec<double>*[3]{&B, &C, &G};
    vec<double> **triangle_10 = new vec<double>*[3]{&B, &F, &G};

    // Bottom face
    vec<double> **triangle_11 = new vec<double>*[3]{&A, &E, &H};
    vec<double> **triangle_12 = new vec<double>*[3]{&A, &D, &H};

    vec<double> ***triangle_vertexes = new vec<double>**[]{triangle_1,  triangle_2,  triangle_3,
                                                           triangle_4,  triangle_5,  triangle_6,
                                                           triangle_7,  triangle_8,  triangle_9,
                                                           triangle_10, triangle_11, triangle_12};

    Face<double> ***faces = new Face<double>**[num_triangles*1];

    Mesh<double> *mesh = new Mesh<double>(num_points, num_triangles, cell_size, 1, points, triangle_vertexes, faces);

    return mesh;
}

Mesh<double> *load_global_mesh(double mesh_dim, uint64_t elements_per_dim)
{
    const uint64_t cell_size  = 8; // Cube
    const uint64_t faces_size = 6; // Cube

    
    const double mesh_x_dim = mesh_dim; 
    const double mesh_y_dim = mesh_dim;   
    const double mesh_z_dim = mesh_dim;

    const uint64_t elements_x_dim = elements_per_dim; // Number of cubes for each dimension
    const uint64_t elements_y_dim = elements_per_dim;
    const uint64_t elements_z_dim = elements_per_dim;

    const int z_points = elements_z_dim + 1;
    const int y_points = elements_y_dim + 1;
    const int x_points = elements_x_dim + 1;

    const double element_x_dim = mesh_x_dim / elements_x_dim;
    const double element_y_dim = mesh_y_dim / elements_y_dim;
    const double element_z_dim = mesh_z_dim / elements_z_dim;
 
    const double num_points       = z_points * y_points * x_points;   
    const double num_cubes        = elements_x_dim * elements_y_dim * elements_z_dim;


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
    vec<double> ***cubes = new vec<double>**[num_cubes];
    for (int z = 0; z < elements_z_dim; z++)
    {
        for (int y = 0; y < elements_y_dim; y++)
        {
            for (int x = 0; x < elements_x_dim; x++)
            {
                int index   = z * x_points * y_points + y * x_points + x;
                int cube_index    = z*elements_x_dim*elements_y_dim + y*elements_x_dim + x;

                // Add cell, contains 8 pointers to the 8 vertexes of the cube
                cubes[cube_index] = new vec<double>*[cell_size]{&(points[index]), &(points[index+1]), &(points[index+x_points]), &(points[index+x_points+1]), 
                                                                &(points[index+x_points*y_points]), &(points[index+x_points*y_points+1]), 
                                                                &(points[index+x_points*y_points+x_points]), &(points[index+x_points*y_points+x_points+1])};
            }
        }
    }


    
    // Create array of faces, each face is a pointer to two neighbouring cells.
    // TODO: Does it matter that faces are stored in the order: XY faces, YZ faces, XZ faces? Cache purposes. Makes sense to have duplicates for this. Every cell has face pointers nearby.
    Face<double> ***faces = new Face<double>**[num_cubes];
    
    
    for (int c = 0; c < num_cubes; c++)
    {
        // Assuming we iterate through the cells, front -> back, left -> right, and down -> up, back/right/up faces of the cells must be created.
        // Front/left/down faces are only created if they are on the edge of a grid, and therefore, no cells have created them.
        

        // Better to have cell0 as nullptr for checks?

        Face<double> *front;
        if ( c < elements_x_dim*elements_y_dim )                                     front = new Face<double>(nullptr, cubes[c]);
        else                                                                         front = faces[c - elements_x_dim*elements_y_dim][BACK_FACE];

        Face<double> *back;
        if ( c >= (elements_z_dim-1)*elements_x_dim*elements_y_dim )                 back = new Face<double>(nullptr, cubes[c]);
        else                                                                         back = new Face<double>(cubes[c], cubes[c + elements_x_dim*elements_y_dim]);

        Face<double> *left;
        if ( c % elements_x_dim == 0 )                                               left = new Face<double>(nullptr, cubes[c]);
        else                                                                         left = faces[c - 1][RIGHT_FACE];

        Face<double> *right;
        if ( (c+1) % elements_x_dim == 0 )                                           right = new Face<double>(nullptr,  cubes[c]);
        else                                                                         right = new Face<double>(cubes[c], cubes[c + 1]);

        Face<double> *down;
        if ( c % (elements_x_dim*elements_y_dim) < element_x_dim )                   down = new Face<double>(nullptr, cubes[c]);
        else                                                                         down = faces[c - elements_x_dim][UP_FACE];

        Face<double> *up;
        if ( (c+elements_x_dim) % (elements_x_dim*elements_y_dim) < element_x_dim )  up = new Face<double>(nullptr, cubes[c]);
        else                                                                         up = new Face<double>(cubes[c + elements_x_dim], cubes[c]);


        faces[c]  = new Face<double>*[faces_size]{front, back, left, right, down, up};
    }

    // const uint64_t total_faces    = elements_x_dim*elements_y_dim*z_points + elements_y_dim*elements_z_dim*x_points + elements_z_dim*elements_x_dim*y_points; // Without duplicates


    
    Mesh<double> *mesh = new Mesh<double>(num_points, num_cubes, cell_size, faces_size, points, cubes, faces);

    return mesh;
}