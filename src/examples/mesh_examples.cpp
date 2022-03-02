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

    vec<double> ***triangle_vertexes = new vec<double>**[]{triangle_1, triangle_2,  triangle_3,
                                                             triangle_4,  triangle_5,  triangle_6,
                                                             triangle_7,  triangle_8,  triangle_9,
                                                             triangle_10, triangle_11, triangle_12};

    Mesh<double> *mesh = new Mesh<double>(num_points, num_triangles, cell_size, points, triangle_vertexes);

    return mesh;
}

Mesh<double> *load_global_mesh(double mesh_dim, int elements_per_dim)
{
    const uint64_t cell_size = 8; // Cube

    
    const double mesh_x_dim = mesh_dim; 
    const double mesh_y_dim = mesh_dim;   
    const double mesh_z_dim = mesh_dim;

    const double mesh_elements_per_x_dim = elements_per_dim; // Number of cubes for each dimension
    const double mesh_elements_per_y_dim = elements_per_dim;
    const double mesh_elements_per_z_dim = elements_per_dim;

    const int z_points = mesh_elements_per_z_dim + 1;
    const int y_points = mesh_elements_per_y_dim + 1;
    const int x_points = mesh_elements_per_x_dim + 1;

    const double element_x_dim = mesh_x_dim / mesh_elements_per_x_dim;
    const double element_y_dim = mesh_y_dim / mesh_elements_per_y_dim;
    const double element_z_dim = mesh_z_dim / mesh_elements_per_z_dim;
 
    const double num_points       = z_points * y_points * x_points;   
    const double num_cubes        = mesh_elements_per_x_dim * mesh_elements_per_y_dim * mesh_elements_per_z_dim;


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
    
    // // Print vectors of each vertex in the mesh
    // for (int p = 0; p < num_points; p++)
    // {
    //     if (p % x_points == 0)
    //         cout << endl << p << ". ";
    //     cout << print_vec(points[p]) << ", ";
    // }
    // cout << endl;

    // Create array of cubes
    vec<double> ***cubes = new vec<double>**[num_cubes];
    for (int z = 0; z < mesh_elements_per_z_dim; z++)
    {
        for (int y = 0; y < mesh_elements_per_y_dim; y++)
        {
            for (int x = 0; x < mesh_elements_per_x_dim; x++)
            {
                int index   = z * x_points * y_points + y * x_points + x;
                int cube_index    = z*mesh_elements_per_x_dim*mesh_elements_per_y_dim + y*mesh_elements_per_x_dim + x;

                cubes[cube_index] = new vec<double>*[cell_size]{&(points[index]), &(points[index+1]), &(points[index+x_points]), &(points[index+x_points+1]), 
                                                        &(points[index+x_points*y_points]), &(points[index+x_points*y_points+1]), &(points[index+x_points*y_points+x_points]), &(points[index+x_points*y_points+x_points+1])};
            }
        }
    }

    // cout << endl << endl;

    // // Print 8 vertex for each cube in the global mesh
    // for (int c = 0; c < num_cubes; c++)
    // {
    //     cout << c << ". ";

    //     for (int v = 0; v < 7; v++)
    //         cout << cubes[c][v] - points << ", ";
    //     cout << cubes[c][7] - points;

    //     cout << endl;
    // }
    // cout << endl;

    // cout << endl << endl;

    
    Mesh<double> *mesh = new Mesh<double>(num_points, num_cubes, cell_size, points, cubes);

    return mesh;
}