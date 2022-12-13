#include "examples/mesh_examples.hpp"

#include <cstdint>
#include <string>  
#include <memory.h>

using namespace minicombust::geometry;
using namespace minicombust::utils;

using namespace std;


void fill_neighbours( uint64_t c, vec<uint64_t> local_position, vec<uint64_t> local_dim, vec<uint64_t> block_position, vec<uint64_t> block_dim, uint64_t **cell_neighbours )
{
    const uint64_t faces_per_cell = 6; // Cube
    
    uint64_t front_index = c - local_dim.x * local_dim.y;
    uint64_t back_index  = c + local_dim.x * local_dim.y;
    uint64_t left_index  = c - 1;
    uint64_t right_index = c + 1;
    uint64_t down_index  = c - local_dim.x;
    uint64_t up_index    = c + local_dim.x;

    // cout << "Bdim " << print_vec(block_dim) << "Ldim " << print_vec(local_dim)  << endl;

    // if (local_position.x -  > block_dim.x)
    // printf("SANITY0\n");

    (*cell_neighbours)[c * faces_per_cell + FRONT_FACE] = (local_position.z == 0             && block_position.z == 0)             ? MESH_BOUNDARY : front_index ;  
    // printf("SANITY1\n");
    (*cell_neighbours)[c * faces_per_cell + BACK_FACE]  = (local_position.z == local_dim.z-1 && block_position.z == block_dim.z-1) ? MESH_BOUNDARY : back_index  ;  
    // printf("SANITY2\n");
    (*cell_neighbours)[c * faces_per_cell + LEFT_FACE]  = (local_position.x == 0             && block_position.x == 0)             ? MESH_BOUNDARY : left_index  ;  
    // printf("SANITY3\n");
    (*cell_neighbours)[c * faces_per_cell + RIGHT_FACE] = (local_position.x == local_dim.x-1 && block_position.x == block_dim.x-1) ? MESH_BOUNDARY : right_index ;  
    // printf("SANITY4\n");
    (*cell_neighbours)[c * faces_per_cell + DOWN_FACE]  = (local_position.y == 0             && block_position.y == 0)             ? MESH_BOUNDARY : down_index  ;  
    // printf("SANITY5\n");
    (*cell_neighbours)[c * faces_per_cell + UP_FACE]    = (local_position.y == local_dim.y-1 && block_position.y == block_dim.y-1) ? MESH_BOUNDARY : up_index    ;  

    // // FRONT
    // if ( c < elements_per_dim.x*elements_per_dim.y )  {
    //     // faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
    //     cell_neighbours[c*faces_per_cell + FRONT_FACE] = MESH_BOUNDARY;
    // }                             
    // else
    // {
    //     cell_neighbours[c*faces_per_cell + FRONT_FACE] = c - elements_per_dim.x*elements_per_dim.y;
    // }

    // // BACK
    // if ( c >= (elements_per_dim.z-1)*elements_per_dim.x*elements_per_dim.y )  {
    //     // faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
    //     cell_neighbours[c*faces_per_cell + BACK_FACE]  = MESH_BOUNDARY;
    // }                             
    // else
    // {
    //     // faces[faces_count++]                           = Face<double>(c, c + elements_per_dim.x*elements_per_dim.y);
    //     cell_neighbours[c*faces_per_cell + BACK_FACE]  = c + elements_per_dim.x*elements_per_dim.y;
    // }

    // // LEFT
    // if ( (c % elements_per_dim.x) == 0 )  {
    //     // faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
    //     cell_neighbours[c*faces_per_cell + LEFT_FACE]  = MESH_BOUNDARY;
    // }                             
    // else
    // {
    //     cell_neighbours[c*faces_per_cell + LEFT_FACE]  = c - 1;
    // }

    // // RIGHT
    // if ( ((c+1) % elements_per_dim.x) == 0 )  {
    //     // faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
    //     cell_neighbours[c*faces_per_cell + RIGHT_FACE] = MESH_BOUNDARY;
    // }                             
    // else
    // {
    //     // faces[faces_count++]                           = Face<double>(c, c + 1);
    //     cell_neighbours[c*faces_per_cell + RIGHT_FACE] = c + 1;
    // }

    // // DOWN
    // if ( (c % (elements_per_dim.x*elements_per_dim.y)) < elements_per_dim.x )  {
    //     // faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
    //     cell_neighbours[c*faces_per_cell + DOWN_FACE]  = MESH_BOUNDARY;
    // }                             
    // else
    // {
    //     cell_neighbours[c*faces_per_cell + DOWN_FACE]  = c - elements_per_dim.x;
    // }

    // // UP
    // if ( ((c+elements_per_dim.x) % (elements_per_dim.x*elements_per_dim.y)) < elements_per_dim.x )  {
    //     // faces[faces_count++]                           = Face<double>(MESH_BOUNDARY, c);
    //     cell_neighbours[c*faces_per_cell + UP_FACE]    = MESH_BOUNDARY;
    // }                             
    // else
    // {
    //     // faces[faces_count++]                           = Face<double>(c, c + elements_per_dim.x);
    //     cell_neighbours[c*faces_per_cell + UP_FACE]    = c + elements_per_dim.x;
    // }
}

Mesh<double> *load_mesh(MPI_Config *mpi_config, vec<double> mesh_dim, vec<uint64_t> elements_per_dim, int flow_ranks)
{
    const uint64_t cell_size  = 8; // Cube
    const uint64_t faces_per_cell = 6; // Cube

    vec<uint64_t> points_per_dim = elements_per_dim + 1UL;
    vec<double>   element_dim   = mesh_dim / vec<double>{ (double)elements_per_dim.x, (double)elements_per_dim.y, (double)elements_per_dim.z };
 
    const uint64_t num_cubes        = elements_per_dim.x * elements_per_dim.y * elements_per_dim.z;

    if ( mpi_config->rank == 0 )
    {
        printf("\nMesh dimensions\n");
        cout << "\tReal dimensions (m)    " << print_vec(mesh_dim)         << endl;
        cout << "\tElement dimensions (m) " << print_vec(elements_per_dim) << endl;
    }

    // Work out dimensions for local 
    int *prime_factors = (int *)malloc(ceil(log2(flow_ranks)) * sizeof(int));
    int nfactors       = get_prime_factors(flow_ranks, prime_factors);

    // Work out the number of rank blocks per dimension.
    vec<uint64_t> flow_elements_per_dim;
    vec<uint64_t> block_dim = {1, 1, 1};
    for ( int f = nfactors - 1; f >= 0; f-- )
    {
        flow_elements_per_dim = elements_per_dim / block_dim;
        int max_component = 0;
        for ( int i = 1; i < 3; i++ )
        {
            if ( flow_elements_per_dim[i-1] < flow_elements_per_dim[i] )
                max_component = i;
        }

        block_dim[max_component]             = block_dim[max_component]             * prime_factors[f];
        flow_elements_per_dim[max_component] = flow_elements_per_dim[max_component] / prime_factors[f];
    }

    if (flow_elements_per_dim.x * flow_elements_per_dim.y * flow_elements_per_dim.z == 0)
        printf("Warning! Flow Rank %d has 0 size mesh\n", mpi_config->particle_flow_rank); 

    // Calculate sizes and displacement for blocks. 
    double   *flow_block_displacements[3]; 
    uint64_t *flow_block_element_sizes[3];
    for ( int i = 0; i < 3; i++ )
    {
        flow_block_displacements[i] = (double *)  malloc((block_dim[i]+1) * sizeof(double));
        flow_block_element_sizes[i] = (uint64_t *)malloc(block_dim[i]     * sizeof(uint64_t));
        if (mpi_config->rank == 0) cout << "Block displacement " << i << ": ";

        double total_displacement = 0.0;
        for (uint64_t b = 0; b < block_dim[i]; b++)
        {
            uint64_t block_elements  = elements_per_dim[i] / block_dim[i]; 
            uint64_t remainder       = elements_per_dim[i] % block_dim[i]; 
            if ( b < remainder ) block_elements++;

            flow_block_displacements[i][b] = total_displacement;
            flow_block_element_sizes[i][b] = block_elements;
            total_displacement            += block_elements * element_dim[i];
            if (mpi_config->rank == 0)  cout << flow_block_displacements[i][b] << " ";
        }
        flow_block_displacements[i][block_dim[i]] = total_displacement;
        if (mpi_config->rank == 0)  cout << flow_block_displacements[i][block_dim[i]]  << endl;
    }

    const uint64_t num_points = (points_per_dim.z + block_dim.z - 1) * (points_per_dim.y + block_dim.y - 1) * (points_per_dim.x + block_dim.x - 1);   
    if (mpi_config->rank == 0) cout << "Total points size " << num_points << endl;  


    // Create array of all the points we'll need. NORMAL LAYOUT.
    // vec<double> current_point = {0, 0, 0};
    // for (uint64_t z = 0; z < points_per_dim.z; z++)
    // {
    //     current_point.y = 0;
    //     for (uint64_t y = 0; y < points_per_dim.y; y++)
    //     {
    //         current_point.x = 0;
    //         for (uint64_t x = 0; x < points_per_dim.x; x++)
    //         {
    //             uint64_t index = z*points_per_dim.x*points_per_dim.y + y*points_per_dim.x + x;
    //             memcpy(points + index, &current_point, sizeof(vec<double>));
    //             current_point = current_point + vec<double>{element_dim.x, 0, 0};
    //         }
    //         current_point = current_point + vec<double>{0, element_dim.y, 0};
    //     }
    //     current_point = current_point + vec<double>{0, 0, element_dim.z};
    // }


    // Create array of cube cells NORMAL LAYOUT
    // for (uint64_t z = 0; z < elements_per_dim.z; z++)
    // {
    //     for (uint64_t y = 0; y < elements_per_dim.y; y++)
    //     {
    //         for (uint64_t x = 0; x < elements_per_dim.x; x++)
    //         {
    //             uint64_t index         = z * points_per_dim.x * points_per_dim.y + y * points_per_dim.x + x;
    //             uint64_t cube_index    = z*elements_per_dim.x*elements_per_dim.y + y*elements_per_dim.x + x;

    //             cubes[cube_index*cell_size + A_VERTEX] = index;
    //             cubes[cube_index*cell_size + B_VERTEX] = index + 1;
    //             cubes[cube_index*cell_size + C_VERTEX] = index + points_per_dim.x;
    //             cubes[cube_index*cell_size + D_VERTEX] = index + points_per_dim.x + 1;
    //             cubes[cube_index*cell_size + E_VERTEX] = index + points_per_dim.x*points_per_dim.y;
    //             cubes[cube_index*cell_size + F_VERTEX] = index + points_per_dim.x*points_per_dim.y + 1;
    //             cubes[cube_index*cell_size + G_VERTEX] = index + points_per_dim.x*points_per_dim.y + points_per_dim.x;
    //             cubes[cube_index*cell_size + H_VERTEX] = index + points_per_dim.x*points_per_dim.y + points_per_dim.x+1;
    //         }
    //     }
    // }

    // Create array of cube cells, and points
    vec<double> *points       = (vec<double> *)malloc(num_points * sizeof(vec<double>));
    uint64_t *cubes           = (uint64_t *)malloc(num_cubes * cell_size      * sizeof(uint64_t));
    uint64_t *cell_neighbours = (uint64_t *)malloc(num_cubes * faces_per_cell * sizeof(uint64_t)); 

    // Create array of cubes, BLOCK ORDER.
    uint64_t block_element_disp = 0; 
    uint64_t block_point_disp   = 0; 
    vec<double> block_real_disp = vec<double> { 0.0, 0.0, 0.0 }; 
    for (uint64_t bz = 0; bz < block_dim[2]; bz++) // Iterate along z blocks
    {
        for (uint64_t by = 0; by < block_dim[1]; by++) // Iterate along y blocks
        {
            for (uint64_t bx = 0; bx < block_dim[0]; bx++) // Iterate along x blocks
            {
                static int block_num = 0;
                uint64_t point_index= 0.0, cube_index = 0.0;

                // Set inner block position to beginning of block
                vec<double> block_inner_real_disp = vec<double> { block_real_disp.x, block_real_disp.y, block_real_disp.z }; 
                
                if (mpi_config->rank == 0) cout << "Block/Rank " << block_num++ << " size = " << flow_block_element_sizes[2][bz] * flow_block_element_sizes[1][by] * flow_block_element_sizes[0][bx] << endl;  

                for (uint64_t z = 0; z < flow_block_element_sizes[2][bz]; z++) // Iterate along z axis within block (bx, by, bz)
                {
                    if (mpi_config->rank == 0)  cout << "z" << z << endl;

                    for (uint64_t y = 0; y < flow_block_element_sizes[1][by]; y++) // Iterate along y axis within block (bx, by, bz)
                    {
                        for (uint64_t x = 0; x < flow_block_element_sizes[0][bx]; x++) // Iterate along x axis within block (bx, by, bz)
                        {
                            // Set indexes
                            // index         = z * points_per_dim.x * points_per_dim.y + y * points_per_dim.x + x;
                            cube_index    = block_element_disp + z *  flow_block_element_sizes[0][bx]      *  flow_block_element_sizes[1][by]      + y *  flow_block_element_sizes[0][bx]      + x;
                            point_index   = block_point_disp   + z * (flow_block_element_sizes[0][bx] + 1) * (flow_block_element_sizes[1][by] + 1) + y * (flow_block_element_sizes[0][bx] + 1) + x;

                            // Create cube
                            cubes[cube_index*cell_size + A_VERTEX] = point_index;
                            cubes[cube_index*cell_size + B_VERTEX] = point_index + 1;
                            cubes[cube_index*cell_size + C_VERTEX] = point_index + (flow_block_element_sizes[0][bx] + 1);
                            cubes[cube_index*cell_size + D_VERTEX] = point_index + (flow_block_element_sizes[0][bx] + 1) + 1;
                            cubes[cube_index*cell_size + E_VERTEX] = point_index + (flow_block_element_sizes[0][bx] + 1)*(flow_block_element_sizes[1][by] + 1);
                            cubes[cube_index*cell_size + F_VERTEX] = point_index + (flow_block_element_sizes[0][bx] + 1)*(flow_block_element_sizes[1][by] + 1) + 1;
                            cubes[cube_index*cell_size + G_VERTEX] = point_index + (flow_block_element_sizes[0][bx] + 1)*(flow_block_element_sizes[1][by] + 1) + (flow_block_element_sizes[0][bx] + 1);
                            cubes[cube_index*cell_size + H_VERTEX] = point_index + (flow_block_element_sizes[0][bx] + 1)*(flow_block_element_sizes[1][by] + 1) + (flow_block_element_sizes[0][bx] + 1)+1;

                            vec<uint64_t> local_position = { x, y, z };
                            vec<uint64_t> local_dim      = { flow_block_element_sizes[0][bx], flow_block_element_sizes[1][by], flow_block_element_sizes[2][bz] };
                            vec<uint64_t> block_position = { bx, by, bz};
                            if (mpi_config->rank == 0)  cout << cube_index << " ";
                            MPI_Barrier(mpi_config->world);
                            fill_neighbours(cube_index, local_position, local_dim, block_position, block_dim, &cell_neighbours);

                            // Create point
                            points[point_index] = vec<double> { block_inner_real_disp.x, block_inner_real_disp.y, block_inner_real_disp.z };

                            // Increment inner x displacement
                            block_inner_real_disp.x += element_dim.x;

                            // if (mpi_config->rank == 0)  cout << point_index << " ";
                            // if (mpi_config->rank == 0)  cout << print_vec(points[point_index]) << "     ";
                        }

                        // Add last x point.
                        points[++point_index] = vec<double> { block_inner_real_disp.x, block_inner_real_disp.y, block_inner_real_disp.z };
                        
                        // Increment inner y displacement, reset inner x displacement to block's
                        block_inner_real_disp.x  = block_real_disp.x;
                        block_inner_real_disp.y += element_dim.y;
                        
                        if (mpi_config->rank == 0)  cout << endl;
                        // if (mpi_config->rank == 0)  cout << print_vec(points[point_index]) << endl;
                        // if (mpi_config->rank == 0)  cout << point_index << endl;
                    }

                    // Add last y points.
                    for ( uint64_t x = 0; x < flow_block_element_sizes[0][bx] + 1; x++ )
                    {
                        points[++point_index] = vec<double> { block_inner_real_disp.x, block_inner_real_disp.y, block_inner_real_disp.z };
                        block_inner_real_disp.x += element_dim.x;

                        // if (mpi_config->rank == 0)  cout << point_index << " ";
                        // if (mpi_config->rank == 0)  cout << print_vec(points[point_index]) << "     ";
                    }                    
                    // Increment inner z displacement, reset inner x/y displacements to block's
                    block_inner_real_disp.z += element_dim.z;
                    block_inner_real_disp.x  = block_real_disp.x;
                    block_inner_real_disp.y  = block_real_disp.y;
                    
                    if (mpi_config->rank == 0)  cout << endl;
                }

                // Add last z points
                if (mpi_config->rank == 0)  cout << "z" << flow_block_element_sizes[2][bz] << endl;
                for ( uint64_t y = 0; y < flow_block_element_sizes[1][by] + 1; y++ )
                {
                    for ( uint64_t x = 0; x < flow_block_element_sizes[0][bx] + 1; x++ )
                    {
                        points[++point_index] = vec<double> { block_inner_real_disp.x, block_inner_real_disp.y, block_inner_real_disp.z };
                        block_inner_real_disp.x += element_dim.x;

                        // if (mpi_config->rank == 0)  cout << point_index << " ";
                        // if (mpi_config->rank == 0)  cout << print_vec(points[point_index]) << "     ";
                    }

                    block_inner_real_disp.x  = block_real_disp.x;
                    block_inner_real_disp.y += element_dim.y;

                    // if (mpi_config->rank == 0)  cout << endl;
                }

                // Increment elements by element block size, ready for next block
                block_element_disp +=  flow_block_element_sizes[0][bx]    *  flow_block_element_sizes[1][by]    *  flow_block_element_sizes[2][bz];
                // Increment points by point block size, ready for next block 
                block_point_disp   += (flow_block_element_sizes[0][bx]+1) * (flow_block_element_sizes[1][by]+1) * (flow_block_element_sizes[2][bz]+1);
                // Increment x block displacement by x dim of block. 
                block_real_disp.x  +=  flow_block_element_sizes[0][bx] * element_dim.x;

                // if (mpi_config->rank == 0)  cout << endl;
            }
            // Increment y block displacement by y dim of block.  Reset x block displacement.
            block_real_disp.x  = 0.0;
            block_real_disp.y += flow_block_element_sizes[0][by] * element_dim.y;
        }
        // Increment z block displacement by z dim of block.  Reset y block displacement.
        block_real_disp.y  = 0.0;
        block_real_disp.z += flow_block_element_sizes[0][bz] * element_dim.z;
    }
    
    // Create array of faces, each face is a pointer to two neighbouring cells.
    const uint64_t faces_size = elements_per_dim.x*elements_per_dim.y*points_per_dim.z + elements_per_dim.y*elements_per_dim.z*points_per_dim.x + elements_per_dim.z*elements_per_dim.x*points_per_dim.y;
    // Face<double>  *faces      = (Face<double> *)malloc(faces_size*sizeof(Face<double>));                      // Faces         = {{0, BOUNDARY}, {0, BOUNDARY}, {0, BOUNDARY}, {0, 1}, ...}; 
    Face<double>  *faces      = nullptr;                               

    Mesh<double> *mesh = new Mesh<double>(mpi_config, num_points, num_cubes, cell_size, faces_size, faces_per_cell, points, cubes, faces, cell_neighbours);

    return mesh;
}
