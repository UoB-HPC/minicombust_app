#include "examples/mesh_examples.hpp"

#include <cstdint>
#include <string>  
#include <memory.h>

using namespace minicombust::geometry;
using namespace minicombust::utils;

using namespace std;


inline void fill_neighbours( uint64_t c, vec<uint64_t> local_position, vec<uint64_t> local_dim, vec<uint64_t> block_position, vec<uint64_t> block_dim, uint64_t **cell_neighbours, uint64_t **flow_block_element_sizes, uint64_t *block_element_disp, uint64_t shmem_disp  )
{
    const uint64_t faces_per_cell = 6; // Cube

    // Assume neighbour is also within current box
    uint64_t front_index = c - local_dim.x * local_dim.y;
    uint64_t back_index  = c + local_dim.x * local_dim.y;
    uint64_t left_index  = c - 1;
    uint64_t right_index = c + 1;
    uint64_t down_index  = c - local_dim.x;
    uint64_t up_index    = c + local_dim.x;

    // Check to see if neigbour is in neighbour block OR outside the mesh. 
    if ( local_position.z == 0  ) // Front
    {
        if (block_position.z == 0)  front_index = MESH_BOUNDARY;
        else 
        {
            block_position.z -= 1;
            uint64_t x  = local_position.x;
            uint64_t y  = local_position.y; 
            uint64_t z  = flow_block_element_sizes[2][block_position.z] - 1;
            front_index = block_element_disp[get_block_id(block_position, block_dim)] + z * flow_block_element_sizes[0][block_position.x] * flow_block_element_sizes[1][block_position.y] + y * flow_block_element_sizes[0][block_position.x] + x;
            block_position.z += 1;
        }
    }

    if ( local_position.z == local_dim.z-1  ) // Back
    {
        if (block_position.z == block_dim.z-1)  back_index = MESH_BOUNDARY;
        else 
        {
            block_position.z += 1;
            uint64_t x  = local_position.x;
            uint64_t y  = local_position.y; 
            uint64_t z  = 0;
            back_index  = block_element_disp[get_block_id(block_position, block_dim)] + z * flow_block_element_sizes[0][block_position.x] * flow_block_element_sizes[1][block_position.y]+ y * flow_block_element_sizes[0][block_position.x] + x;
            block_position.z -= 1;
        }
    }

    if ( local_position.x == 0  ) // Left
    {
        if (block_position.x == 0)  left_index = MESH_BOUNDARY;
        else 
        {
            block_position.x -= 1;
            uint64_t x  = flow_block_element_sizes[0][block_position.x] - 1;
            uint64_t y  = local_position.y; 
            uint64_t z  = local_position.z;
            left_index  = block_element_disp[get_block_id(block_position, block_dim)] + z * flow_block_element_sizes[0][block_position.x] * flow_block_element_sizes[1][block_position.y]+ y * flow_block_element_sizes[0][block_position.x] + x;
            block_position.x += 1;
        }
    }

    if ( local_position.x == local_dim.x-1  ) // Right
    {
        if (block_position.x == block_dim.x-1)  right_index = MESH_BOUNDARY;
        else 
        {
            block_position.x += 1;
            uint64_t x  = 0;
            uint64_t y  = local_position.y; 
            uint64_t z  = local_position.z;
            right_index = block_element_disp[get_block_id(block_position, block_dim)] + z * flow_block_element_sizes[0][block_position.x] * flow_block_element_sizes[1][block_position.y]+ y * flow_block_element_sizes[0][block_position.x] + x;
            block_position.x -= 1;
        }
    }

    if ( local_position.y == 0  ) // Down
    {
        if (block_position.y == 0)  down_index = MESH_BOUNDARY;
        else 
        {
            block_position.y -= 1;
            uint64_t x = local_position.x;
            uint64_t y = flow_block_element_sizes[1][block_position.y] - 1; 
            uint64_t z = local_position.z;
            down_index = block_element_disp[get_block_id(block_position, block_dim)] + z * flow_block_element_sizes[0][block_position.x] * flow_block_element_sizes[1][block_position.y]+ y * flow_block_element_sizes[0][block_position.x] + x;          
            block_position.y += 1;
        }
    }

    if ( local_position.y == local_dim.y-1  ) // Up
    {
        if (block_position.y == block_dim.y-1)  up_index = MESH_BOUNDARY;
        else 
        {
            block_position.y += 1;
            uint64_t x = local_position.x;
            uint64_t y = 0; 
            uint64_t z = local_position.z;
            up_index   = block_element_disp[get_block_id(block_position, block_dim)] + z * flow_block_element_sizes[0][block_position.x] * flow_block_element_sizes[1][block_position.y]+ y * flow_block_element_sizes[0][block_position.x] + x;
            block_position.y -= 1;
        }
    }

    // if (c == 0) cout << "{ F" << front_index << " B" << back_index << " L" << left_index << " R" << right_index << " D" << down_index << " U" << up_index << "} " ; 

    (*cell_neighbours)[(c - shmem_disp) * faces_per_cell + FRONT_FACE] = front_index ;
    (*cell_neighbours)[(c - shmem_disp) * faces_per_cell + BACK_FACE]  = back_index  ;
    (*cell_neighbours)[(c - shmem_disp) * faces_per_cell + LEFT_FACE]  = left_index  ;
    (*cell_neighbours)[(c - shmem_disp) * faces_per_cell + RIGHT_FACE] = right_index ;
    (*cell_neighbours)[(c - shmem_disp) * faces_per_cell + DOWN_FACE]  = down_index  ;
    (*cell_neighbours)[(c - shmem_disp) * faces_per_cell + UP_FACE]    = up_index    ;

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

    // Work out dimensions for local 
    int *prime_factors = (int *)malloc(ceil(log2(flow_ranks)) * sizeof(int));
    int nfactors       = get_prime_factors(flow_ranks, prime_factors);
    // printf("Ranks %d  PFranks %d flow ranks %d\n", mpi_config->rank, mpi_config->particle_flow_rank, flow_ranks );

    // Work out the number of rank blocks per dimension.
    vec<uint64_t> flow_elements_per_dim;
    vec<uint64_t> block_dim = {1, 1, 1};
    for ( int f = nfactors - 1; f >= 0; f-- )
    {
        // if (mpi_config->rank == 0) printf("factor %d\n", prime_factors[f]);
        flow_elements_per_dim = elements_per_dim / block_dim;
        int max_component = 0;
        for ( int i = 1; i < 3; i++ )
        {
            if ( flow_elements_per_dim[max_component] <= flow_elements_per_dim[i] )
                max_component = i;
        }

        block_dim[max_component]             = block_dim[max_component]             * prime_factors[f];
        flow_elements_per_dim[max_component] = flow_elements_per_dim[max_component] / prime_factors[f];
    }
    

    // Bound block dims at element widths
    for ( int i = 0; i < 3; i++ )
    {
        block_dim.x = min(block_dim.x, elements_per_dim.x);
        block_dim.y = min(block_dim.y, elements_per_dim.y);
        block_dim.z = min(block_dim.z, elements_per_dim.z);
    }

    uint64_t num_blocks = block_dim.x * block_dim.y * block_dim.z;
    

    // Calculate sizes and displacement for blocks. 
    double   *flow_block_displacements[3]; 
    uint64_t *flow_block_element_sizes[3];
    char dim_chars[3] = {'x', 'y', 'z'};
    for ( int i = 0; i < 3; i++ )
    {
        flow_block_displacements[i] = (double *)   malloc((block_dim[i]+1) * sizeof(double));
        flow_block_element_sizes[i] = (uint64_t *) malloc(block_dim[i]     * sizeof(uint64_t));

        double total_displacement = 0.0;
        for (uint64_t b = 0; b < block_dim[i]; b++)
        {
            uint64_t block_elements  = elements_per_dim[i] / block_dim[i]; 
            uint64_t remainder       = elements_per_dim[i] % block_dim[i]; 
            if ( b < remainder ) block_elements++;

            flow_block_displacements[i][b] = total_displacement;
            flow_block_element_sizes[i][b] = block_elements;

            total_displacement            += block_elements * element_dim[i];
        }
        flow_block_displacements[i][block_dim[i]] = total_displacement;
    }

    // const uint64_t num_points = (points_per_dim.z + block_dim.z - 1) * (points_per_dim.y + block_dim.y - 1) * (points_per_dim.x + block_dim.x - 1);   
    const uint64_t num_points = points_per_dim.z * points_per_dim.y * points_per_dim.x;   
    
        uint64_t *block_element_disp = (uint64_t *) malloc( (flow_ranks+1) * sizeof(uint64_t) ); 
    uint64_t *block_point_disps  = (uint64_t *) malloc( (flow_ranks+1) * sizeof(uint64_t) ); 
    uint64_t cell_displacement  = 0; 
    uint64_t point_displacement = 0; 
    block_element_disp[0] = cell_displacement;
    block_point_disps[0]   = point_displacement;
    for (uint64_t bz = 0; bz < block_dim[2]; bz++) // Iterate along z blocks
    {
        for (uint64_t by = 0; by < block_dim[1]; by++) // Iterate along y blocks
        {
            for (uint64_t bx = 0; bx < block_dim[0]; bx++) // Iterate along x blocks
            {
                // Increment elements by element block size, ready for next block
                vec<uint64_t> block_position = { bx, by, bz};
                cell_displacement  += flow_block_element_sizes[0][bx] * flow_block_element_sizes[1][by] * flow_block_element_sizes[2][bz];
                point_displacement += (flow_block_element_sizes[0][bx] + 1) * (flow_block_element_sizes[1][by] + 1) * (flow_block_element_sizes[2][bz] + 1);
                block_element_disp[get_block_id(block_position, block_dim) + 1] = cell_displacement;
                block_point_disps[get_block_id(block_position, block_dim) + 1]  = point_displacement;
            }
        }
    }

    if ( mpi_config->rank == 0 )
    {
        printf("\nMesh dimensions\n");
        cout << "\tReal dimensions (m)   : " << print_vec(mesh_dim)         << endl;
        cout << "\tTotal cells           : " << num_cubes                   << endl;
        cout << "\tTotal points          : " << num_points                  << endl;
        cout << "\tElement dimensions    : " << print_vec(elements_per_dim) << endl;
        cout << "\tFlow block dimensions : " << print_vec(block_dim)        << endl;
        cout << "\tFlow blocks           : " << num_blocks                  << endl;

        vec<uint64_t> average_block_size = { 0, 0, 0 };
        for (int i = 0; i < 3; i++) 
        {   
            cout << "\tBlock displacement " << dim_chars[i] << "  : ";
            for (uint64_t b = 0; b < block_dim[i] + 1; b++)
                cout << flow_block_displacements[i][b] << " ";

            for (uint64_t b = 0; b < block_dim[i]; b++)
                average_block_size[i] = average_block_size[i] + flow_block_element_sizes[i][b];

            average_block_size[i] = average_block_size[i] / block_dim[i];
            cout << endl;
        }

        printf( "\tAvg Flow dimensons    : %lu %lu %lu (%lu cells)\n", average_block_size.x,  average_block_size.y,  average_block_size.z,
                                                                       average_block_size.x * average_block_size.y * average_block_size.z );
        cout << "\tIdle flow ranks       : " << flow_ranks - num_blocks << endl;
    }


    MPI_Comm_split_type ( mpi_config->world, MPI_COMM_TYPE_SHARED, mpi_config->rank, MPI_INFO_NULL, &mpi_config->node_world );
    MPI_Comm_rank (mpi_config->node_world,  &mpi_config->node_rank);
    MPI_Comm_size (mpi_config->node_world,  &mpi_config->node_world_size);

    uint64_t *shmem_cell_disps      = (uint64_t *)    malloc((mpi_config->node_world_size + 1) * sizeof(uint64_t));
    uint64_t  shmem_cell_size       = num_cubes  / mpi_config->node_world_size;
    int       shmem_cell_remainder  = num_cubes % mpi_config->node_world_size;
    uint64_t  shmem_cell_disp       = 0;

    uint64_t *shmem_point_disps     = (uint64_t *) malloc((mpi_config->node_world_size + 1) * sizeof(uint64_t));
    uint64_t  shmem_point_size      = num_points / mpi_config->node_world_size;
    int       shmem_point_remainder = num_points % mpi_config->node_world_size;
    uint64_t  shmem_point_disp      = 0;

    for ( int r = 0; r < mpi_config->node_world_size + 1; r++ )
    {
        shmem_cell_disps[r]   = shmem_cell_disp;
        shmem_cell_disp      += shmem_cell_size  + ( r < shmem_cell_remainder );


        shmem_point_disps[r]  = shmem_point_disp;
        shmem_point_disp     += shmem_point_size + ( r < shmem_point_remainder );
    }
    // printf("Rank %d, node rank %d node size %d cell disp %lu point_disp %lu\n", mpi_config->rank, mpi_config->node_rank, mpi_config->node_world_size, shmem_cell_disps[mpi_config->node_rank], shmem_point_disps[mpi_config->node_rank]);

    uint64_t *shmem_cells;
    uint64_t *shmem_cell_neighbours;
    MPI_Aint shmem_cell_winsize             = (shmem_cell_disps[mpi_config->node_rank+1] - shmem_cell_disps[mpi_config->node_rank]) * cell_size      * sizeof(uint64_t);
    MPI_Aint shmem_cell_neighbours_winsize  = (shmem_cell_disps[mpi_config->node_rank+1] - shmem_cell_disps[mpi_config->node_rank]) * faces_per_cell * sizeof(uint64_t);

    MPI_Win_allocate_shared ( shmem_cell_winsize,            sizeof(uint64_t), MPI_INFO_NULL, mpi_config->node_world, &shmem_cells,           &mpi_config->win_cells );
    MPI_Win_allocate_shared ( shmem_cell_neighbours_winsize, sizeof(uint64_t), MPI_INFO_NULL, mpi_config->node_world, &shmem_cell_neighbours, &mpi_config->win_cell_neighbours );

    vec<double> *shmem_points;
    uint8_t     *shmem_cells_per_point;
    MPI_Aint shmem_points_winsize           = (shmem_point_disps[mpi_config->node_rank+1] - shmem_point_disps[mpi_config->node_rank]) * sizeof(vec<double>);
    MPI_Aint shmem_cells_per_points_winsize = (shmem_point_disps[mpi_config->node_rank+1] - shmem_point_disps[mpi_config->node_rank]) * sizeof(uint8_t);

    MPI_Win_allocate_shared ( shmem_points_winsize,           sizeof(vec<double>), MPI_INFO_NULL, mpi_config->node_world, &shmem_points,          &mpi_config->win_points );
    MPI_Win_allocate_shared ( shmem_cells_per_points_winsize, sizeof(uint8_t),     MPI_INFO_NULL, mpi_config->node_world, &shmem_cells_per_point, &mpi_config->win_cells_per_point );


    #pragma ivdep
    for ( uint64_t index = shmem_point_disps[mpi_config->node_rank]; index < shmem_point_disps[mpi_config->node_rank+1]; index++ )
    {
        const uint64_t x = index % points_per_dim.x;
        const uint64_t y = (index / points_per_dim.x) % points_per_dim.y;
        const uint64_t z = index / (points_per_dim.x * points_per_dim.y);


        shmem_points[index - shmem_point_disps[mpi_config->node_rank]].x = (double)x * element_dim.x;
        shmem_points[index - shmem_point_disps[mpi_config->node_rank]].y = (double)y * element_dim.y;
        shmem_points[index - shmem_point_disps[mpi_config->node_rank]].z = (double)z * element_dim.z;

        const int first_dims = ((x == 0)) + ((y == 0)) + (z == 0);
        const int last_dims  = (x == points_per_dim.x - 1) + (y == points_per_dim.y - 1) + (z == points_per_dim.z - 1);
        shmem_cells_per_point[index - shmem_point_disps[mpi_config->node_rank]] = (uint8_t)pow(2, 3 - first_dims - last_dims);
        // if (index == 540968)
        //     printf("%lu %lu %lu cpp %lu\n", x, y, z, shmem_cells_per_point[index - shmem_point_disps[mpi_config->node_rank]]);
    }


    // Create array of cubes, BLOCK ORDER.
    vec<uint64_t> outer_pos = vec<uint64_t> { 0, 0, 0 }; 
    vec<uint64_t> inner_pos = vec<uint64_t> { 0, 0, 0 }; 
    for (uint64_t bz = 0; bz < block_dim[2]; bz++) // Iterate along z blocks
    {
        for (uint64_t by = 0; by < block_dim[1]; by++) // Iterate along y blocks
        {
            for (uint64_t bx = 0; bx < block_dim[0]; bx++) // Iterate along x blocks
            {
                vec<uint64_t> block_position = { bx, by, bz};
                uint64_t cube_index = 0;

                // Set inner block position to beginning of block
                inner_pos = vec<uint64_t> { outer_pos.x, outer_pos.y, outer_pos.z }; 
                
                for (uint64_t z = 0; z < flow_block_element_sizes[2][bz]; z++) // Iterate along z axis within block (bx, by, bz)
                {
                    // if (mpi_config->rank == 0)  cout << "z" << z << endl;

                    for (uint64_t y = 0; y < flow_block_element_sizes[1][by]; y++) // Iterate along y axis within block (bx, by, bz)
                    {
                        for (uint64_t x = 0; x < flow_block_element_sizes[0][bx]; x++) // Iterate along x axis within block (bx, by, bz)
                        {
                            vec<uint64_t> local_position = { x, y, z };
                            vec<uint64_t> local_dim      = { flow_block_element_sizes[0][bx], flow_block_element_sizes[1][by], flow_block_element_sizes[2][bz] };

                            cube_index      = block_element_disp[get_block_id(block_position, block_dim)] + z * local_dim.x       * local_dim.y + y       * local_dim.x + x;

                            bool write_cell  = ( (cube_index  >= shmem_cell_disps[mpi_config->node_rank])  && (cube_index  < shmem_cell_disps[mpi_config->node_rank+1])  );

                            if (write_cell)
                            {
                                const uint64_t global_point_id = inner_pos.x + inner_pos.y * points_per_dim.x + inner_pos.z * points_per_dim.x * points_per_dim.y;

                                // printf("Rank %d cube %lu index %lu\n", mpi_config->rank, cube_index, cube_index - shmem_cell_disps[mpi_config->node_rank]);
                                fill_neighbours(cube_index, local_position, local_dim, block_position, block_dim, &shmem_cell_neighbours, flow_block_element_sizes, block_element_disp, shmem_cell_disps[mpi_config->node_rank]);
                                cube_index -= shmem_cell_disps[mpi_config->node_rank];
                                
                                shmem_cells[cube_index*cell_size + A_VERTEX] = global_point_id;
                                shmem_cells[cube_index*cell_size + H_VERTEX] = global_point_id + points_per_dim.x * points_per_dim.y + points_per_dim.x + 1;
                                shmem_cells[cube_index*cell_size + B_VERTEX] = global_point_id + 1;
                                shmem_cells[cube_index*cell_size + C_VERTEX] = global_point_id + points_per_dim.x ; 
                                shmem_cells[cube_index*cell_size + D_VERTEX] = global_point_id + points_per_dim.x + 1;
                                shmem_cells[cube_index*cell_size + E_VERTEX] = global_point_id + points_per_dim.x * points_per_dim.y; 
                                shmem_cells[cube_index*cell_size + F_VERTEX] = global_point_id + points_per_dim.x * points_per_dim.y + 1;
                                shmem_cells[cube_index*cell_size + G_VERTEX] = global_point_id + points_per_dim.x * points_per_dim.y + points_per_dim.x ; 
                                // printf("Rank %d cube %lu index end %lu\n", mpi_config->rank, cube_index + shmem_cell_disps[mpi_config->node_rank], cube_index );

                                // for (int i = 0; i < cell_size; i++)
                                //     if ( shmem_cells[(cube_index)*cell_size + i] == 540968 )  printf("Cube index %lu has node %lu \n", cube_index + shmem_cell_disps[mpi_config->node_rank], 540968 );
                            }

                            inner_pos.x++; 

                        } // Inner block x

                        inner_pos.x  = outer_pos.x;
                        inner_pos.y++;

                    } // Inner block y

                    inner_pos.z++;
                    inner_pos.x = outer_pos.x;
                    inner_pos.y = outer_pos.y;

                } // Inner block z

                outer_pos.x  +=  flow_block_element_sizes[0][bx];

            } // Outer block x

            outer_pos.x  = 0;
            outer_pos.y += flow_block_element_sizes[1][by];

        } // Outer block y

        outer_pos.x  = 0;
        outer_pos.y  = 0;
        outer_pos.z += flow_block_element_sizes[2][bz];

    } // Outer block z

    MPI_Barrier(mpi_config->world);

    // Create array of faces, each face is a pointer to two neighbouring cells.
    uint64_t faces_size = 0;
    Face<uint64_t> *faces      = nullptr;
    uint64_t       *cell_faces = nullptr;
    if ( mpi_config->solver_type == FLOW )
    {
        vec<uint64_t> local_flow_dim = vec<uint64_t> { flow_block_element_sizes[0][mpi_config->particle_flow_rank % block_dim.x], 
                                                       flow_block_element_sizes[1][(mpi_config->particle_flow_rank / block_dim.x) % block_dim.y],
                                                       flow_block_element_sizes[2][mpi_config->particle_flow_rank / (block_dim.x * block_dim.y)] };
        
        uint64_t face_count = 0;
        const uint64_t cell_shmem_disp = shmem_cell_disps[mpi_config->node_rank];
        const uint64_t cell_block_disp = block_element_disp[mpi_config->particle_flow_rank];
        const uint64_t local_flow_size = local_flow_dim.x * local_flow_dim.y * local_flow_dim.z;

        faces_size = (local_flow_dim.x + 1)*local_flow_dim.y*local_flow_dim.z + (local_flow_dim.y + 1)*local_flow_dim.x*local_flow_dim.z + (local_flow_dim.z + 1)*local_flow_dim.x*local_flow_dim.y;
        faces      = (Face<uint64_t> *)    malloc(faces_size * sizeof(Face<uint64_t>));                        // Faces  = {{0, BOUNDARY}, {0, BOUNDARY}, {0, BOUNDARY}, {0, 1}, ...}; 
        cell_faces = (uint64_t *)malloc(local_flow_size * faces_per_cell * sizeof(uint64_t));  // Cfaces = {f0, f1, f2, f3, f4, f5, f1, f2, f4, f5}; 
        // printf("faces size %lu cell_faces size %lu disp %lu \n", faces_size, local_flow_size * faces_per_cell, cell_shmem_disp);

        int face_indexes [faces_per_cell * local_flow_size];
        for (uint64_t i = 0; i < faces_per_cell * local_flow_size; i++)  face_indexes[i] = -1;
        for ( uint64_t cell = block_element_disp[mpi_config->particle_flow_rank]; cell < block_element_disp[mpi_config->particle_flow_rank+1]; cell++ )
        {
            for ( uint64_t f_id = 0; f_id < faces_per_cell; f_id++ )
            {
                // printf("Cell %lu face %lu: Checking if already created\n", cell, f_id);
                const uint64_t face_neighbour_cell = shmem_cell_neighbours[(int)((cell - cell_shmem_disp) * faces_per_cell + f_id)];
                
                if ( face_indexes[ (cell - cell_block_disp) * faces_per_cell + f_id ] == -1 )
                {  
                    faces [face_count] = Face<uint64_t> ( cell, face_neighbour_cell );

                    if ( face_neighbour_cell >= block_element_disp[mpi_config->particle_flow_rank] && face_neighbour_cell < block_element_disp[mpi_config->particle_flow_rank+1] )
                            face_indexes[ (face_neighbour_cell - cell_block_disp) * faces_per_cell + (f_id ^ 1) ] = face_count;

                    // printf("Cell %lu face %lu: face_id %d new\n", cell, f_id, face_count);
                    cell_faces [(cell - cell_block_disp) * faces_per_cell + f_id] = face_count;

                    face_count++;
                }
                else
                {
                    // printf("Cell %lu face %lu: face_id %lu\n", cell, f_id, face_indexes[ (cell - cell_block_disp) * faces_per_cell + f_id ]);
                    cell_faces [(cell - cell_block_disp) * faces_per_cell + f_id] = face_indexes[ (cell - cell_block_disp) * faces_per_cell + f_id ];
                }
            }
        }
    }
    // printf("Rank %d done\n", mpi_config->rank);

    MPI_Barrier(mpi_config->world);

    Mesh<double> *mesh = new Mesh<double>(mpi_config, num_points, num_cubes, cell_size, faces_size, faces_per_cell, shmem_points, shmem_cells, faces, cell_faces, shmem_cell_neighbours, shmem_cells_per_point, num_blocks, shmem_cell_disps, shmem_point_disps, block_element_disp, block_dim);

    return mesh;
}
