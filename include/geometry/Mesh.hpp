#pragma once

#include <map>

#include <memory.h>

#include "utils/utils.hpp"

using namespace minicombust::utils; 

namespace minicombust::geometry 
{   
    static const uint64_t MESH_BOUNDARY = UINT64_MAX;

    enum FACE_DIRECTIONS { NOFACE_ERROR = -1, FRONT_FACE = 0, BACK_FACE = 1, LEFT_FACE = 2, RIGHT_FACE = 3, DOWN_FACE = 4, UP_FACE = 5};
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

            MPI_Config *mpi_config;

            // Calculate the centre point in each cell
            // Computed as the average of all vertex positions
            void calculate_cell_centres(void) {

                for (uint64_t c = 0; c < shmem_mesh_size; ++c) 
                {
                    // printf("Rank %d calculating cell %lu local_points_disp %lu local_points_size %lu\n", mpi_config->rank, c, local_points_disp, local_points_size);
                    cell_centres[c] = vec<T>{0.0, 0.0, 0.0};
                    for (uint64_t i = 0; i < cell_size; ++i) 
                    {
                        // printf("Rank %d %lu point index %lu point index real %lu\n", mpi_config->rank, i, cells[c*cell_size + i] - local_points_disp, cells[c*cell_size + i] );
                        cell_centres[c] += points[cells[c*cell_size + i] - shmem_point_disp];
                    }
                    cell_centres[c] /= static_cast<T>(cell_size);
                }
                
            }
 
        public:
            const uint64_t points_size;         // Number of points in the mesh
            const uint64_t mesh_size;           // Number of polygons in the mesh
            const uint64_t cell_size;           // Number of points in the cell
            const uint64_t faces_size;          // Number of unique faces in the mesh
            const uint64_t faces_per_cell;      // Number of faces in a cell

            uint64_t shmem_mesh_size;    // Number of polygons in the mesh stored in shmem window
            uint64_t shmem_point_size;   // Number of points   in the mesh stored in shmem window

            uint64_t shmem_cell_disp;    // Number of polygons in the mesh that a flow rank owns.
            uint64_t shmem_point_disp;   // Number of points   in the mesh that a flow rank owns.

            uint64_t local_mesh_size;     // Number of polygons in the mesh that a flow rank owns.
            uint64_t local_cells_disp;    // Number of polygons in the mesh that a flow rank owns.
            uint64_t local_points_size;   // Number of points in the mesh that a flow rank owns.
            uint64_t local_points_disp;   // Number of points in the mesh that a flow rank owns.
            
            
            vec<T> cell_size_vector;      // Cell size
            vec<T> *points;               // Mesh points    = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, ...}:
            uint64_t *cells;              // Cells          = {{0, 1, 2, 300, 40, 36, 7, 2}, {1, 2, 4, 300}, ...};
            Face<T> *faces;               // Faces          = {{0, BOUNDARY}, {0, BOUNDARY}, {0, BOUNDARY}, {0, 1}, ...};  TODO: Not needed by particle ranks (25% mesh mem)
            vec<T> *cell_centres;         // Cell centres   = {{0.5, 3.0, 4.0}, {2.5, 3.0, 4.0}, ...};
            uint64_t *cell_neighbours;    // Cell faces     = {{0, 1, 2, 3, 4, 5}, {6, 1, 7, 3, 8, 5}}
            uint8_t *cells_per_point;     // Number of neighbouring cells for each point


            uint64_t     num_blocks;
            uint64_t     *block_element_disp;
            uint64_t     *block_point_disp;
            vec<uint64_t> flow_block_dim;
            
            uint64_t *shmem_cell_disps;
            uint64_t *shmem_point_disps;
 
            uint64_t *particles_per_point; // Number of particles in each cell

            // Particle source terms
            T      *evaporated_fuel_mass_rate;       // particle_rate_of_mass
            T      *particle_energy_rate;            // particle_energy
            vec<T> *particle_momentum_rate;          // particle_momentum_rate

            // Flow source terms
            const vec<T> dummy_gas_vel = {20., 0.1, 0.1};
            const T      dummy_gas_pre = 4000.;
            const T      dummy_gas_tem = 2000.;

            flow_aos<T> *flow_terms;
            flow_aos<T> *flow_grad_terms;

            size_t points_array_size               = 0;
            size_t cells_array_size                = 0;

            size_t cell_centre_size                = 0;
            size_t faces_array_size                = 0;
            size_t particles_per_point_size        = 0;

            size_t cell_neighbours_array_size      = 0;
            size_t cells_per_point_size            = 0;

            size_t evaporated_fuel_mass_rate_size  = 0;
            size_t particle_energy_size            = 0;
            size_t particle_momentum_rate_size     = 0;

            size_t block_disp_size                 = 0;

            size_t flow_term_size                  = 0;

            Mesh(MPI_Config *mpi_config, uint64_t points_size, uint64_t mesh_size, uint64_t cell_size, uint64_t faces_size, uint64_t faces_per_cell, vec<T> *points, uint64_t *cells, Face<T> *faces, uint64_t *cell_neighbours, uint8_t *cells_per_point, uint64_t num_blocks, uint64_t *shmem_cell_disps, uint64_t *shmem_point_disps, uint64_t *block_element_disp, uint64_t *block_point_disp, vec<uint64_t> flow_block_dim) 
            : mpi_config(mpi_config), points_size(points_size), mesh_size(mesh_size), cell_size(cell_size), faces_size(faces_size), faces_per_cell(faces_per_cell), points(points), cells(cells), faces(faces), cell_neighbours(cell_neighbours), cells_per_point(cells_per_point), num_blocks(num_blocks), shmem_cell_disps(shmem_cell_disps), shmem_point_disps(shmem_point_disps), block_element_disp(block_element_disp), block_point_disp(block_point_disp), flow_block_dim(flow_block_dim)
            {
                
                shmem_cell_disp   = shmem_cell_disps[mpi_config->node_rank];
                shmem_point_disp  = shmem_point_disps[mpi_config->node_rank];
                local_cells_disp  = ( mpi_config->solver_type == FLOW ) ? block_element_disp[mpi_config->particle_flow_rank]  : 0;
                local_points_disp = ( mpi_config->solver_type == FLOW ) ? block_point_disp[mpi_config->particle_flow_rank]    : 0;


                shmem_mesh_size   = shmem_cell_disps[mpi_config->node_rank + 1]  - shmem_cell_disp;
                shmem_point_size  = shmem_point_disps[mpi_config->node_rank + 1] - shmem_point_disp;
                local_mesh_size   = ( mpi_config->solver_type == FLOW ) ? block_element_disp[mpi_config->particle_flow_rank + 1] - local_cells_disp  : mesh_size;
                local_points_size = ( mpi_config->solver_type == FLOW ) ? block_point_disp[mpi_config->particle_flow_rank + 1]   - local_points_disp : points_size;

                // Allocate space for and calculate cell centre co-ordinates
                points_array_size     = shmem_point_size * sizeof(vec<double>);
                cells_per_point_size  = shmem_point_size * sizeof(uint8_t);
                // particles_per_point_size        = local_points_size * sizeof(uint64_t);

                cells_array_size                = shmem_mesh_size * cell_size * sizeof(uint64_t);
                cell_centre_size                = shmem_mesh_size * sizeof(vec<T>);

                block_disp_size                 = num_blocks * sizeof(uint64_t);
                 
                if (mpi_config->solver_type == FLOW || mpi_config->world_size == 1)
                {
                    faces_array_size                = faces_size * sizeof(Face<T>);
                    
                    evaporated_fuel_mass_rate_size  = local_mesh_size * sizeof(T);
                    particle_energy_size            = local_mesh_size * sizeof(T);
                    particle_momentum_rate_size     = local_mesh_size * sizeof(vec<T>);
                    
                    flow_term_size                  = local_mesh_size * sizeof(flow_aos<T>);

                    evaporated_fuel_mass_rate                    = (T *)          malloc(evaporated_fuel_mass_rate_size);
                    particle_energy_rate                         = (T *)          malloc(particle_energy_size);
                    particle_momentum_rate                       = (vec<T> *)     malloc(particle_momentum_rate_size);

                    flow_terms                                   = (flow_aos<T> *)malloc(flow_term_size);
                    flow_grad_terms                              = (flow_aos<T> *)malloc(flow_term_size);
                    
                    for (uint64_t c = 0; c < local_mesh_size; c++)
                    {
                        flow_terms[c]      = {dummy_gas_vel, dummy_gas_pre, dummy_gas_tem};
                        flow_grad_terms[c] = {{0.0, 0.0, 0.0}, 0.0, 0.0};
                    }
                }
                else if (mpi_config->solver_type == PARTICLE || mpi_config->world_size == 1)
                {
                    cell_size_vector = points[cells[H_VERTEX] - shmem_point_disp] - points[cells[A_VERTEX] - shmem_point_disp];

                }
                cell_neighbours_array_size      = shmem_mesh_size * faces_per_cell * sizeof(uint64_t);

                // particles_per_point = (uint64_t *) malloc(particles_per_point_size);
                // cell_centres        = (vec<T> *)   malloc(cell_centre_size);

                MPI_Aint shmem_cell_winsize = cell_centre_size;
                MPI_Win_allocate_shared ( shmem_cell_winsize, sizeof(vec<double>), MPI_INFO_NULL, mpi_config->node_world, &cell_centres, &mpi_config->win_cell_centres );

                calculate_cell_centres();
                MPI_Barrier(mpi_config->world);

                
                uint64_t memory_usage          = get_memory_usage();
                uint64_t total_memory_usage    = memory_usage;
                uint64_t particle_memory_usage = memory_usage;

                uint64_t total_points_array_size               = points_array_size;
                uint64_t total_cells_array_size                = cells_array_size;
                uint64_t total_faces_array_size                = faces_array_size;
                uint64_t total_cell_centre_size                = cell_centre_size;
                uint64_t total_cell_neighbours_array_size      = cell_neighbours_array_size;
                uint64_t total_cells_per_point_size            = cells_per_point_size;
                uint64_t total_evaporated_fuel_mass_rate_size  = evaporated_fuel_mass_rate_size;
                uint64_t total_particle_energy_size            = particle_energy_size;
                uint64_t total_particle_momentum_rate_size     = particle_momentum_rate_size;
                uint64_t total_block_disp_size                 = 2 * block_disp_size;
                uint64_t total_flow_term_size                  = 2 * flow_term_size;        

                if (mpi_config->rank == 0)
                {
                    MPI_Reduce(MPI_IN_PLACE, &total_memory_usage,    1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);

                    MPI_Reduce(MPI_IN_PLACE, &total_points_array_size,               1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cells_array_size,                1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_faces_array_size,                1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_centre_size,                1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_neighbours_array_size,      1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cells_per_point_size,            1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_evaporated_fuel_mass_rate_size,  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_particle_energy_size,            1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_particle_momentum_rate_size,     1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_block_disp_size,                 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(MPI_IN_PLACE, &total_flow_term_size,                  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);

                    if (mpi_config->solver_type == PARTICLE)
                        MPI_Reduce(MPI_IN_PLACE, &particle_memory_usage, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);


                    printf("\nMesh storage requirements (%d processes) : \n", mpi_config->world_size);
                    printf("\tpoints_array_size                 (TOTAL %8.2f MB) (AVG %8.2f MB)   (%" PRIu64 " vertexes)\n"       , (float) total_points_array_size                / 1000000.0, (float) total_points_array_size                / (1000000.0 * mpi_config->world_size), points_size);
                    printf("\tcells_array_size                  (TOTAL %8.2f MB) (AVG %8.2f MB)   (%" PRIu64 " cells)\n"          , (float) total_cells_array_size                 / 1000000.0, (float) total_cells_array_size                 / (1000000.0 * mpi_config->world_size), mesh_size);
                    printf("\tfaces_array_size                  (TOTAL %8.2f MB) (AVG %8.2f MB)   (%" PRIu64 " faces)\n"          , (float) total_faces_array_size                 / 1000000.0, (float) total_faces_array_size                 / (1000000.0 * mpi_config->world_size), faces_size);
                    printf("\tcell_centre_size                  (TOTAL %8.2f MB) (AVG %8.2f MB) \n"                               , (float) total_cell_centre_size                 / 1000000.0, (float) total_cell_centre_size                 / (1000000.0 * mpi_config->world_size));
                    printf("\tcell_neighbours_array_size        (TOTAL %8.2f MB) (AVG %8.2f MB) \n"                               , (float) total_cell_neighbours_array_size       / 1000000.0, (float) total_cell_neighbours_array_size       / (1000000.0 * mpi_config->world_size));
                    printf("\tcells_per_point_size              (TOTAL %8.2f MB) (AVG %8.2f MB) \n"                               , (float) total_cells_per_point_size             / 1000000.0, (float) total_cells_per_point_size             / (1000000.0 * mpi_config->world_size));
                    printf("\tblock_disp_size                   (TOTAL %8.2f MB) (AVG %8.2f MB) \n"                               , (float) total_block_disp_size                  / 1000000.0, (float) total_block_disp_size                  / (1000000.0 * mpi_config->world_size));
                    printf("\tevaporated_fuel_mass_rate_size    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"                               , (float) total_evaporated_fuel_mass_rate_size   / 1000000.0, (float) total_evaporated_fuel_mass_rate_size   / (1000000.0 * mpi_config->world_size));
                    printf("\tparticle_energy_size              (TOTAL %8.2f MB) (AVG %8.2f MB) \n"                               , (float) total_particle_energy_size             / 1000000.0, (float) total_particle_energy_size             / (1000000.0 * mpi_config->world_size));
                    printf("\tparticle_momentum_rate_size       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"                               , (float) total_particle_momentum_rate_size      / 1000000.0, (float) total_particle_momentum_rate_size      / (1000000.0 * mpi_config->world_size));
                    printf("\t2 * block_disp_size               (TOTAL %8.2f MB) (AVG %8.2f MB) \n"                               , (float) total_block_disp_size                  / 1000000.0, (float) total_block_disp_size                  / (1000000.0 * mpi_config->world_size));
                    printf("\t2 * flow_term_size                (TOTAL %8.2f MB) (AVG %8.2f MB) \n\n"                             , (float) total_flow_term_size                   / 1000000.0, (float) total_flow_term_size                   / (1000000.0 * mpi_config->world_size));

                    printf("\tFlow rank mesh size               (TOTAL %12.2f MB) (AVG %.2f MB) \n"                    , (float)(total_memory_usage - particle_memory_usage)/1000000.0, (float)(total_memory_usage - particle_memory_usage)/(1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\tParticle rank mesh size           (TOTAL %12.2f MB) (AVG %.2f MB) \n"                    , (float)particle_memory_usage/1000000.0,                        (float)particle_memory_usage/(1000000.0 * (mpi_config->world_size - mpi_config->particle_flow_world_size)));
                    printf("\tTotal mesh size                   (TOTAL %12.2f MB) \n\n"                                , (float)total_memory_usage/1000000.0);
                }
                else
                {
                    MPI_Reduce(&total_memory_usage, nullptr,    1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);

                    MPI_Reduce(&total_points_array_size,               nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_cells_array_size,                nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_faces_array_size,                nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_cell_centre_size,                nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_cell_neighbours_array_size,      nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_cells_per_point_size,            nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_evaporated_fuel_mass_rate_size,  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_particle_energy_size,            nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_particle_momentum_rate_size,     nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_block_disp_size,                 nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);
                    MPI_Reduce(&total_flow_term_size,                  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->world);

                    if (mpi_config->solver_type == PARTICLE)
                        MPI_Reduce(&particle_memory_usage, nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                }
            }

            inline uint64_t get_memory_usage ()
            {
                return points_array_size 
                     + cells_array_size 
                     + faces_array_size 
                     + cell_centre_size 
                     + cell_neighbours_array_size 
                     + cells_per_point_size 
                     + evaporated_fuel_mass_rate_size 
                     + particle_energy_size 
                     + particle_momentum_rate_size 
                     + 2 * block_disp_size 
                     + 2 * flow_term_size;
            }

            // void clear_particles_per_point_array(void)
            // {
            //     memset(particles_per_point, 0, local_points_size * sizeof(uint64_t));
            // }

            inline uint64_t get_block_id(const uint64_t cell)
            {
                uint64_t low  = 0;
                uint64_t high = num_blocks;
                

                uint64_t block = UINT64_MAX;
                while (block == UINT64_MAX)
                {

                    uint64_t mid = (low + high) / 2;

                    if (low == mid)  block = mid;

                    if ( cell >= block_element_disp[low] && cell < block_element_disp[mid] )
                        high = mid;
                    else
                        low = mid;
                }
                return block;
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
