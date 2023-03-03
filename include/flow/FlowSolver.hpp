#pragma once

#include "utils/utils.hpp"

#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

namespace minicombust::flow 
{
    template<class T>
    class FlowSolver 
    {
        using Eigen::RowMajor;

        private:

            Mesh<T> *mesh;

            double delta;

            vector<uint64_t *>        neighbour_indexes;
            vector<particle_aos<T> *> cell_particle_aos;

            T turbulence_field;
            T combustion_field;
            T flow_field;

            vector<unordered_set<uint64_t>>             unordered_neighbours_set;
            unordered_set<uint64_t>                     new_cells_set;
            vector<unordered_map<uint64_t, uint64_t>>   cell_particle_field_map;
            unordered_map<uint64_t, uint64_t>           node_to_position_map;
            vector<unordered_set<uint64_t>>             local_particle_node_sets;

            uint64_t    *interp_node_indexes;
            flow_aos<T> *interp_node_flow_fields;
            uint64_t    *send_buffers_interp_node_indexes;
            flow_aos<T> *send_buffers_interp_node_flow_fields;

            bool *async_locks;
            
            uint64_t         *send_counts;         
            uint64_t        **recv_indexes;        
            particle_aos<T> **recv_indexed_fields; 

            Face<T>       *face_fields;
            T             *face_mass_fluxes;
            T             *face_areas;
            T             *face_lambdas;
            T             *face_rlencos;
            vec<T>        *face_normals;
            vec<T>        *face_centers;
            T             *residual;
            phi_vector<T>  A_phi;
            phi_vector<T>  phi;
            phi_vector<T>  old_phi;
            phi_vector<T>  S_phi;
            phi_vector<vec<T>> phi_grad;
            Eigen::SparseMatrix<T, RowMajor> A_spmatrix;
            Eigen::BiCGSTAB<Eigen::SparseMatrix<T>> eigen_solver;

            T effective_viscosity;

            T *cell_densities;
            T *cell_volumes;

            vector<int> ranks;
            int      *elements;
            uint64_t *element_disps;

            MPI_Request bcast_request;
            vector<MPI_Status>  statuses;
            vector<MPI_Request> send_requests;
            vector<MPI_Request> recv_requests;

            Flow_Logger logger;


        public:
            MPI_Config *mpi_config;
            PerformanceLogger<T> performance_logger;

            vector<size_t> cell_index_array_size;
            vector<size_t> cell_particle_array_size;

            size_t cell_flow_array_size;

            size_t node_index_array_size;
            size_t node_flow_array_size;

            size_t send_buffers_node_index_array_size;
            size_t send_buffers_node_flow_array_size;

            size_t face_field_array_size;
            size_t face_mass_fluxes_array_size;
            size_t face_areas_array_size;
            size_t face_centers_array_size;
            size_t face_lambdas_array_size;
            size_t face_rlencos_array_size;
            size_t face_normals_array_size;
            size_t phi_array_size;
            size_t phi_grad_array_size;
            size_t source_phi_array_size;
            
            size_t density_array_size;
            size_t volume_array_size;

            uint64_t max_storage;

            double time_stats[11] = {0.0};

            const MPI_Status empty_mpi_status = { 0, 0, 0, 0, 0};

            FlowSolver(MPI_Config *mpi_config, Mesh<T> *mesh, double delta) : mesh(mesh), delta(delta), mpi_config(mpi_config)
            {
                if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Entered FlowSolver constructor.\n", mpi_config->particle_flow_rank);
                
                const float fraction  = 0.125;
                max_storage           = max((uint64_t)(fraction * mesh->local_mesh_size), 1UL);

                int particle_ranks = mpi_config->world_size - mpi_config->particle_flow_world_size;

                // Compute array sizes
                cell_index_array_size.push_back(max_storage    * sizeof(uint64_t));
                cell_particle_array_size.push_back(max_storage * sizeof(particle_aos<T>));

                node_index_array_size   = max_storage * sizeof(uint64_t);
                node_flow_array_size    = max_storage * sizeof(flow_aos<T>);

                send_buffers_node_index_array_size   = max_storage * sizeof(uint64_t);
                send_buffers_node_flow_array_size    = max_storage * sizeof(flow_aos<T>);

                async_locks = (bool*)malloc(4 * mesh->num_blocks * sizeof(bool));
                
                send_counts    =              (uint64_t*) malloc(mesh->num_blocks * sizeof(uint64_t));
                recv_indexes   =             (uint64_t**) malloc(mesh->num_blocks * sizeof(uint64_t*));
                recv_indexed_fields = (particle_aos<T>**) malloc(mesh->num_blocks * sizeof(particle_aos<T>*));

                elements        = (int*)malloc(particle_ranks          * sizeof(int));
                element_disps   = (uint64_t*)malloc((particle_ranks+1) * sizeof(uint64_t));

                // Allocate arrays
                neighbour_indexes.push_back((uint64_t*)         malloc(cell_index_array_size[0]));
                cell_particle_aos.push_back((particle_aos<T> * )malloc(cell_particle_array_size[0]));

                local_particle_node_sets.push_back(unordered_set<uint64_t>());

                interp_node_indexes      = (uint64_t * )    malloc(node_index_array_size);
                interp_node_flow_fields  = (flow_aos<T> * ) malloc(node_flow_array_size);

                send_buffers_interp_node_indexes      = (uint64_t * )    malloc(send_buffers_node_index_array_size);
                send_buffers_interp_node_flow_fields  = (flow_aos<T> * ) malloc(send_buffers_node_flow_array_size);

                unordered_neighbours_set.push_back(unordered_set<uint64_t>());
                cell_particle_field_map.push_back(unordered_map<uint64_t, uint64_t>());

                // Allocate face data
                face_field_array_size       = mesh->faces_size * sizeof(Face<T>);
                face_centers_array_size     = mesh->faces_size * sizeof(vec<T>);
                face_normals_array_size     = mesh->faces_size * sizeof(vec<T>);
                face_mass_fluxes_array_size = mesh->faces_size * sizeof(T);
                face_areas_array_size       = mesh->faces_size * sizeof(T);
                face_lambdas_array_size     = mesh->faces_size * sizeof(T);
                face_rlencos_array_size     = mesh->faces_size * sizeof(T);

                face_fields      = (Face<T> *) malloc( face_field_array_size       );
                face_centers     = (vec<T>  *) malloc( face_centers_array_size     );
                face_normals     = (vec<T>  *) malloc( face_normals_array_size     );
                face_mass_fluxes = (T *)       malloc( face_mass_fluxes_array_size );       
                face_areas       = (T *)       malloc( face_areas_array_size       ); 
                face_lambdas     = (T *)       malloc( face_lambdas_array_size     );   
                face_rlencos     = (T *)       malloc( face_rlencos_array_size     );   

                uint64_t nboundaries = 1;
                phi_array_size        = (mesh->local_mesh_size + nboundaries) * sizeof(T);
                phi_grad_array_size   = (mesh->local_mesh_size + nboundaries) * sizeof(vec<T>);
                source_phi_array_size =  mesh->local_mesh_size                * sizeof(T);
                phi.U           = (T *)malloc(phi_array_size);
                phi.V           = (T *)malloc(phi_array_size);
                phi.W           = (T *)malloc(phi_array_size);
                phi.P           = (T *)malloc(phi_array_size);
                old_phi.U       = (T *)malloc(phi_array_size);
                old_phi.V       = (T *)malloc(phi_array_size);
                old_phi.W       = (T *)malloc(phi_array_size);
                old_phi.P       = (T *)malloc(phi_array_size);
                phi_grad.U      = (vec<T> *)malloc(phi_grad_array_size);
                phi_grad.V      = (vec<T> *)malloc(phi_grad_array_size);
                phi_grad.W      = (vec<T> *)malloc(phi_grad_array_size);
                phi_grad.P      = (vec<T> *)malloc(phi_grad_array_size);
                A_phi.U         = (T *)malloc(source_phi_array_size);
                A_phi.V         = (T *)malloc(source_phi_array_size);
                A_phi.W         = (T *)malloc(source_phi_array_size);
                A_phi.P         = (T *)malloc(source_phi_array_size);
                S_phi.U         = (T *)malloc(source_phi_array_size);
                S_phi.V         = (T *)malloc(source_phi_array_size);
                S_phi.W         = (T *)malloc(source_phi_array_size);
                S_phi.P         = (T *)malloc(source_phi_array_size);
                residual        = (T *)malloc(source_phi_array_size);

                density_array_size = mesh->local_mesh_size * sizeof(T);
                volume_array_size  = mesh->local_mesh_size * sizeof(T);
                cell_densities     = (T *)malloc(density_array_size);
                cell_volumes       = (T *)malloc(volume_array_size);

                if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Setting up face data.\n", mpi_config->particle_flow_rank);


                #pragma ivdep
                for ( uint64_t face = 0; face < mesh->faces_size; face++ )  
                {
                    uint64_t cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                    uint64_t cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

                    if ( mesh->faces[face].cell1 == MESH_BOUNDARY )  continue;

                    uint64_t *cell_nodes0 = &mesh->cells[cell0 * mesh->cell_size];
                    uint64_t *cell_nodes1 = &mesh->cells[cell1 * mesh->cell_size];

                    uint64_t node_count = 0;
                    uint64_t face_node_ids[4];
                    vec<T>  *face_nodes[4];
                    for ( uint64_t n0 = 0; n0 < mesh->cell_size; n0++ )
                    {
                        for ( uint64_t n1 = 0; n1 < mesh->cell_size; n1++ )
                        {
                            if ( cell_nodes0[n0] == cell_nodes1[n1] )
                                face_node_ids[node_count++] = cell_nodes0[n0];
                        }
                    }

                    face_nodes[0] = &mesh->points[face_node_ids[0] - mesh->shmem_point_disp];
                    face_nodes[1] = &mesh->points[face_node_ids[1] - mesh->shmem_point_disp];
                    face_nodes[2] = &mesh->points[face_node_ids[2] - mesh->shmem_point_disp];
                    face_nodes[3] = &mesh->points[face_node_ids[3] - mesh->shmem_point_disp];

                    vec<T> cell0_cell1_vec = mesh->cell_centers[cell1] - mesh->cell_centers[cell0];
                    
                    face_mass_fluxes[face] = 0.0;
                    face_areas[face]       = magnitude(*face_nodes[2] - *face_nodes[0]) * magnitude(*face_nodes[1] - *face_nodes[0]);
                    face_centers[face]     = (*face_nodes[0] + *face_nodes[1] + *face_nodes[2] + *face_nodes[3]) / 4.0;
                    face_lambdas[face]     = magnitude(face_centers[face] - mesh->cell_centers[cell0]) / magnitude(cell0_cell1_vec) ;
                    face_normals[face]     = normalise(cross_product(*face_nodes[2] - *face_nodes[0], *face_nodes[1] - *face_nodes[0])); 
                    face_rlencos[face]     = face_areas[face] / magnitude(cell0_cell1_vec) / vector_cosangle(face_normals[face], cell0_cell1_vec);
                }

                const T visc_lambda = 0.001;  
                effective_viscosity = visc_lambda; // NOTE: Localise this to cells and boundaries when implementing Turbulence model

                if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Setting up cell data.\n", mpi_config->particle_flow_rank);

                #pragma ivdep
                for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
                {
                    const uint64_t shmem_cell = block_cell + mesh->local_cells_disp - mesh->shmem_cell_disp;
                    const uint64_t *cell_nodes = &mesh->cells[shmem_cell * mesh->cell_size];

                    A_phi.U[block_cell]   = mesh->dummy_gas_vel.x;
                    A_phi.V[block_cell]   = mesh->dummy_gas_vel.y;
                    A_phi.W[block_cell]   = mesh->dummy_gas_vel.z;

                    phi.U[block_cell]     = mesh->dummy_gas_vel.x;
                    phi.V[block_cell]     = mesh->dummy_gas_vel.y;
                    phi.W[block_cell]     = mesh->dummy_gas_vel.z;

                    old_phi.U[block_cell] = mesh->dummy_gas_vel.x;
                    old_phi.V[block_cell] = mesh->dummy_gas_vel.y;
                    old_phi.W[block_cell] = mesh->dummy_gas_vel.z;

                    cell_densities[block_cell] = 1.2;

                    const T width   = cell_nodes[B_VERTEX] - cell_nodes[A_VERTEX];
                    const T height  = cell_nodes[C_VERTEX] - cell_nodes[A_VERTEX];
                    const T length  = cell_nodes[E_VERTEX] - cell_nodes[A_VERTEX];
                    cell_volumes[block_cell] = width * height * length;

                    for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
                    {
                        if ( mesh->cell_neighbours[shmem_cell * mesh->faces_per_cell + f] == MESH_BOUNDARY )
                        {
                            const uint64_t face = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];

                            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

                            vec<T>  *face_nodes[4];
                            face_nodes[0] = &mesh->points[cell_nodes[CUBE_FACE_VERTEX_MAP[f][0]] - mesh->shmem_point_disp];
                            face_nodes[1] = &mesh->points[cell_nodes[CUBE_FACE_VERTEX_MAP[f][1]] - mesh->shmem_point_disp];
                            face_nodes[2] = &mesh->points[cell_nodes[CUBE_FACE_VERTEX_MAP[f][2]] - mesh->shmem_point_disp];
                            face_nodes[3] = &mesh->points[cell_nodes[CUBE_FACE_VERTEX_MAP[f][3]] - mesh->shmem_point_disp];

                            face_mass_fluxes[face] = 0.0;
                            face_lambdas[face]     = 1.0;
                            face_areas[face]       = magnitude(*face_nodes[2] - *face_nodes[0]) * magnitude(*face_nodes[1] - *face_nodes[0]);
                            face_centers[face]     = (*face_nodes[0] + *face_nodes[1] + *face_nodes[2] + *face_nodes[3]) / 4.0;
                            face_normals[face]     = normalise(cross_product(*face_nodes[2] - *face_nodes[0], *face_nodes[1] - *face_nodes[0])); 

                            vec<T> cell0_center_vec = mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0];
                            face_rlencos[face]     = face_areas[face] / magnitude(cell0_center_vec) / vector_cosangle(face_normals[face], cell0_center_vec);
                        }
                    }
                }

                if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Done cell data.\n", mpi_config->particle_flow_rank);


                Eigen::SparseMatrix<T, RowMajor> new_matrix( mesh->local_mesh_size, mesh->local_mesh_size + 1 );
                new_matrix.reserve( mesh->faces_size );
                A_spmatrix = new_matrix;

                send_requests.push_back ( MPI_REQUEST_NULL );
                send_requests.push_back ( MPI_REQUEST_NULL );
                recv_requests.push_back ( MPI_REQUEST_NULL );
                recv_requests.push_back ( MPI_REQUEST_NULL );
                statuses.push_back ( empty_mpi_status );

                memset(&logger, 0, sizeof(Flow_Logger));

                // Array sizes
                uint64_t total_node_index_array_size              = node_index_array_size;
                uint64_t total_node_flow_array_size               = node_flow_array_size;
                uint64_t total_send_buffers_node_index_array_size = send_buffers_node_index_array_size;
                uint64_t total_send_buffers_node_flow_array_size  = send_buffers_node_flow_array_size;
                uint64_t total_face_field_array_size              = face_field_array_size;
                uint64_t total_face_centers_array_size            = face_centers_array_size;
                uint64_t total_face_normals_array_size            = face_normals_array_size;
                uint64_t total_face_mass_fluxes_array_size        = face_mass_fluxes_array_size;
                uint64_t total_face_areas_array_size              = face_areas_array_size;
                uint64_t total_face_lambdas_array_size            = face_lambdas_array_size;
                uint64_t total_face_rlencos_array_size            = face_rlencos_array_size;
                uint64_t total_phi_array_size                     = 8 * phi_array_size;
                uint64_t total_phi_grad_array_size                = 4 * phi_grad_array_size;
                uint64_t total_source_phi_array_size              = 4 * source_phi_array_size;
                uint64_t total_A_array_size                       = 4 * source_phi_array_size;
                uint64_t total_residual_size                      = source_phi_array_size;
                uint64_t total_volume_array_size                  = volume_array_size;
                uint64_t total_density_array_size                 = density_array_size;

                // STL sizes
                uint64_t total_unordered_neighbours_set_size   = unordered_neighbours_set[0].size() * sizeof(uint64_t) ;
                uint64_t total_cell_particle_field_map_size    = cell_particle_field_map[0].size()  * sizeof(uint64_t);
                uint64_t total_node_to_position_map_size       = node_to_position_map.size()        * sizeof(uint64_t);
                uint64_t total_mpi_requests_size               = recv_requests.size() * send_requests.size() * sizeof(MPI_Request);
                uint64_t total_mpi_statuses_size               = statuses.size()                             * sizeof(MPI_Status);
                uint64_t total_new_cells_size                  = new_cells_set.size()                        * sizeof(uint64_t);
                uint64_t total_ranks_size                      = ranks.size()                                * sizeof(uint64_t);
                
                uint64_t total_local_particle_node_sets_size   = 0;
                for ( uint64_t i = 0; i < local_particle_node_sets.size(); i++ )
                    total_local_particle_node_sets_size += local_particle_node_sets[i].size() * sizeof(uint64_t);

                uint64_t total_cell_index_array_size    = 0;
                uint64_t total_cell_particle_array_size = 0;
                for ( uint64_t i = 0; i < cell_index_array_size.size(); i++ )
                {
                    total_cell_index_array_size    = cell_index_array_size[i];
                    total_cell_particle_array_size = cell_particle_array_size[i];
                }

                uint64_t total_memory_usage = get_array_memory_usage() + get_stl_memory_usage();
                if (mpi_config->particle_flow_rank == 0)
                {
                    MPI_Reduce(MPI_IN_PLACE, &total_memory_usage,                           1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);

                    MPI_Reduce(MPI_IN_PLACE, &total_cell_index_array_size,                  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_particle_array_size,               1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_node_index_array_size,                  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_node_flow_array_size,                   1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_send_buffers_node_index_array_size,     1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_send_buffers_node_flow_array_size,      1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_unordered_neighbours_set_size,          1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_cell_particle_field_map_size,           1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_node_to_position_map_size,              1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_mpi_requests_size,                      1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_mpi_statuses_size,                      1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_new_cells_size,                         1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_ranks_size,                             1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_local_particle_node_sets_size,          1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_face_field_array_size,                  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_face_centers_array_size,                1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_face_normals_array_size,                1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_face_mass_fluxes_array_size,            1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_face_areas_array_size,                  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_face_lambdas_array_size,                1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_face_rlencos_array_size,                1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_phi_array_size,                         1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_phi_grad_array_size,                    1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_source_phi_array_size,                  1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_A_array_size,                           1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_residual_size,                          1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_volume_array_size,                      1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(MPI_IN_PLACE, &total_density_array_size,                     1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);


                    printf("Flow solver storage requirements (%d processes) : \n", mpi_config->particle_flow_world_size);
                    printf("\ttotal_cell_index_array_size                               (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_index_array_size              / 1000000.0, (float) total_cell_index_array_size              / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_cell_particle_array_size                            (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_array_size           / 1000000.0, (float) total_cell_particle_array_size           / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_node_index_array_size                               (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_index_array_size              / 1000000.0, (float) total_node_index_array_size              / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_node_flow_array_size                                (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_flow_array_size               / 1000000.0, (float) total_node_flow_array_size               / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_send_buffers_node_index_array_size                  (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_send_buffers_node_index_array_size / 1000000.0, (float) total_send_buffers_node_index_array_size / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_send_buffers_node_flow_array_size                   (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_send_buffers_node_flow_array_size  / 1000000.0, (float) total_send_buffers_node_flow_array_size  / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_face_field_array_size                               (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_face_field_array_size              / 1000000.0, (float) total_face_field_array_size              / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_face_centers_array_size                             (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_face_centers_array_size            / 1000000.0, (float) total_face_centers_array_size            / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_face_normals_array_size                             (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_face_normals_array_size            / 1000000.0, (float) total_face_normals_array_size            / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_face_mass_fluxes_array_size                         (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_face_mass_fluxes_array_size        / 1000000.0, (float) total_face_mass_fluxes_array_size        / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_face_areas_array_size                               (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_face_areas_array_size              / 1000000.0, (float) total_face_areas_array_size              / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_face_lambdas_array_size                             (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_face_lambdas_array_size            / 1000000.0, (float) total_face_lambdas_array_size            / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_face_rlencos_array_size                             (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_face_rlencos_array_size            / 1000000.0, (float) total_face_rlencos_array_size            / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_phi_array_size                                      (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_phi_array_size                     / 1000000.0, (float) total_phi_array_size                     / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_phi_grad_array_size                                 (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_phi_grad_array_size                / 1000000.0, (float) total_phi_grad_array_size                / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_S_phi_array_size                                    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_source_phi_array_size              / 1000000.0, (float) total_source_phi_array_size              / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_A_array_size                                        (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_A_array_size                       / 1000000.0, (float) total_A_array_size                       / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_residual_size                                       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_residual_size                      / 1000000.0, (float) total_residual_size                      / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_volume_array_size                                   (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_volume_array_size                  / 1000000.0, (float) total_volume_array_size                  / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_density_array_size                                  (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_density_array_size                 / 1000000.0, (float) total_density_array_size                 / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_unordered_neighbours_set_size       (STL set)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_unordered_neighbours_set_size      / 1000000.0, (float) total_unordered_neighbours_set_size      / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_cell_particle_field_map_size        (STL map)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_field_map_size       / 1000000.0, (float) total_cell_particle_field_map_size       / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_node_to_position_map_size           (STL map)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_to_position_map_size          / 1000000.0, (float) total_node_to_position_map_size          / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_mpi_requests_size                   (STL vector)    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_mpi_requests_size                  / 1000000.0, (float) total_mpi_requests_size                  / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_mpi_statuses_size                   (STL vector)    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_mpi_statuses_size                  / 1000000.0, (float) total_mpi_statuses_size                  / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_ranks_size                          (STL vector)    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_ranks_size                         / 1000000.0, (float) total_ranks_size                         / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_new_cells_size                      (STL set)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_new_cells_size                     / 1000000.0, (float) total_new_cells_size                     / (1000000.0 * mpi_config->particle_flow_world_size));
                    printf("\ttotal_local_particle_node_sets_size       (STL set)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n\n"  , (float) total_local_particle_node_sets_size      / 1000000.0, (float) total_local_particle_node_sets_size      / (1000000.0 * mpi_config->particle_flow_world_size));
                    
                    printf("\tFlow solver size                                          (TOTAL %12.2f MB) (AVG %.2f MB) \n\n"  , (float)total_memory_usage                                          /1000000.0,  (float)total_memory_usage / (1000000.0 * mpi_config->particle_flow_world_size));
                }
                else
                {
                    MPI_Reduce(&total_memory_usage,                       nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);

                    MPI_Reduce(&total_cell_index_array_size,              nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_cell_particle_array_size,           nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_node_index_array_size,              nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_node_flow_array_size,               nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_send_buffers_node_index_array_size, nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_send_buffers_node_flow_array_size,  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_unordered_neighbours_set_size,      nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_cell_particle_field_map_size,       nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_node_to_position_map_size,          nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_mpi_requests_size,                  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_mpi_statuses_size,                  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_new_cells_size,                     nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_ranks_size,                         nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_local_particle_node_sets_size,      nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_face_field_array_size,              nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_face_centers_array_size,            nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_face_normals_array_size,            nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_face_mass_fluxes_array_size,        nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_face_areas_array_size,              nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_face_lambdas_array_size,            nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_face_rlencos_array_size,            nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_phi_array_size,                     nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_phi_grad_array_size,                nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_source_phi_array_size,              nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_A_array_size,                       nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_residual_size,                      nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_volume_array_size,                  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_density_array_size,                 nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                }

                MPI_Barrier(mpi_config->world);

                // for (uint64_t b = 0; b < mesh->num_blocks; b++)
                // {
                //     if ((uint64_t)mpi_config->particle_flow_rank == b)
                //     {
                //         MPI_Comm_split(mpi_config->world, 1, mpi_config->rank, &mpi_config->every_one_flow_world[b]);
                //         MPI_Comm_rank(mpi_config->every_one_flow_world[b], &mpi_config->every_one_flow_rank[b]);
                //         MPI_Comm_size(mpi_config->every_one_flow_world[b], &mpi_config->every_one_flow_world_size[b]);
                //     }
                //     else
                //     {
                //         MPI_Comm_split(mpi_config->world, MPI_UNDEFINED, mpi_config->rank, &mpi_config->every_one_flow_world[b]);
                //     }
                // }

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);
            }

            // void resize_cell_indexes(uint64_t *elements, uint64_t ***new_cell_indexes)
            // {
            //     while ( cell_index_array_size < ((uint64_t) *elements * sizeof(uint64_t)) )
            //     {
            //         cell_index_array_size     *= 2;

            //         neighbour_indexes     = (uint64_t*) realloc(neighbour_indexes, cell_index_array_size);
            //     }

            //     if (new_cell_indexes != NULL) **new_cell_indexes = neighbour_indexes;
            // }

            // void resize_cell_flow (uint64_t elements)
            // {
            //     resize_cell_indexes(elements, NULL);
            //     while ( cell_flow_array_size < ((size_t) elements * sizeof(flow_aos<T>)) )
            //     {
            //         cell_flow_array_size *= 2;

            //         neighbour_flow_aos_buffer      = (flow_aos<T> *)realloc(neighbour_flow_aos_buffer,      cell_flow_array_size);
            //         neighbour_flow_grad_aos_buffer = (flow_aos<T> *)realloc(neighbour_flow_grad_aos_buffer, cell_flow_array_size);
            //     }
            // }

            // void resize_cell_particle (uint64_t *elements, uint64_t ***new_cell_indexes, particle_aos<T> ***new_cell_particle)
            // {
            //     resize_cell_indexes(elements, new_cell_indexes);
            //     while ( cell_particle_array_size < ((size_t) *elements * sizeof(particle_aos<T>)) )
            //     {
            //         cell_particle_array_size *= 2;

            //         cell_particle_aos = (particle_aos<T> *)realloc(cell_particle_aos,  cell_particle_array_size);
            //     }

            //     if (new_cell_particle != NULL)  **new_cell_particle = cell_particle_aos;
            // }


            void resize_cell_particle (uint64_t elements, uint64_t index)
            {
                while ( cell_index_array_size[index] < ((size_t) elements * sizeof(uint64_t)) )
                {
                    cell_index_array_size[index]    *= 2;
                    cell_particle_array_size[index] *= 2;

                    neighbour_indexes[index] = (uint64_t *)       realloc(neighbour_indexes[index],  cell_index_array_size[index]);
                    cell_particle_aos[index] = (particle_aos<T> *)realloc(cell_particle_aos[index],  cell_particle_array_size[index]);
                }
            }

            void resize_nodes_arrays (uint64_t elements)
            {
                while ( node_index_array_size < ((size_t) elements * sizeof(uint64_t)) )
                {
                    node_index_array_size *= 2;
                    node_flow_array_size  *= 2;

                    interp_node_indexes     = (uint64_t*)    realloc(interp_node_indexes,     node_index_array_size);
                    interp_node_flow_fields = (flow_aos<T> *)realloc(interp_node_flow_fields, node_flow_array_size);
                }
            }

            void resize_send_buffers_nodes_arrays (uint64_t elements)
            {
                while ( send_buffers_node_index_array_size < ((size_t) elements * sizeof(uint64_t)) )
                {
                    send_buffers_node_index_array_size *= 2;
                    send_buffers_node_flow_array_size  *= 2;

                    send_buffers_interp_node_indexes     = (uint64_t*)    realloc(send_buffers_interp_node_indexes,     send_buffers_node_index_array_size);
                    send_buffers_interp_node_flow_fields = (flow_aos<T> *)realloc(send_buffers_interp_node_flow_fields, send_buffers_node_flow_array_size);
                }
            }

            size_t get_array_memory_usage ()
            {
                uint64_t total_node_index_array_size              = node_index_array_size;
                uint64_t total_node_flow_array_size               = node_flow_array_size;
                uint64_t total_send_buffers_node_index_array_size = send_buffers_node_index_array_size;
                uint64_t total_send_buffers_node_flow_array_size  = send_buffers_node_flow_array_size;
                uint64_t total_face_field_array_size              = face_field_array_size;
                uint64_t total_phi_array_size                     = 8 * phi_array_size;
                uint64_t total_phi_grad_array_size                = 4 * phi_grad_array_size;
                uint64_t total_source_phi_array_size              = 4 * source_phi_array_size;

                uint64_t total_face_centers_array_size            = face_centers_array_size;
                uint64_t total_face_normals_array_size            = face_normals_array_size;
                uint64_t total_face_mass_fluxes_array_size        = face_mass_fluxes_array_size;
                uint64_t total_face_areas_array_size              = face_areas_array_size;
                uint64_t total_face_lambdas_array_size            = face_lambdas_array_size;
                uint64_t total_face_rlencos_array_size            = face_rlencos_array_size;

                uint64_t total_cell_index_array_size    = 0;
                uint64_t total_cell_particle_array_size = 0;
                for ( uint64_t i = 0; i < cell_index_array_size.size(); i++ )
                {
                    total_cell_index_array_size    = cell_index_array_size[i];
                    total_cell_particle_array_size = cell_particle_array_size[i];
                }

                return total_cell_index_array_size + total_cell_particle_array_size + total_node_index_array_size + total_node_flow_array_size + 
                       total_send_buffers_node_index_array_size + total_send_buffers_node_flow_array_size + total_face_field_array_size + 
                       total_phi_array_size + total_source_phi_array_size + total_phi_grad_array_size +
                       total_face_centers_array_size + total_face_normals_array_size + total_face_mass_fluxes_array_size +
                       total_face_areas_array_size + total_face_lambdas_array_size + total_face_rlencos_array_size;
            }

            size_t get_stl_memory_usage ()
            {
                uint64_t total_unordered_neighbours_set_size   = unordered_neighbours_set[0].size()          * sizeof(uint64_t) ;
                uint64_t total_cell_particle_field_map_size    = cell_particle_field_map[0].size()           * sizeof(uint64_t);
                uint64_t total_node_to_position_map_size       = node_to_position_map.size()                 * sizeof(uint64_t);
                uint64_t total_mpi_requests_size               = recv_requests.size() * send_requests.size() * sizeof(MPI_Request);
                uint64_t total_mpi_statuses_size               = statuses.size()                             * sizeof(MPI_Status);
                uint64_t total_new_cells_size                  = new_cells_set.size()                        * sizeof(uint64_t);
                uint64_t total_ranks_size                      = ranks.size()                                * sizeof(uint64_t);
                uint64_t total_local_particle_node_sets_size   = 0;
                for ( uint64_t i = 0; i < local_particle_node_sets.size(); i++ )
                    total_local_particle_node_sets_size += local_particle_node_sets[i].size() * sizeof(uint64_t);


                return total_unordered_neighbours_set_size + total_cell_particle_field_map_size + total_node_to_position_map_size + total_mpi_requests_size + total_mpi_statuses_size + total_new_cells_size + total_local_particle_node_sets_size + total_ranks_size;
            }

            bool is_halo( uint64_t cell );

            void print_logger_stats(uint64_t timesteps, double runtime);
            
            void get_neighbour_cells(const uint64_t recv_id);
            void interpolate_to_nodes();

            void update_flow_field();  // Synchronize point with flow solver

            void setup_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component );
            void solve_sparse_matrix ( T *A_phi_component, T *phi_component, T *old_phi_component, T *S_phi_component );
            void calculate_flux_UVW ();
            void calculate_UVW ();
            void get_phi_gradient ( T *phi_component, vec<T> *phi_grad_component );

            void solve_combustion_equations();
            void update_combustion_fields();

            void solve_turbulence_equations();
            void update_turbulence_fields();

            void solve_flow_equations();

            void timestep();


    }; // class FlowSolver

}   // namespace minicombust::flow 