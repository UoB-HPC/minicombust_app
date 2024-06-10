#pragma once

#include "utils/utils.hpp"
#include <petscksp.h>

namespace minicombust::flow 
{
    template<class T>
    class FlowSolver 
    {
        private:

            uint64_t timestep_count = 0;

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
            phi_vector<T>  A_phi;
            phi_vector<T>  phi;
            //phi_vector<T>  old_phi;
            phi_vector<T>  S_phi;
            phi_vector<vec<T>> phi_grad;
            
            T effective_viscosity;
			T inlet_effective_viscosity;

			Mat A;
			Vec b, u;
			KSP ksp, pressure_ksp;

            T *cell_densities;
            T *cell_volumes;

            vector<int> ranks;
            int      *elements;
            uint64_t *element_disps;

            uint64_t nhalos = 0;
            vector<int> halo_ranks;
            vector<int> halo_sizes;
            vector<int> halo_disps;
            vector<MPI_Datatype> halo_mpi_double_datatypes;
            vector<MPI_Datatype> halo_mpi_vec_double_datatypes;
            vector<vector<uint64_t>> halo_rank_recv_indexes;
            unordered_map<uint64_t, uint64_t> boundary_map;

			vector<uint64_t*>            halo_indexes;
			vector<phi_vector<T>>        phi_send_buffers;
			vector<phi_vector<vec<T>>>   phi_grad_send_buffers;

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

			double compute_time;

            uint64_t max_storage;

            double flow_timings[22] = {0.0};
            double time_stats[11] = {0.0};
			double vel_total_time = 0.0, vel_flux_time = 0.0;
			double vel_setup_time = 0.0, vel_solve_time = 0.0;

			double pres_total_time = 0.0, pres_flux_time = 0.0;
			double pres_setup_time = 0.0, pres_solve_time = 0.0;
			double pres_halo_time = 0.0;
	
			double sca_total_time[7] = {0.0}, sca_flux_time[7] = {0.0};
			double sca_setup_time[7] = {0.0}, sca_solve_time[7] = {0.0};
			double sca_terb_time[2] = {0.0};

			double fgm_lookup_time = 0.0;	

            const MPI_Status empty_mpi_status = { 0, 0, 0, 0, 0};

			T FGM_table[100][100][100][100];

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

                async_locks = (bool*)malloc(4 * particle_ranks + 1 * sizeof(bool));
                
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

				compute_time = 0;

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

                setup_halos();

                phi_array_size        = (mesh->local_mesh_size + nhalos + mesh->boundary_cells_size) * sizeof(T);
                phi_grad_array_size   = (mesh->local_mesh_size + nhalos + mesh->boundary_cells_size) * sizeof(vec<T>);
                source_phi_array_size = (mesh->local_mesh_size + nhalos + mesh->boundary_cells_size) * sizeof(T);

                phi.U           = (T *)malloc(phi_array_size);
                phi.V           = (T *)malloc(phi_array_size);
                phi.W           = (T *)malloc(phi_array_size);
                phi.P           = (T *)malloc(phi_array_size);
				phi.PP          = (T *)malloc(phi_array_size);
				phi.TE          = (T *)malloc(phi_array_size);
				phi.ED          = (T *)malloc(phi_array_size);
				phi.TP          = (T *)malloc(phi_array_size);
                phi.TEM         = (T *)malloc(phi_array_size);
				phi.FUL			= (T *)malloc(phi_array_size);
				phi.PRO			= (T *)malloc(phi_array_size);
				phi.VARF		= (T *)malloc(phi_array_size);
				phi.VARP		= (T *)malloc(phi_array_size);
				//old_phi.U       = (T *)malloc(phi_array_size);
                //old_phi.V       = (T *)malloc(phi_array_size);
                //old_phi.W       = (T *)malloc(phi_array_size);
                //old_phi.P       = (T *)malloc(phi_array_size);
				//old_phi.PP      = (T *)malloc(phi_array_size);
                phi_grad.U      = (vec<T> *)malloc(phi_grad_array_size);
                phi_grad.V      = (vec<T> *)malloc(phi_grad_array_size);
                phi_grad.W      = (vec<T> *)malloc(phi_grad_array_size);
                phi_grad.P      = (vec<T> *)malloc(phi_grad_array_size);
                phi_grad.PP     = (vec<T> *)malloc(phi_grad_array_size);
				phi_grad.TE     = (vec<T> *)malloc(phi_grad_array_size);
                phi_grad.ED     = (vec<T> *)malloc(phi_grad_array_size);
				phi_grad.TEM    = (vec<T> *)malloc(phi_grad_array_size);
				phi_grad.FUL	= (vec<T> *)malloc(phi_grad_array_size);
				phi_grad.PRO	= (vec<T> *)malloc(phi_grad_array_size);
				phi_grad.VARF   = (vec<T> *)malloc(phi_grad_array_size);
				phi_grad.VARP   = (vec<T> *)malloc(phi_grad_array_size);
				A_phi.U         = (T *)malloc(source_phi_array_size);
                A_phi.V         = (T *)malloc(source_phi_array_size);
                A_phi.W         = (T *)malloc(source_phi_array_size);
				S_phi.U         = (T *)malloc(source_phi_array_size);
                S_phi.V         = (T *)malloc(source_phi_array_size);
                S_phi.W         = (T *)malloc(source_phi_array_size);

                density_array_size = (mesh->local_mesh_size + nhalos) * sizeof(T);
                volume_array_size  = (mesh->local_mesh_size + nhalos) * sizeof(T);
                cell_densities     = (T *)malloc(density_array_size);
                cell_volumes       = (T *)malloc(volume_array_size);

                #pragma ivdep
                for ( uint64_t face = 0; face < mesh->faces_size; face++ )  
                {
                    uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                    uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

                    if ( mesh->faces[face].cell1 >= mesh->mesh_size )  continue;

                    uint64_t node_count = 0;
                    uint64_t face_node_ids[4];
                    vec<T>  *face_nodes[4];
                    for ( uint64_t n0 = 0; n0 < mesh->cell_size; n0++ )
                    {
                        for ( uint64_t n1 = 0; n1 < mesh->cell_size; n1++ )
                        {
                            if ( mesh->cells[shmem_cell0 * mesh->cell_size + n0] == mesh->cells[shmem_cell1 * mesh->cell_size + n1] )
                                face_node_ids[node_count++] = mesh->cells[shmem_cell0 * mesh->cell_size + n0];
                        }
                    }

                    face_nodes[0] = &mesh->points[face_node_ids[0] - mesh->shmem_point_disp];
                    face_nodes[1] = &mesh->points[face_node_ids[1] - mesh->shmem_point_disp];
                    face_nodes[2] = &mesh->points[face_node_ids[2] - mesh->shmem_point_disp];
                    face_nodes[3] = &mesh->points[face_node_ids[3] - mesh->shmem_point_disp];

                    vec<T> cell0_cell1_vec = mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0];
                    
                    face_areas[face]       = magnitude(*(face_nodes[2]) - *(face_nodes[0])) * magnitude(*(face_nodes[1]) - *(face_nodes[0]));
					face_centers[face]     = (*(face_nodes[0]) + *(face_nodes[1]) + *(face_nodes[2]) + *(face_nodes[3]));
					face_centers[face]     = face_centers[face] / double(4.0);
                    
					face_lambdas[face]     = magnitude(face_centers[face] - mesh->cell_centers[shmem_cell0]) / magnitude(cell0_cell1_vec) ;
                    face_normals[face]     = cross_product(*(face_nodes[2]) - *(face_nodes[0]), *(face_nodes[1]) - *(face_nodes[0])); 

                    if ( dot_product(face_normals[face], mesh->cell_centers[shmem_cell1] -  mesh->cell_centers[shmem_cell0]) < 0 )
                                    face_normals[face] = -1. * face_normals[face];

                    face_rlencos[face]     = face_areas[face] / magnitude(cell0_cell1_vec) / vector_cosangle(face_normals[face], cell0_cell1_vec);


                    face_mass_fluxes[face] = 0.25 * face_normals[face].x;
                    face_mass_fluxes[face] = 0.0;
                }

                const T visc_lambda = 0.000014;  
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
                    phi.P[block_cell]     = mesh->dummy_gas_pre;
					phi.PP[block_cell]    = 0.0;
					phi.TE[block_cell]    = mesh->dummy_gas_turbTE;
					phi.ED[block_cell]    = mesh->dummy_gas_turbED;
					phi.TP[block_cell]    = 0.0;
					phi.TEM[block_cell]   = mesh->dummy_gas_tem;
					phi.FUL[block_cell]   = mesh->dummy_gas_fuel;
					phi.PRO[block_cell]   = mesh->dummy_gas_pro;
					phi.VARF[block_cell]  = mesh->dummy_gas_fuel;
					phi.VARP[block_cell]  = mesh->dummy_gas_pro;

                    // old_phi.U[block_cell] = mesh->dummy_gas_vel.x;
                    // old_phi.V[block_cell] = mesh->dummy_gas_vel.y;
                    // old_phi.W[block_cell] = mesh->dummy_gas_vel.z;
                    // old_phi.P[block_cell] = mesh->dummy_gas_pre;

                    phi_grad.U[block_cell]    = {0.0, 0.0, 0.0};
                    phi_grad.V[block_cell]    = {0.0, 0.0, 0.0};
                    phi_grad.W[block_cell]    = {0.0, 0.0, 0.0};
                    phi_grad.P[block_cell]    = {0.0, 0.0, 0.0};
					phi_grad.PP[block_cell]   = {0.0, 0.0, 0.0};
					phi_grad.TE[block_cell]   = {0.0, 0.0, 0.0};
					phi_grad.ED[block_cell]   = {0.0, 0.0, 0.0};
					phi_grad.TEM[block_cell]  = {0.0, 0.0, 0.0};
					phi_grad.FUL[block_cell]  = {0.0, 0.0, 0.0};
					phi_grad.PRO[block_cell]  = {0.0, 0.0, 0.0};
					phi_grad.VARF[block_cell] = {0.0, 0.0, 0.0};
					phi_grad.VARP[block_cell] = {0.0, 0.0, 0.0};

                    cell_densities[block_cell] = 1.2;

                    const T width   = magnitude(mesh->points[cell_nodes[B_VERTEX] - mesh->shmem_point_disp] - mesh->points[cell_nodes[A_VERTEX] - mesh->shmem_point_disp]);
                    const T height  = magnitude(mesh->points[cell_nodes[C_VERTEX] - mesh->shmem_point_disp] - mesh->points[cell_nodes[A_VERTEX] - mesh->shmem_point_disp]);
                    const T length  = magnitude(mesh->points[cell_nodes[E_VERTEX] - mesh->shmem_point_disp] - mesh->points[cell_nodes[A_VERTEX] - mesh->shmem_point_disp]);
                    cell_volumes[block_cell] = width * height * length;

                    for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
                    {
                        if ( mesh->cell_neighbours[shmem_cell * mesh->faces_per_cell + f] >= mesh->mesh_size )
                        {
                            const uint64_t face = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];

                            vec<T>  *face_nodes[4];
                            face_nodes[0] = &mesh->points[cell_nodes[CUBE_FACE_VERTEX_MAP[f][0]] - mesh->shmem_point_disp];
                            face_nodes[1] = &mesh->points[cell_nodes[CUBE_FACE_VERTEX_MAP[f][1]] - mesh->shmem_point_disp];
                            face_nodes[2] = &mesh->points[cell_nodes[CUBE_FACE_VERTEX_MAP[f][2]] - mesh->shmem_point_disp];
                            face_nodes[3] = &mesh->points[cell_nodes[CUBE_FACE_VERTEX_MAP[f][3]] - mesh->shmem_point_disp];
                            
                            face_lambdas[face]     = 1.0;
						
                            face_areas[face]       = magnitude(*face_nodes[2] - *face_nodes[0]) * magnitude(*face_nodes[1] - *face_nodes[0]);
							
							face_centers[face]     = (*face_nodes[0] + *face_nodes[1] + *face_nodes[2] + *face_nodes[3]) / 4.0;

							face_normals[face]      = cross_product(*face_nodes[2] - *face_nodes[0], *face_nodes[1] - *face_nodes[0]);

                            const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                            if ( mesh->boundary_types[boundary_cell] == INLET )
                            {
                                face_mass_fluxes[face] = 0.25;
                                face_mass_fluxes[face] = 0.0;
                                if ( dot_product(face_normals[face], mesh->cell_centers[shmem_cell] - face_centers[face]) > 0 )
                                    face_normals[face] = -1. * face_normals[face];
                            }
                            else if ( mesh->boundary_types[boundary_cell] == OUTLET )
                            {
                                face_mass_fluxes[face] = 0.25;
                                face_mass_fluxes[face] = 0.0;

                                if ( dot_product(face_normals[face], mesh->cell_centers[shmem_cell] - face_centers[face]) > 0 )
                                    face_normals[face] = -1. * face_normals[face];
                            }
                            else if ( mesh->boundary_types[boundary_cell] == WALL )
                            {
                                face_mass_fluxes[face] = 0.0;

                                if ( dot_product(face_normals[face], mesh->cell_centers[shmem_cell] - face_centers[face]) > 0 )
                                    face_normals[face] = -1. * face_normals[face];
                            }

                            vec<T> cell0_facecenter_vec = face_centers[face] - mesh->cell_centers[shmem_cell];
                            face_rlencos[face]          = face_areas[face] / magnitude(cell0_facecenter_vec) / vector_cosangle(face_normals[face], cell0_facecenter_vec);

                        }
                    }
                }


                exchange_phi_halos();
                exchange_cell_info_halos();

                #pragma ivdep
                for ( uint64_t boundary_cell = 0; boundary_cell < mesh->boundary_cells_size; boundary_cell++ )
                {
                    const uint64_t block_cell = boundary_cell + mesh->local_mesh_size + nhalos;

                    phi.U[block_cell]     = 0.0;
                    phi.V[block_cell]     = 0.0;
                    phi.W[block_cell]     = 0.0;
                    phi.P[block_cell]     = mesh->dummy_gas_pre;
					phi.PP[block_cell]    = 0.0;
					phi.TE[block_cell]    = mesh->dummy_gas_turbTE;
					phi.ED[block_cell]    = mesh->dummy_gas_turbED;
					phi.TP[block_cell]    = 0.0;
					phi.TEM[block_cell]   = mesh->dummy_gas_tem;
					phi.FUL[block_cell]   = mesh->dummy_gas_fuel;
                    phi.PRO[block_cell]   = mesh->dummy_gas_pro;
					phi.VARF[block_cell]  = mesh->dummy_gas_fuel;
                    phi.VARP[block_cell]  = mesh->dummy_gas_pro;

					T velmag2 = pow(mesh->dummy_gas_vel.x,2) + pow(mesh->dummy_gas_vel.y,2) + pow(mesh->dummy_gas_vel.z,2);
					inlet_effective_viscosity = effective_viscosity + 1.2*0.09*(3.0/2.0*((0.1*0.1)*velmag2))/((pow(0.09,0.75) * pow((3.0/2.0*((0.1*0.1)*velmag2)),1.5)) + 0.00000000000000000001);
                    
					if ( mesh->boundary_types[boundary_cell] == INLET )
                    {
                        phi.U[block_cell]     = mesh->dummy_gas_vel.x;
                        phi.V[block_cell]     = 0.0;
                        phi.W[block_cell]     = 0.0;
                        phi.P[block_cell]     = mesh->dummy_gas_pre;
						phi.PP[block_cell]    = 0.0;
						phi.TE[block_cell]    = 3.0/2.0*((0.1*0.1)*velmag2);
						phi.ED[block_cell]    = pow(0.09,0.75) * pow(phi.TE[block_cell],1.5);
						phi.TP[block_cell]    = 0.0;
						phi.TEM[block_cell]   = mesh->dummy_gas_tem;
						phi.FUL[block_cell]   = mesh->dummy_gas_fuel;
						phi.PRO[block_cell]   = 0.0;
						phi.VARF[block_cell]  = mesh->dummy_gas_fuel;
						phi.VARP[block_cell]  = 0.0;
					}
                    else if ( mesh->boundary_types[boundary_cell] == OUTLET )
                    {
                        phi.U[block_cell]     = mesh->dummy_gas_vel.x;
                        phi.V[block_cell]     = 0.0;
                        phi.W[block_cell]     = 0.0;
                        phi.P[block_cell]     = mesh->dummy_gas_pre;
						phi.PP[block_cell]    = 0.0;
						phi.TE[block_cell]    = mesh->dummy_gas_turbTE;
						phi.ED[block_cell]    = mesh->dummy_gas_turbED;
						phi.TP[block_cell]    = 0.0;
						phi.TEM[block_cell]   = mesh->dummy_gas_tem;
						phi.FUL[block_cell]   = mesh->dummy_gas_fuel;
						phi.PRO[block_cell]   = mesh->dummy_gas_pro;
						phi.VARF[block_cell]  = mesh->dummy_gas_fuel;
						phi.VARP[block_cell]  = mesh->dummy_gas_pro;
					}
                    else if ( mesh->boundary_types[boundary_cell] == WALL )
                    {
                        phi.U[block_cell]     = 0.0;
                        phi.V[block_cell]     = 0.0;
                        phi.W[block_cell]     = 0.0;
                        phi.P[block_cell]     = mesh->dummy_gas_pre;
						phi.PP[block_cell]    = 0.0;
						phi.TE[block_cell]    = mesh->dummy_gas_turbTE;
						phi.ED[block_cell]    = mesh->dummy_gas_turbED;
						phi.TP[block_cell]    = 0.0;
						phi.TEM[block_cell]   = 293.0;
						phi.FUL[block_cell]   = 0.0;
						phi.PRO[block_cell]   = 0.0;
						phi.VARF[block_cell]  = 0.0;
						phi.VARP[block_cell]  = 0.0;
					}

                    // old_phi.U[block_cell] = 0.0;
                    // old_phi.V[block_cell] = 0.0;
                    // old_phi.W[block_cell] = 0.0;
                    // old_phi.P[block_cell] = mesh->dummy_gas_pre;
 
                    phi_grad.U[block_cell]    = {0.0, 0.0, 0.0};
                    phi_grad.V[block_cell]    = {0.0, 0.0, 0.0};
                    phi_grad.W[block_cell]    = {0.0, 0.0, 0.0};
                    phi_grad.P[block_cell]    = {0.0, 0.0, 0.0};
					phi_grad.PP[block_cell]   = {0.0, 0.0, 0.0};
					phi_grad.TE[block_cell]   = {0.0, 0.0, 0.0};
					phi_grad.ED[block_cell]   = {0.0, 0.0, 0.0};                    
					phi_grad.TEM[block_cell]  = {0.0, 0.0, 0.0};
					phi_grad.FUL[block_cell]  = {0.0, 0.0, 0.0};
                    phi_grad.PRO[block_cell]  = {0.0, 0.0, 0.0};
					phi_grad.VARF[block_cell] = {0.0, 0.0, 0.0};
					phi_grad.VARP[block_cell] = {0.0, 0.0, 0.0}; 
                }


                if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Done cell data.\n", mpi_config->particle_flow_rank);

				//Mat, Vec and ksp for sparse linear solve
				MatCreate(PETSC_COMM_WORLD, &A);
				MatSetSizes(A, mesh->local_mesh_size, mesh->local_mesh_size, mesh->mesh_size, mesh->mesh_size);
				MatSetFromOptions(A);
				if(mpi_config->particle_flow_world_size > 1)
				{
					MatMPIAIJSetPreallocation(A, 7, NULL, 6, NULL);	
				}
				else
				{
					MatSeqAIJSetPreallocation(A, 7, NULL);
				}
				MatSetUp(A);

				VecCreate(PETSC_COMM_WORLD, &b);
				VecSetSizes(b, mesh->local_mesh_size, mesh->mesh_size);
				VecSetFromOptions(b);
				VecDuplicate(b, &u);

				PC pc;

				KSPCreate(PETSC_COMM_WORLD, &ksp);
				KSPSetType(ksp, KSPBCGS);
				KSPGetPC(ksp, &pc);
                PCSetType(pc, PCNONE);
				KSPSetPC(ksp, pc);
				KSPSetOperators(ksp, A, A);
				KSPSetFromOptions(ksp);

				KSPCreate(PETSC_COMM_WORLD, &pressure_ksp);
				KSPSetType(pressure_ksp, KSPFGMRES);
                KSPGetPC(pressure_ksp, &pc);
                PCSetType(pc, PCGAMG);
                PCGAMGSetThresholdScale(pc, 0.0);
                PetscReal thres[4] = {0.0,0.0,0.0,0.0};
                PCGAMGSetThreshold(pc, thres, 4);
                PCGAMGSetAggressiveLevels(pc,4);
				KSPSetPC(pressure_ksp, pc);	
                KSPSetOperators(pressure_ksp, A, A);
                KSPSetFromOptions(pressure_ksp);

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
                uint64_t total_phi_array_size                     = 13 * phi_array_size;
                uint64_t total_phi_grad_array_size                = 12 * phi_grad_array_size;
                uint64_t total_source_phi_array_size              = 3 * source_phi_array_size; 
                uint64_t total_A_array_size                       = 3 * source_phi_array_size;
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
                    MPI_Reduce(&total_volume_array_size,                  nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                    MPI_Reduce(&total_density_array_size,                 nullptr, 1, MPI_UINT64_T, MPI_SUM, 0, mpi_config->particle_flow_world);
                }

                MPI_Barrier(mpi_config->world);

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);
            }


            void process_halo_neighbour ( set<uint64_t>& unique_neighbours, uint64_t neighbour )
            {
				if ( neighbour >= mesh->mesh_size){
					//We don't want boundary nodes	
				}
                else if ( neighbour - mesh->local_cells_disp >= mesh->local_mesh_size ) // Boundary with other flow blocks.
                {
                    const uint64_t flow_rank = mesh->get_block_id( neighbour );

                    vector<int>::iterator it = find(halo_ranks.begin(), halo_ranks.end(), flow_rank) ;

                    if ( it == halo_ranks.end() ) // Neighbour flow block not in halo ranks
                    {
                        // Add flow rank and init recv sizes as 1
                        halo_ranks.push_back(flow_rank);
                        halo_rank_recv_indexes.push_back(vector<uint64_t>());
                        halo_rank_recv_indexes[halo_ranks.size()-1].push_back(neighbour);

                        // Record neighbour as seen, record index in phi arrays in map for later accesses.
                        unique_neighbours.insert(neighbour);
                    }
                    else if ( !unique_neighbours.contains(neighbour) )
                    {
                        // Record neighbour as seen, record index in phi arrays in map for later accesses. Increment amount of data to recieve later.
                        uint64_t halo_rank_index = distance(halo_ranks.begin(), it);
                        unique_neighbours.insert(neighbour);
                        halo_rank_recv_indexes[halo_rank_index].push_back(neighbour);
                    }
                }
            }

            void setup_halos ()
            {
                set<uint64_t> unique_neighbours;
                for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
                {
                    const uint64_t cell       = block_cell + mesh->local_cells_disp;
                    const uint64_t shmem_cell = cell       - mesh->shmem_cell_disp;
                    // Get 6 immediate neighbours
                    const uint64_t below_neighbour                = mesh->cell_neighbours[ shmem_cell * mesh->faces_per_cell + DOWN_FACE];
                    const uint64_t above_neighbour                = mesh->cell_neighbours[ shmem_cell * mesh->faces_per_cell + UP_FACE];
                    const uint64_t around_left_neighbour          = mesh->cell_neighbours[ shmem_cell * mesh->faces_per_cell + LEFT_FACE];
                    const uint64_t around_right_neighbour         = mesh->cell_neighbours[ shmem_cell * mesh->faces_per_cell + RIGHT_FACE];
                    const uint64_t around_front_neighbour         = mesh->cell_neighbours[ shmem_cell * mesh->faces_per_cell + FRONT_FACE];
                    const uint64_t around_back_neighbour          = mesh->cell_neighbours[ shmem_cell * mesh->faces_per_cell + BACK_FACE];
                    process_halo_neighbour(unique_neighbours, below_neighbour);             // Immediate neighbour cell indexes are correct   
                    process_halo_neighbour(unique_neighbours, above_neighbour);             // Immediate neighbour cell indexes are correct  
                    process_halo_neighbour(unique_neighbours, around_left_neighbour);       // Immediate neighbour cell indexes are correct   
                    process_halo_neighbour(unique_neighbours, around_right_neighbour);      // Immediate neighbour cell indexes are correct   
                    process_halo_neighbour(unique_neighbours, around_front_neighbour);      // Immediate neighbour cell indexes are correct   
                    process_halo_neighbour(unique_neighbours, around_back_neighbour);       // Immediate neighbour cell indexes are correct   


					//TODO: do we need this bit I don't think we need it.
                    // Get 8 cells neighbours around
                    if ( around_left_neighbour != MESH_BOUNDARY  )   // If neighbour isn't edge of mesh and isn't a halo cell
                    {
                        const uint64_t around_left_front_neighbour    = mesh->cell_neighbours[ (around_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                        const uint64_t around_left_back_neighbour     = mesh->cell_neighbours[ (around_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                        process_halo_neighbour(unique_neighbours, around_left_front_neighbour);    
                        process_halo_neighbour(unique_neighbours, around_left_back_neighbour);     
                    }
                    if ( around_right_neighbour != MESH_BOUNDARY )
                    {
                        const uint64_t around_right_front_neighbour   = mesh->cell_neighbours[ (around_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell + FRONT_FACE] ;
                        const uint64_t around_right_back_neighbour    = mesh->cell_neighbours[ (around_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell + BACK_FACE]  ;
                        process_halo_neighbour(unique_neighbours, around_right_front_neighbour);   
                        process_halo_neighbour(unique_neighbours, around_right_back_neighbour); 
                    }
                    if ( below_neighbour != MESH_BOUNDARY )
                    {
                        // Get 8 cells around below cell
                        const uint64_t below_left_neighbour           = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + LEFT_FACE]  ;
                        const uint64_t below_right_neighbour          = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + RIGHT_FACE] ;
                        const uint64_t below_front_neighbour          = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + FRONT_FACE] ;
                        const uint64_t below_back_neighbour           = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + BACK_FACE]  ;
                        process_halo_neighbour(unique_neighbours, below_left_neighbour);           
                        process_halo_neighbour(unique_neighbours, below_right_neighbour);          
                        process_halo_neighbour(unique_neighbours, below_front_neighbour);          
                        process_halo_neighbour(unique_neighbours, below_back_neighbour);           
                        if ( below_left_neighbour != MESH_BOUNDARY )
                        {
                            const uint64_t below_left_front_neighbour     = mesh->cell_neighbours[ (below_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + FRONT_FACE] ;
                            const uint64_t below_left_back_neighbour      = mesh->cell_neighbours[ (below_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + BACK_FACE]  ;
                            process_halo_neighbour(unique_neighbours, below_left_front_neighbour);     
                            process_halo_neighbour(unique_neighbours, below_left_back_neighbour);      
                        }
                        if ( below_right_neighbour != MESH_BOUNDARY )
                        {
                            const uint64_t below_right_front_neighbour    = mesh->cell_neighbours[ (below_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                            const uint64_t below_right_back_neighbour     = mesh->cell_neighbours[ (below_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                            process_halo_neighbour(unique_neighbours, below_right_front_neighbour);    
                            process_halo_neighbour(unique_neighbours, below_right_back_neighbour); 
                        }
                    }
                    if ( above_neighbour != MESH_BOUNDARY )
                    {
                        // Get 8 cells neighbours above
                        const uint64_t above_left_neighbour           = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + LEFT_FACE]  ;
                        const uint64_t above_right_neighbour          = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + RIGHT_FACE] ;
                        const uint64_t above_front_neighbour          = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + FRONT_FACE] ;
                        const uint64_t above_back_neighbour           = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + BACK_FACE]  ;
                        process_halo_neighbour(unique_neighbours, above_left_neighbour);           
                        process_halo_neighbour(unique_neighbours, above_right_neighbour);          
                        process_halo_neighbour(unique_neighbours, above_front_neighbour);          
                        process_halo_neighbour(unique_neighbours, above_back_neighbour);           
                        if ( above_left_neighbour != MESH_BOUNDARY )
                        {
                            const uint64_t above_left_front_neighbour     = mesh->cell_neighbours[ (above_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + FRONT_FACE] ;
                            const uint64_t above_left_back_neighbour      = mesh->cell_neighbours[ (above_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + BACK_FACE]  ;
                            process_halo_neighbour(unique_neighbours, above_left_front_neighbour);     
                            process_halo_neighbour(unique_neighbours, above_left_back_neighbour);      
                        }
                        if ( above_right_neighbour != MESH_BOUNDARY )
                        {
                            const uint64_t above_right_front_neighbour    = mesh->cell_neighbours[ (above_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                            const uint64_t above_right_back_neighbour     = mesh->cell_neighbours[ (above_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                            process_halo_neighbour(unique_neighbours, above_right_front_neighbour);    
                            process_halo_neighbour(unique_neighbours, above_right_back_neighbour);     
                        }
                    }
                }

				for(auto i : unique_neighbours)
				{
					boundary_map[i]  = mesh->local_mesh_size + nhalos++;
				}
				
				sort(halo_ranks.begin(), halo_ranks.end());
				sort(halo_rank_recv_indexes.begin(), halo_rank_recv_indexes.end());
                
				if ( halo_ranks.size() == 0 )  return;

                uint64_t current_disp = 0;
                halo_disps.push_back(current_disp);
                MPI_Request send_requests[halo_ranks.size()];
                for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
                {
                    const int num_indexes = halo_rank_recv_indexes[r].size();
                    MPI_Isend( halo_rank_recv_indexes[r].data(), num_indexes,  MPI_UINT64_T, halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[r] );
                    current_disp += num_indexes;
                    halo_sizes.push_back(num_indexes);
                    halo_disps.push_back(current_disp);
                }

                MPI_Status status;
                int buff_size = halo_rank_recv_indexes[0].size() + 1;
                int      *buffer      = (int *)      malloc(buff_size * sizeof(int));
                
				for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
                {
					uint64_t * uint_buffer = (uint64_t *) malloc(buff_size * sizeof(uint64_t));
                    int num_indexes;
                    MPI_Probe (halo_ranks[r], 0, mpi_config->particle_flow_world, &status );
                    MPI_Get_count( &status, MPI_UINT64_T, &num_indexes );
                    if ( num_indexes > buff_size )
                    {
                        buff_size = num_indexes;
                        buffer      = (int *)      realloc(buffer,      (buff_size + 1) * sizeof(int));
                        uint_buffer = (uint64_t *) realloc(uint_buffer, (buff_size + 1) * sizeof(uint64_t));
                    }

                    MPI_Recv( uint_buffer, num_indexes, MPI_UINT64_T, halo_ranks[r], 0, mpi_config->particle_flow_world, MPI_STATUS_IGNORE );

					for (int i = 0; i < num_indexes; i++)
                    {
                        buffer[i] = (int)(uint_buffer[i] - mesh->local_cells_disp);
						uint_buffer[i] = uint_buffer[i] - mesh->local_cells_disp;
					}
					
					halo_indexes.push_back(uint_buffer);

					phi_vector<T> phi_tmp;
					phi_vector<vec<T>> phi_grad_tmp;
			
					phi_tmp.U            = (T *)malloc(sizeof(T) * num_indexes);
					phi_tmp.V            = (T *)malloc(sizeof(T) * num_indexes);
					phi_tmp.W            = (T *)malloc(sizeof(T) * num_indexes);
					phi_tmp.P            = (T *)malloc(sizeof(T) * num_indexes);
                    phi_tmp.PP           = (T *)malloc(sizeof(T) * num_indexes);
                    phi_tmp.TE           = (T *)malloc(sizeof(T) * num_indexes);
					phi_tmp.ED           = (T *)malloc(sizeof(T) * num_indexes);
                    phi_tmp.TEM          = (T *)malloc(sizeof(T) * num_indexes);
                    phi_tmp.FUL          = (T *)malloc(sizeof(T) * num_indexes);
					phi_tmp.PRO          = (T *)malloc(sizeof(T) * num_indexes);
                    phi_tmp.VARF         = (T *)malloc(sizeof(T) * num_indexes);
                    phi_tmp.VARP         = (T *)malloc(sizeof(T) * num_indexes);					

					phi_grad_tmp.U       = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
					phi_grad_tmp.V       = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
					phi_grad_tmp.W       = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
					phi_grad_tmp.P       = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
                    phi_grad_tmp.PP      = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
                    phi_grad_tmp.TE      = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
					phi_grad_tmp.ED      = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
                    phi_grad_tmp.TEM     = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
                    phi_grad_tmp.FUL     = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
					phi_grad_tmp.PRO     = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
                    phi_grad_tmp.VARF    = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);
                    phi_grad_tmp.VARP    = (vec<T> *)malloc(sizeof(vec<T>) * num_indexes);

					phi_send_buffers.push_back(phi_tmp);
					phi_grad_send_buffers.push_back(phi_grad_tmp);


                    MPI_Datatype indexed_type, vec_indexed_type;
                    MPI_Type_create_indexed_block(num_indexes, 1, buffer, MPI_DOUBLE,                    &indexed_type);
                    MPI_Type_create_indexed_block(num_indexes, 1, buffer, mpi_config->MPI_VEC_STRUCTURE, &vec_indexed_type);
                    MPI_Type_commit(&indexed_type);
                    MPI_Type_commit(&vec_indexed_type);
                    halo_mpi_double_datatypes.push_back(indexed_type);
                    halo_mpi_vec_double_datatypes.push_back(vec_indexed_type);
                }

                MPI_Waitall(halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
                halo_rank_recv_indexes.clear();

                free(buffer);
            }


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
                uint64_t total_phi_array_size                     = 13 * phi_array_size;
                uint64_t total_phi_grad_array_size                = 12 * phi_grad_array_size;
                uint64_t total_source_phi_array_size              = 3 * source_phi_array_size;
				uint64_t total_A_array_size						  = 3 * source_phi_array_size;

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
                       total_face_areas_array_size + total_face_lambdas_array_size + total_face_rlencos_array_size +
					   total_A_array_size;
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

			void pack_phi_halo_buffer(phi_vector<T> send_buffer, phi_vector<T> phi, uint64_t *indexes, uint64_t buf_size);			
			void pack_phi_grad_halo_buffer(phi_vector<vec<T>> send_buffer, phi_vector<vec<T>> phi_grad, uint64_t *indexes, uint64_t buf_size);
			void pack_PP_halo_buffer(phi_vector<T> send_buffer, phi_vector<T> phi, uint64_t *indexes, uint64_t buf_size);
			void pack_PP_grad_halo_buffer(phi_vector<vec<T>> send_buffer, phi_vector<vec<T>> phi_grad, uint64_t *indexes, uint64_t buf_size);
			void pack_Aphi_halo_buffer(phi_vector<T> send_buffer, phi_vector<T> phi, uint64_t *indexes, uint64_t buf_size);

			void output_data(uint64_t timestep);
            void print_logger_stats(uint64_t timesteps, double runtime);
            
            void exchange_cell_info_halos ();
            void exchange_grad_halos();
			void exchange_phi_halos();
			void exchange_single_phi_halo(T *phi_component);
			void exchange_single_grad_halo(vec<T> *phi_grad_component);	
            void exchange_A_halos (T *A_phi_component);
            
            void get_neighbour_cells(const uint64_t recv_id);
            void interpolate_to_nodes();

            void update_flow_field();  // Synchronize point with flow solver

			void set_up_field();
			void precomp_AU();
			void precomp_mass_flux();
			void set_up_fgm_table();
			
            void setup_sparse_matrix  ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component );
            void update_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component );
            void solve_sparse_matrix ( T *phi_component);
            void calculate_flux_UVW ();
            void calculate_UVW ();

            void calculate_mass_flux ();
            void setup_pressure_matrix  ( );
            void solve_pressure_matrix  ( );
            void calculate_pressure ();
			void Update_P_at_boundaries(T *phi_component);
			void update_mass_flux();
			void update_P(T *phi_component, vec<T> *phi_grad_component);            

            void solve(T A[][3], T *b, T *out);
            void get_phi_gradients ();
            void limit_phi_gradients ();
			void limit_phi_gradient(T *phi_component, vec<T> *phi_grad_component);
			void get_phi_gradient ( T *phi_component, vec<T> *phi_grad_component, bool pressure );

			void Scalar_solve(int type, T *phi_component, vec<T> *phi_grad_component);
			void FluxScalar(int type, T *phi_component, vec<T> *phi_grad_component);
			void solveTurbulenceModels(int type);

			void FGM_loop_up();

            void timestep();
    }; // class FlowSolver

}   // namespace minicombust::flow 
