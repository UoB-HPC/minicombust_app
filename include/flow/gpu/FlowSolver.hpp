#pragma once
#include "utils/utils.hpp"
#include "amgx_c.h"
#include "cuda_runtime.h"

#define AMGX_SAFE_CALL(rc) \
{ \
  AMGX_RC err;     \
  char msg[4096];   \
  switch(err = (rc)) {    \
  case AMGX_RC_OK: \
    break; \
  default: \
    fprintf(stderr, "AMGX ERROR: file %s line %6d\n", __FILE__, __LINE__); \
    AMGX_get_error_string(err, msg, 4096);\
    fprintf(stderr, "AMGX ERROR: %s\n", msg); \
    AMGX_abort(NULL,1);\
    break; \
  } \
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* print callback*/
void print_callback(const char *msg, int length)
{
	int rank;
	int size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

   	if(rank == (size-1)) {printf("%s", msg);}
}

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

			bool first_press = true;
			bool first_mat = true;
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

			//GPU stuff
			phi_vector<T>        gpu_phi;
			phi_vector<T>	     gpu_A_phi;
			phi_vector<T>	     gpu_S_phi;
			phi_vector<vec<T>>   gpu_phi_grad;
			Face<uint64_t>	     *gpu_faces;
			uint64_t		     *gpu_cell_faces;
			uint64_t		     *gpu_boundary_map_keys;
			uint64_t		     *gpu_boundary_map_values;
			vec<T>   		     *gpu_face_centers;
			vec<T>			     *full_cell_centers;
		    vec<T>			     *gpu_cell_centers;
			T				     *gpu_face_rlencos;
			T				     *gpu_face_mass_fluxes;
			T				     *gpu_cell_densities;
			T				     *gpu_cell_volumes;
			uint64_t		     *gpu_boundary_types;
			T				     *gpu_face_lambdas;
			vec<T>			     *gpu_face_normals;
			T				     *gpu_face_areas;
			Face<T>			     *gpu_face_fields;
			particle_aos<double> *gpu_particle_terms;
			int					 *gpu_halo_ranks;
			int					 *gpu_halo_sizes;
			int					 *gpu_halo_disps;
			MPI_Datatype		 *gpu_halo_mpi_vec_double_datatypes;
			MPI_Datatype		 *gpu_halo_mpi_double_datatypes;
			double				 *gpu_fgm_table;

			int partition_vector_size;
            int *partition_vector;


			//TODO: we don't nee all this memory decolration anymore
			/*T  *full_data_A;
	        T  *full_data_bU;
    	    T  *full_data_bV;
        	T  *full_data_bW;
	        T  *full_data_bP;
    	    T  *full_data_bTE;
        	T  *full_data_bED;
	        T  *full_data_bT;
    	    T  *full_data_bFU;
        	T  *full_data_bPR;
	        T  *full_data_bVFU;
    	    T  *full_data_bVPR;*/

			/*size_t A_pitch;
	        size_t bU_pitch;
    	    size_t bV_pitch;
        	size_t bW_pitch;
	        size_t bP_pitch;
    	    size_t bTE_pitch;
        	size_t bED_pitch;
 	       	size_t bT_pitch;
    	    size_t bFU_pitch;
        	size_t bPR_pitch;
	        size_t bVFU_pitch;
    	    size_t bVPR_pitch;*/
	
			T *values;
        	int *nnz;
        	int *rows_ptr;
        	int64_t *col_indices;

//PETSC
/*			Mat A;
			Vec b, u;
			KSP ksp;

			Mat grad_A;
			Vec grad_b, grad_u, grad_bU, grad_bV, grad_bW;
			Vec grad_bP, grad_bTE, grad_bED, grad_bT, grad_bFU;
			Vec grad_bPR, grad_bVFU, grad_bVPR;
			KSP grad_ksp;
*/

			int nrings;
			AMGX_Mode mode = AMGX_mode_dDDI;
			
			AMGX_config_handle cfg;
			AMGX_resources_handle main_rsrc;
			AMGX_matrix_handle A;
			AMGX_vector_handle b, u;
			AMGX_solver_handle solver;
	
			AMGX_config_handle pressure_cfg;
            AMGX_resources_handle pressure_rsrc;
            AMGX_matrix_handle pressure_A;
            AMGX_vector_handle pressure_b, pressure_u;
            AMGX_solver_handle pressure_solver;

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

			double flow_timings[11] = {0.0};
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
               
				//Set up which GPUs to use
                int gpu_count = 0;
                cudaGetDeviceCount(&gpu_count);
                int rank = mpi_config->particle_flow_rank;
                int lrank = rank % gpu_count;
                printf("Process %d selecting device %d of %d\n", rank, lrank, gpu_count);
                cudaSetDevice(lrank);
 
				partition_vector_size = mesh->mesh_size;
            	partition_vector = (int *)malloc(partition_vector_size * sizeof(int));

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

                async_locks = (bool*)malloc((4 * mesh->num_blocks)+1 * sizeof(bool));
                
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

				//GPU
				cudaMalloc(&gpu_faces, mesh->faces_size*sizeof(Face<uint64_t>));
				cudaMalloc(&gpu_cell_faces, 
						   mesh->local_mesh_size * mesh->faces_per_cell	* sizeof(uint64_t));
	
				cudaMemcpy(gpu_faces, mesh->faces, 
						   mesh->faces_size*sizeof(Face<uint64_t>), cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_cell_faces, mesh->cell_faces, 
						   mesh->local_mesh_size * mesh->faces_per_cell * sizeof(uint64_t),
							cudaMemcpyHostToDevice);

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


                exchange_phi_halos_cpu();
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
					phi.TEM[block_cell]     = mesh->dummy_gas_tem;
					phi.FUL[block_cell]   = mesh->dummy_gas_fuel;
                    phi.PRO[block_cell]   = mesh->dummy_gas_pro;
					phi.VARF[block_cell]  = mesh->dummy_gas_fuel;
                    phi.VARP[block_cell]  = mesh->dummy_gas_pro;

					T velmag2 = pow(mesh->dummy_gas_vel.x,2) + pow(mesh->dummy_gas_vel.y,2) + pow(mesh->dummy_gas_vel.z,2);
					inlet_effective_viscosity = effective_viscosity + 1.2*0.09*(3.0/2.0*((0.1*0.1)*velmag2))/((pow(0.09,0.75) * pow((3.0/2.0*((0.1*0.1)*velmag2)),1.5)) + 0.00000000000000000001);
                    
					if ( mesh->boundary_types[boundary_cell] == INLET )
                    {
                        phi.U[block_cell]     = 50.0;
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
                        phi.U[block_cell]     = 50.0;
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

				//Transfer data to GPU
				cudaMalloc(&gpu_phi.U, phi_array_size);
                cudaMalloc(&gpu_phi.V, phi_array_size);
                cudaMalloc(&gpu_phi.W, phi_array_size);
				cudaMalloc(&gpu_phi.P, phi_array_size);
				cudaMalloc(&gpu_phi.PP, phi_array_size);
				cudaMalloc(&gpu_phi.TE, phi_array_size);
				cudaMalloc(&gpu_phi.ED, phi_array_size);
				cudaMalloc(&gpu_phi.TP, phi_array_size);
				cudaMalloc(&gpu_phi.TEM, phi_array_size);
				cudaMalloc(&gpu_phi.FUL, phi_array_size);
				cudaMalloc(&gpu_phi.PRO, phi_array_size);
				cudaMalloc(&gpu_phi.VARF, phi_array_size);
				cudaMalloc(&gpu_phi.VARP, phi_array_size);
				cudaMalloc(&gpu_A_phi.U, source_phi_array_size);
				cudaMalloc(&gpu_A_phi.V, source_phi_array_size);
				cudaMalloc(&gpu_A_phi.W, source_phi_array_size);
				cudaMalloc(&gpu_S_phi.U, source_phi_array_size);
				cudaMalloc(&gpu_S_phi.V, source_phi_array_size);
				cudaMalloc(&gpu_S_phi.W, source_phi_array_size);
				cudaMalloc(&gpu_face_rlencos, face_rlencos_array_size);
				cudaMalloc(&gpu_face_mass_fluxes, face_mass_fluxes_array_size);
				cudaMalloc(&gpu_cell_densities, density_array_size);
				cudaMalloc(&gpu_cell_volumes, volume_array_size);
				cudaMalloc(&gpu_boundary_types, 6 * sizeof(uint64_t));
				cudaMalloc(&gpu_face_lambdas, face_lambdas_array_size);
				cudaMalloc(&gpu_face_normals, face_normals_array_size);
				cudaMalloc(&gpu_boundary_map_keys, boundary_map.size() * sizeof(uint64_t));
				cudaMalloc(&gpu_boundary_map_values, boundary_map.size() * sizeof(uint64_t));
				cudaMalloc(&gpu_face_areas, face_areas_array_size);
				cudaMalloc(&gpu_face_fields, face_field_array_size);
				cudaMalloc(&gpu_particle_terms, mesh->local_mesh_size * sizeof(particle_aos<T>));
				cudaMalloc(&gpu_halo_ranks, halo_ranks.size() * sizeof(int));
				cudaMalloc(&gpu_halo_sizes, halo_sizes.size() * sizeof(int));
				cudaMalloc(&gpu_halo_disps, halo_disps.size() * sizeof(int));
				cudaMalloc(&gpu_halo_mpi_double_datatypes, halo_mpi_double_datatypes.size() * sizeof(MPI_Datatype));
				cudaMalloc(&gpu_halo_mpi_vec_double_datatypes, halo_mpi_vec_double_datatypes.size() * sizeof(MPI_Datatype));

				//async_locks = (bool*)malloc(halo_ranks.size() * sizeof(bool));

				gpuErrchk(cudaMalloc(&rows_ptr, sizeof(int) *(mesh->local_mesh_size+1)));
        		gpuErrchk(cudaMalloc(&col_indices, sizeof(int64_t) * (mesh->local_mesh_size*7)));
        		gpuErrchk(cudaMalloc(&values, sizeof(T) * (mesh->local_mesh_size*7)));
        		gpuErrchk(cudaMalloc(&nnz, sizeof(int)));

				//Allocate big data arrays Note: the storage for these is fairly large.
        		/*gpuErrchk(cudaMallocPitch(&full_data_A, &A_pitch, 9 * sizeof(T), mesh->local_mesh_size));
  		      	gpuErrchk(cudaMallocPitch(&full_data_bU, &bU_pitch, 3 * sizeof(T), mesh->local_mesh_size));
        		gpuErrchk(cudaMallocPitch(&full_data_bV, &bV_pitch, 3 * sizeof(T), mesh->local_mesh_size));
		        gpuErrchk(cudaMallocPitch(&full_data_bW, &bW_pitch, 3 * sizeof(T), mesh->local_mesh_size));
		        gpuErrchk(cudaMallocPitch(&full_data_bP, &bP_pitch, 3 * sizeof(T), mesh->local_mesh_size));
        		gpuErrchk(cudaMallocPitch(&full_data_bTE, &bTE_pitch, 3 * sizeof(T), mesh->local_mesh_size));
		        gpuErrchk(cudaMallocPitch(&full_data_bED, &bED_pitch, 3 * sizeof(T), mesh->local_mesh_size));
        		gpuErrchk(cudaMallocPitch(&full_data_bT, &bT_pitch, 3 * sizeof(T), mesh->local_mesh_size));
		        gpuErrchk(cudaMallocPitch(&full_data_bFU, &bFU_pitch, 3 * sizeof(T), mesh->local_mesh_size));
        		gpuErrchk(cudaMallocPitch(&full_data_bPR, &bPR_pitch, 3 * sizeof(T), mesh->local_mesh_size));
		        gpuErrchk(cudaMallocPitch(&full_data_bVFU, &bVFU_pitch, 3 * sizeof(T), mesh->local_mesh_size));
        		gpuErrchk(cudaMallocPitch(&full_data_bVPR, &bVPR_pitch, 3 * sizeof(T), mesh->local_mesh_size));*/

				/*gpuErrchk(cudaMallocPitch(&full_data_A, 9 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bU, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bV, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bW, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bP, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bTE, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bED, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bT, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bFU, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bPR, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bVFU, 3 * sizeof(T)));
                gpuErrchk(cudaMallocPitch(&full_data_bVPR, 3 * sizeof(T)));*/
			
				cudaMemcpy(gpu_phi.U, phi.U, phi_array_size, 
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.V, phi.V, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.W, phi.W, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.P, phi.P, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.PP, phi.PP, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.TE, phi.TE, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.ED, phi.ED, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.TP, phi.TP, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.TEM, phi.TEM, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.FUL, phi.FUL, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.PRO, phi.PRO, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.VARF, phi.VARF, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.VARP, phi.VARP, phi_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_A_phi.U, A_phi.U, source_phi_array_size,
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_A_phi.V, A_phi.V, source_phi_array_size,
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_A_phi.W, A_phi.W, source_phi_array_size,
						   cudaMemcpyHostToDevice);
				cudaMemset(gpu_S_phi.U, 0.0, source_phi_array_size);
                cudaMemset(gpu_S_phi.V, 0.0, source_phi_array_size);
                cudaMemset(gpu_S_phi.W, 0.0, source_phi_array_size);
				cudaMemcpy(gpu_face_rlencos, face_rlencos, face_rlencos_array_size,
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_mass_fluxes, face_mass_fluxes,
						   face_mass_fluxes_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_cell_densities, cell_densities, 
						   density_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_cell_volumes, cell_volumes,
						   volume_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_boundary_types, mesh->boundary_types, 6 * sizeof(uint64_t),
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_lambdas, face_lambdas, face_lambdas_array_size,
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_normals, face_normals, face_normals_array_size, 
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_areas, face_areas, face_areas_array_size,
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_fields, face_fields, face_field_array_size,
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_particle_terms, mesh->particle_terms, 
						   mesh->local_mesh_size * sizeof(particle_aos<T>),
						   cudaMemcpyHostToDevice);

				int* tmp = &halo_ranks[0];
				cudaMemcpy(gpu_halo_ranks, tmp, halo_ranks.size() * sizeof(int),
						   cudaMemcpyHostToDevice);
				tmp = &halo_sizes[0];
				cudaMemcpy(gpu_halo_sizes, tmp, halo_sizes.size() * sizeof(int),
						   cudaMemcpyHostToDevice);
				tmp = &halo_disps[0];
				cudaMemcpy(gpu_halo_disps, tmp, halo_disps.size() * sizeof(int),
						   cudaMemcpyHostToDevice);
				MPI_Datatype* type_tmp;
				type_tmp = &halo_mpi_double_datatypes[0];
				cudaMemcpy(gpu_halo_mpi_double_datatypes, type_tmp, 
						   halo_mpi_double_datatypes.size() * sizeof(MPI_Datatype),
						   cudaMemcpyHostToDevice);
				type_tmp = &halo_mpi_vec_double_datatypes[0];
				cudaMemcpy(gpu_halo_mpi_vec_double_datatypes, type_tmp,
						   halo_mpi_vec_double_datatypes.size() * sizeof(MPI_Datatype),
						   cudaMemcpyHostToDevice);

                cudaMalloc(&gpu_phi_grad.U, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.V, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.W, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.P, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.PP, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.TE, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.ED, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.TEM, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.FUL, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.PRO, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.VARF, phi_grad_array_size);
                cudaMalloc(&gpu_phi_grad.VARP, phi_grad_array_size);
	
				cudaMemcpy(gpu_phi_grad.U, phi_grad.U, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi_grad.V, phi_grad.V, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi_grad.W, phi_grad.W, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.P, phi_grad.P, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.PP, phi_grad.PP, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.TE, phi_grad.TE, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.ED, phi_grad.ED, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.TEM, phi_grad.TEM, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.FUL, phi_grad.FUL, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.PRO, phi_grad.PRO, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.VARF, phi_grad.VARF, phi_grad_array_size,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.VARP, phi_grad.VARP, phi_grad_array_size,
                           cudaMemcpyHostToDevice);

				cudaMalloc(&gpu_face_centers, mesh->faces_size * sizeof(vec<T>));
                cudaMalloc(&gpu_cell_centers, mesh->mesh_size * sizeof(vec<T>));

				//boundary map to gpu
				uint64_t * full_boundary_map_keys = (uint64_t *) malloc(boundary_map.size() * sizeof(uint64_t));
				uint64_t * full_boundary_map_values = (uint64_t *) malloc(boundary_map.size() * sizeof(uint64_t));
				int map_index = 0;
				for(const std::pair<uint64_t, uint64_t>& n : boundary_map)
				{
					full_boundary_map_keys[map_index] = n.first;
					full_boundary_map_values[map_index] = n.second;
					map_index++;
				}

				cudaMemcpy(gpu_boundary_map_keys, full_boundary_map_keys,
							boundary_map.size() * sizeof(uint64_t), 
							cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_boundary_map_values, full_boundary_map_values,
							boundary_map.size() * sizeof(uint64_t),
							cudaMemcpyHostToDevice);

				free(full_boundary_map_keys);
				free(full_boundary_map_values);

				//cell centers to gpu
				full_cell_centers = (vec<T> *) malloc(mesh->mesh_size * sizeof(vec<T>));
				for(uint64_t i = 0; i < mesh->mesh_size; i++)
				{
					const uint64_t cell = i - mesh->shmem_cell_disp;
					full_cell_centers[i] = mesh->cell_centers[cell];
					//printf("the center of cell %lu is (%3.8f,%3.8f,%3.8f)\n",i,full_cell_centers[i].x,full_cell_centers[i].y,full_cell_centers[i].z);
				} 

				cudaMemcpy(gpu_face_centers, face_centers, mesh->faces_size * sizeof(vec<T>),
						   cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_cell_centers, full_cell_centers, 
						   mesh->mesh_size * sizeof(vec<T>),
						   cudaMemcpyHostToDevice);

				free(full_cell_centers);

                if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Done cell data.\n", mpi_config->particle_flow_rank);

				//Create config for AMGX
				AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
				AMGX_SAFE_CALL(AMGX_install_signal_handler());
				AMGX_SAFE_CALL(AMGX_config_create_from_file(&pressure_cfg, "/scratch/space1/e609/suc/minicombust_app/test/solvers/FGMRES_PRESSURE.json"));
				AMGX_SAFE_CALL(AMGX_config_add_parameters(&pressure_cfg, "exception_handling=1"));
				
				AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, "/scratch/space1/e609/suc/minicombust_app/test/solvers/PBICGSTAB_NOPREC.json"));
                AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));	
				
				AMGX_resources_create(&main_rsrc, cfg,
									  &mpi_config->particle_flow_world, 1, &lrank);
				AMGX_resources_create(&pressure_rsrc, pressure_cfg,
                                      &mpi_config->particle_flow_world, 1, &lrank);	
				
				//Create matrix and vectors for AMGX
				AMGX_matrix_create(&A, main_rsrc, mode);
				AMGX_vector_create(&u, main_rsrc, mode);
				AMGX_vector_create(&b, main_rsrc, mode);

				AMGX_matrix_create(&pressure_A, pressure_rsrc, mode);
                AMGX_vector_create(&pressure_u, pressure_rsrc, mode);
                AMGX_vector_create(&pressure_b, pressure_rsrc, mode);
				
				//Create solvers for AMGX
				AMGX_solver_create(&solver, main_rsrc, mode, cfg);
				AMGX_solver_create(&pressure_solver, pressure_rsrc, mode, cfg);


				AMGX_config_get_default_number_of_rings(cfg, &nrings);

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
                uint64_t *uint_buffer = (uint64_t *) malloc(buff_size * sizeof(uint64_t));
                for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
                {
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
                    }

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

			void AMGX_free()
			{
				AMGX_solver_destroy(solver);
				AMGX_solver_destroy(pressure_solver);

				AMGX_vector_destroy(u);
				AMGX_vector_destroy(b);
				AMGX_vector_destroy(pressure_b);
				AMGX_vector_destroy(pressure_u);

				AMGX_matrix_destroy(A);
				AMGX_matrix_destroy(pressure_A);

				AMGX_resources_destroy(main_rsrc);
				AMGX_resources_destroy(pressure_rsrc);

				AMGX_config_destroy(cfg);
				AMGX_config_destroy(pressure_cfg);
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

			void output_data(uint64_t timestep);
            void print_logger_stats(uint64_t timesteps, double runtime);
            
            void exchange_cell_info_halos ();
            void exchange_grad_halos();
			void exchange_phi_halos();
			void exchange_phi_halos_cpu();
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

            void get_phi_gradients ();
            void limit_phi_gradients ();
			void limit_phi_gradient(T *phi_component, vec<T> *phi_grad_component);
			void get_phi_gradient ( T *phi_component, vec<T> *phi_grad_component);

			void Scalar_solve(int type, T *phi_component, vec<T> *phi_grad_component);
			void FluxScalar(int type, T *phi_component, vec<T> *phi_grad_component);
			void solveTurbulenceModels(int type);

			void FGM_look_up();

            void timestep();
    }; // class FlowSolver

}   // namespace minicombust::flow 