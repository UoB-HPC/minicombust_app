#pragma once
#include "utils/utils.hpp"
#include "amgx_c.h"
#include "cuda_runtime.h"
#include "MemoryTracker.hpp"



// #include <cuco/detail/static_map/static_map_ref.inl>
// #include "flow/gpu/gpu_hash_map.inl"
#include "flow/gpu/gpu_kernels.cuh"

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

            vector<uint64_t *>        gpu_neighbour_indexes;
            vector<particle_aos<T> *> gpu_cell_particle_aos;

            T turbulence_field;
            T combustion_field;
            T flow_field;

            vector<unordered_set<uint64_t>>             unordered_neighbours_set;
            unordered_set<uint64_t>                     new_cells_set;
            vector<unordered_map<uint64_t, uint64_t>>   cell_particle_field_map;
            unordered_map<uint64_t, uint64_t>           node_to_position_map;
            unordered_map<uint64_t, uint64_t>           global_node_to_local_node_map;
            vector<unordered_set<uint64_t>>             local_particle_node_sets;

            uint64_t    *interp_node_indexes;
            flow_aos<T> *interp_node_flow_fields;
            uint64_t    *send_buffers_interp_node_indexes;
            flow_aos<T> *send_buffers_interp_node_flow_fields;

            uint64_t    *gpu_send_buffers_interp_node_indexes;
            flow_aos<T> *gpu_send_buffers_interp_node_flow_fields;

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
            phi_vector<T>  phi_nodes;
			
            //phi_vector<T>  old_phi;
            phi_vector<T>  S_phi;
            phi_vector<vec<T>> phi_grad;
            
            T effective_viscosity;
			T inlet_effective_viscosity;

            Hash_map *node_hash_map, *boundary_hash_map;
            Hash_map *gpu_node_hash_map, *gpu_boundary_hash_map;

			//GPU stuff
			phi_vector<T>	     gpu_A_phi;
			phi_vector<T>	     gpu_S_phi;
			phi_vector<T>        gpu_phi;
			phi_vector<T>        gpu_phi_nodes;
			phi_vector<vec<T>>   gpu_phi_grad;
			Face<uint64_t>	     *gpu_faces;
			uint64_t		     *gpu_cell_faces;
			uint64_t		     *gpu_boundary_map;
			uint64_t		     *gpu_boundary_map_keys;
			uint64_t		     *gpu_boundary_map_values;
            int        		     *gpu_seen_node;
            uint32_t             *gpu_atomic_buffer_index;
            uint64_t             *gpu_node_buffer_disp;
            uint64_t		     *gpu_node_map;
			uint64_t		     *gpu_node_map_keys;
			uint64_t		     *gpu_node_map_values;
			vec<T>   		     *gpu_face_centers;
			vec<T>			     *full_cell_centers;
		    vec<T>			     *gpu_cell_centers;
		    vec<T>			     *gpu_local_nodes;
		    uint8_t			     *gpu_cells_per_point;
		    uint64_t			 *gpu_local_cells;
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
			double				 *gpu_fgm_table;

            cudaStream_t process_gpu_fields_stream;
            cudaStream_t stream2;

			int partition_vector_size;
            int *partition_vector;

            MemoryTracker *mtracker;


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
            vector<int> processed_ranks;
            int      *elements;
            int      *node_elements;
            uint64_t *element_disps;

            uint64_t nhalos = 0;
            vector<int> halo_ranks;
            vector<int> halo_sizes;
            vector<int> halo_disps;
            vector<MPI_Datatype> halo_mpi_double_datatypes;
            vector<MPI_Datatype> halo_mpi_vec_double_datatypes;
            vector<vector<uint64_t>> halo_rank_recv_indexes;
            unordered_map<uint64_t, uint64_t> boundary_map;

            vector<uint64_t*>            gpu_halo_indexes;
            vector<phi_vector<T>>        gpu_phi_send_buffers;
			vector<phi_vector<vec<T>>>   gpu_phi_grad_send_buffers;

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

            size_t send_buffer_disp;
            size_t send_buffers_node_index_array_size;
            size_t send_buffers_node_flow_array_size;
            size_t gpu_send_buffers_node_index_array_size;
            size_t gpu_send_buffers_node_flow_array_size;

            size_t face_field_array_size;
            size_t face_mass_fluxes_array_size;
            size_t face_areas_array_size;
            size_t face_centers_array_size;
            size_t face_lambdas_array_size;
            size_t face_rlencos_array_size;
            size_t face_normals_array_size;
            size_t phi_array_size;
            size_t phi_nodes_array_size;
            size_t phi_grad_array_size;
            size_t source_phi_array_size;
            
            size_t density_array_size;
            size_t volume_array_size;

            int gpu_send_buffer_elements;

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
               
                mtracker = new MemoryTracker(mpi_config);

                cudaStreamCreate(&process_gpu_fields_stream);
                cudaStreamCreate(&stream2);

				//Set up which GPUs to use
                int gpu_count = 0;
                cudaGetDeviceCount(&gpu_count);
                int rank = mpi_config->particle_flow_rank;
                int lrank = rank % gpu_count;
                // printf("Process %d selecting device %d of %d\n", rank, lrank, gpu_count);
            //     cudaSetDevice(lrank);
            //    cudaError_t ret1;                      
            //    for (int i = 0; i < gpu_count; i++)                                       
            //        ret1 = cudaDeviceEnablePeerAccess ( i, 0);  

                // printf("Compile time check:\n");                                              
                // #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT                   
                //     printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
                // #elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT                
                //     printf("This MPI library does not have CUDA-aware support.\n");               
                // #else                                                                             
                //     printf("This MPI library cannot determine if there is CUDA-aware support.\n"); 
                // #endif /* MPIX_CUDA_AWARE_SUPPORT */                                                    
                                                                                                        
                //     printf("Run time check:\n");                                                        
                // #if defined(MPIX_CUDA_AWARE_SUPPORT)                                                    
                //     if (1 == MPIX_Query_cuda_support()) {                                               
                //         printf("This MPI library has CUDA-aware support.\n");                           
                //     } else {                                                                            
                //         printf("This MPI library does not have CUDA-aware support.\n");                 
                //     }                                                                                   
                // #else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */                                           
                //     printf("This MPI library cannot determine if there is CUDA-aware support.\n");      
                // #endif /* MPIX_CUDA_AWARE_SUPPORT */      
 
				partition_vector_size = mesh->mesh_size;

                partition_vector = (int *)mtracker->allocate_host("partition_vector", partition_vector_size * sizeof(int));
                const float fraction  = 0.0005;
                max_storage           = max((uint64_t)(fraction * mesh->local_mesh_size), 1UL);

                int particle_ranks = mpi_config->world_size - mpi_config->particle_flow_world_size;

                // Compute array sizes
                cell_index_array_size.push_back(max_storage    * sizeof(uint64_t));
                cell_particle_array_size.push_back(max_storage * sizeof(particle_aos<T>));

                node_index_array_size   = max_storage * sizeof(uint64_t);
                node_flow_array_size    = max_storage * sizeof(flow_aos<T>);

                send_buffers_node_index_array_size      = max_storage * sizeof(uint64_t);
                send_buffers_node_flow_array_size       = max_storage * sizeof(flow_aos<T>);

                gpu_send_buffer_elements = (100 * 80000 / 10) * 8; // (niters * particles_per_iter / particles_per_cell) * 8 nodes_per_cell. 
                gpu_send_buffers_node_index_array_size  = gpu_send_buffer_elements * sizeof(uint64_t);
                gpu_send_buffers_node_flow_array_size   = gpu_send_buffer_elements * sizeof(flow_aos<T>);

                async_locks = (bool*)mtracker->allocate_host("async_locks", (4 * particle_ranks + 1) * sizeof(bool));
                
                send_counts    =              (uint64_t*) mtracker->allocate_host("send_counts", mesh->num_blocks * sizeof(uint64_t));
                recv_indexes   =             (uint64_t**) mtracker->allocate_host("recv_indexes", mesh->num_blocks * sizeof(uint64_t*));
                recv_indexed_fields = (particle_aos<T>**) mtracker->allocate_host("recv_indexed_fields", mesh->num_blocks * sizeof(particle_aos<T>*));

                elements        = (int*)mtracker->allocate_host("elements", particle_ranks          * sizeof(int));
                node_elements   = (int*)mtracker->allocate_host("elements", particle_ranks          * sizeof(int));
                element_disps   = (uint64_t*)mtracker->allocate_host("element_disps", (particle_ranks+1) * sizeof(uint64_t));

                // Allocate arrays
                neighbour_indexes.push_back((uint64_t*)          mtracker->allocate_host("neighbour_indexes", cell_index_array_size[0]));
                cell_particle_aos.push_back((particle_aos<T> * ) mtracker->allocate_host("cell_particle_aos", cell_particle_array_size[0]));
                uint64_t *gpu_neighbour_indexes_tmp;
                particle_aos<T> *gpu_cell_particle_aos_tmp;
                mtracker->allocate_device("gpu_neighbour_indexes_tmp", (void**)&gpu_neighbour_indexes_tmp,  cell_index_array_size[0]);
                mtracker->allocate_device("gpu_cell_particle_aos_tmp", (void**)&gpu_cell_particle_aos_tmp, cell_particle_array_size[0]);
                gpu_neighbour_indexes.push_back(gpu_neighbour_indexes_tmp);
                gpu_cell_particle_aos.push_back(gpu_cell_particle_aos_tmp);

                local_particle_node_sets.push_back(unordered_set<uint64_t>());

                interp_node_indexes      = (uint64_t * )    mtracker->allocate_host("interp_node_indexes", node_index_array_size);
                interp_node_flow_fields  = (flow_aos<T> * ) mtracker->allocate_host("interp_node_flow_fields", node_flow_array_size);

                send_buffers_interp_node_indexes            = (uint64_t * )    mtracker->allocate_host("send_buffers_interp_node_indexes", gpu_send_buffers_node_index_array_size);
                send_buffers_interp_node_flow_fields        = (flow_aos<T> * ) mtracker->allocate_host("send_buffers_interp_node_flow_fields", gpu_send_buffers_node_flow_array_size);

                mtracker->allocate_device("gpu_send_buffers_interp_node_indexes", (void**)&gpu_send_buffers_interp_node_indexes,     gpu_send_buffers_node_index_array_size);
                mtracker->allocate_device("gpu_send_buffers_interp_node_flow_fields", (void**)&gpu_send_buffers_interp_node_flow_fields, gpu_send_buffers_node_flow_array_size);

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

                face_fields      = (Face<T> *) mtracker->allocate_host("face_fields",  face_field_array_size       );
                face_centers     = (vec<T>  *) mtracker->allocate_host("face_centers",  face_centers_array_size     );
                face_normals     = (vec<T>  *) mtracker->allocate_host("face_normals",  face_normals_array_size     );
                face_mass_fluxes = (T *)       mtracker->allocate_host("face_mass_fluxes",  face_mass_fluxes_array_size );       
                face_areas       = (T *)       mtracker->allocate_host("face_areas",  face_areas_array_size       ); 
                face_lambdas     = (T *)       mtracker->allocate_host("face_lambdas",  face_lambdas_array_size     );   
                face_rlencos     = (T *)       mtracker->allocate_host("face_rlencos",  face_rlencos_array_size     );   

                setup_halos();

                // Create global node to local node map
                for (uint64_t local_cell = 0; local_cell < mesh->local_mesh_size; local_cell++)
                {
                    uint64_t cell = local_cell + mesh->local_cells_disp;

                    #pragma ivdep
                    for (uint64_t n = 0; n < mesh->cell_size; n++)
                    {
                        uint64_t node = mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + n];
                        
                        if (!global_node_to_local_node_map.contains(node))
                        {
                            global_node_to_local_node_map[node] = global_node_to_local_node_map.size();
                        }
                    }
                }
                // Iterate over halos
                for ( auto cell_halo_pair : boundary_map )
                {
                    uint64_t cell = cell_halo_pair.first;
                    uint64_t halo = cell_halo_pair.second;

                    #pragma ivdep
                    for (uint64_t n = 0; n < mesh->cell_size; n++)
                    {
                        uint64_t node = mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + n];
                        
                        if (!global_node_to_local_node_map.contains(node))
                        {
                            global_node_to_local_node_map[node] = global_node_to_local_node_map.size();
                        }
                    }
                }

                mtracker->allocate_device("gpu_seen_node",           (void**)&gpu_seen_node,           global_node_to_local_node_map.size() * sizeof(int) );
                mtracker->allocate_device("gpu_atomic_buffer_index", (void**)&gpu_atomic_buffer_index, sizeof(uint32_t) );
                mtracker->allocate_device("gpu_node_buffer_disp",    (void**)&gpu_node_buffer_disp, sizeof(uint64_t) );

                phi_array_size        = (mesh->local_mesh_size + nhalos + mesh->boundary_cells_size) * sizeof(T);
                phi_nodes_array_size  = (global_node_to_local_node_map.size()) * sizeof(T);
                phi_grad_array_size   = (mesh->local_mesh_size + nhalos + mesh->boundary_cells_size) * sizeof(vec<T>);
                source_phi_array_size = (mesh->local_mesh_size + nhalos + mesh->boundary_cells_size) * sizeof(T);

                mtracker->allocate_cuda_host("phi_nodes", (void**)&phi_nodes.U,   phi_nodes_array_size, (void*)0x0);
                mtracker->allocate_cuda_host("phi_nodes", (void**)&phi_nodes.V,   phi_nodes_array_size, (void*)0x0);
                mtracker->allocate_cuda_host("phi_nodes", (void**)&phi_nodes.W,   phi_nodes_array_size, (void*)0x0);
                mtracker->allocate_cuda_host("phi_nodes", (void**)&phi_nodes.P,   phi_nodes_array_size, (void*)0x0);
                mtracker->allocate_cuda_host("phi_nodes", (void**)&phi_nodes.TEM, phi_nodes_array_size, (void*)0x0);

                mtracker->allocate_device("gpu_phi_nodes", (void**)&gpu_phi_nodes.U,   phi_nodes_array_size, (void**)0x0);
                mtracker->allocate_device("gpu_phi_nodes", (void**)&gpu_phi_nodes.V,   phi_nodes_array_size, (void**)0x0);
                mtracker->allocate_device("gpu_phi_nodes", (void**)&gpu_phi_nodes.W,   phi_nodes_array_size, (void**)0x0);
                mtracker->allocate_device("gpu_phi_nodes", (void**)&gpu_phi_nodes.P,   phi_nodes_array_size, (void**)0x0);
                mtracker->allocate_device("gpu_phi_nodes", (void**)&gpu_phi_nodes.TEM, phi_nodes_array_size, (void**)0x0);

                mtracker->allocate_cuda_host("phi", (void**)&phi.U          , phi_array_size, (void*)0x1);
                mtracker->allocate_cuda_host("phi", (void**)&phi.V          , phi_array_size, (void*)0x1);
                mtracker->allocate_cuda_host("phi", (void**)&phi.W          , phi_array_size, (void*)0x1);
                mtracker->allocate_cuda_host("phi", (void**)&phi.P          , phi_array_size, (void*)0x1);
				mtracker->allocate_cuda_host("phi", (void**)&phi.PP         , phi_array_size, (void*)0x1);
				mtracker->allocate_cuda_host("phi", (void**)&phi.TE         , phi_array_size, (void*)0x1);
				mtracker->allocate_cuda_host("phi", (void**)&phi.ED         , phi_array_size, (void*)0x1);
				mtracker->allocate_cuda_host("phi", (void**)&phi.TP         , phi_array_size, (void*)0x1);
                mtracker->allocate_cuda_host("phi", (void**)&phi.TEM        , phi_array_size, (void*)0x1);
				mtracker->allocate_cuda_host("phi", (void**)&phi.FUL		, phi_array_size, (void*)0x1);
				mtracker->allocate_cuda_host("phi", (void**)&phi.PRO		, phi_array_size, (void*)0x1);
				mtracker->allocate_cuda_host("phi", (void**)&phi.VARF	    , phi_array_size, (void*)0x1);
				mtracker->allocate_cuda_host("phi", (void**)&phi.VARP	    , phi_array_size, (void*)0x1);
                phi_grad.U      = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
                phi_grad.V      = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
                phi_grad.W      = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
                phi_grad.P      = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
                phi_grad.PP     = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
				phi_grad.TE     = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
                phi_grad.ED     = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
				phi_grad.TEM    = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
				phi_grad.FUL	= (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
				phi_grad.PRO	= (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
				phi_grad.VARF   = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
				phi_grad.VARP   = (vec<T> *)mtracker->allocate_host("phi_grad", phi_grad_array_size, (void*)0x2);
				A_phi.U         = (T *)mtracker->allocate_host("A_phi", source_phi_array_size, (void*)0x3);
                A_phi.V         = (T *)mtracker->allocate_host("A_phi", source_phi_array_size, (void*)0x3);
                A_phi.W         = (T *)mtracker->allocate_host("A_phi", source_phi_array_size, (void*)0x3);
				S_phi.U         = (T *)mtracker->allocate_host("S_phi", source_phi_array_size, (void*)0x4);
                S_phi.V         = (T *)mtracker->allocate_host("S_phi", source_phi_array_size, (void*)0x4);
                S_phi.W         = (T *)mtracker->allocate_host("S_phi", source_phi_array_size, (void*)0x4);

				//GPU
				mtracker->allocate_device("gpu_faces", (void**)&gpu_faces, mesh->faces_size*sizeof(Face<uint64_t>));
				mtracker->allocate_device("gpu_cell_faces", (void**)&gpu_cell_faces, mesh->local_mesh_size * mesh->faces_per_cell	* sizeof(uint64_t));
	
				cudaMemcpy(gpu_faces, mesh->faces, 
						   mesh->faces_size*sizeof(Face<uint64_t>), cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_cell_faces, mesh->cell_faces, 
						   mesh->local_mesh_size * mesh->faces_per_cell * sizeof(uint64_t),
							cudaMemcpyHostToDevice);

				density_array_size = (mesh->local_mesh_size + nhalos) * sizeof(T);
                volume_array_size  = (mesh->local_mesh_size + nhalos) * sizeof(T);
                cell_densities     = (T *)mtracker->allocate_host("cell_densities", density_array_size);
                cell_volumes       = (T *)mtracker->allocate_host("cell_volumes", volume_array_size);

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
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.U, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.V, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.W, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.P, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.PP, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.TE, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.ED, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.TP, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.TEM, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.FUL, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.PRO, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.VARF, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_phi", (void**)&gpu_phi.VARP, phi_array_size, (void*)0x1);
                mtracker->allocate_device("gpu_A_phi", (void**)&gpu_A_phi.U, source_phi_array_size, (void*)0x2);
                mtracker->allocate_device("gpu_A_phi", (void**)&gpu_A_phi.V, source_phi_array_size, (void*)0x2);
                mtracker->allocate_device("gpu_A_phi", (void**)&gpu_A_phi.W, source_phi_array_size, (void*)0x2);
                mtracker->allocate_device("gpu_S_phi", (void**)&gpu_S_phi.U, source_phi_array_size, (void*)0x3);
                mtracker->allocate_device("gpu_S_phi", (void**)&gpu_S_phi.V, source_phi_array_size, (void*)0x3);
                mtracker->allocate_device("gpu_S_phi", (void**)&gpu_S_phi.W, source_phi_array_size, (void*)0x3);
                mtracker->allocate_device("gpu_face_rlencos", (void**)&gpu_face_rlencos, face_rlencos_array_size);
                mtracker->allocate_device("gpu_face_mass_fluxes", (void**)&gpu_face_mass_fluxes, face_mass_fluxes_array_size);
                mtracker->allocate_device("gpu_cell_densities", (void**)&gpu_cell_densities, density_array_size);
                mtracker->allocate_device("gpu_cell_volumes", (void**)&gpu_cell_volumes, volume_array_size);
                mtracker->allocate_device("gpu_boundary_types", (void**)&gpu_boundary_types, 6 * sizeof(uint64_t));
                mtracker->allocate_device("gpu_face_lambdas", (void**)&gpu_face_lambdas, face_lambdas_array_size);
                mtracker->allocate_device("gpu_face_normals", (void**)&gpu_face_normals, face_normals_array_size);
                mtracker->allocate_device("gpu_boundary_map", (void**)&gpu_boundary_map, mesh->mesh_size * sizeof(uint64_t));
                mtracker->allocate_device("gpu_boundary_map_keys", (void**)&gpu_boundary_map_keys, boundary_map.size() * sizeof(uint64_t));
                mtracker->allocate_device("gpu_boundary_map_values", (void**)&gpu_boundary_map_values, boundary_map.size() * sizeof(uint64_t));
                printf("global_node_to_local_node_map_size %lu ps %lu\n", global_node_to_local_node_map.size(), mesh->points_size);

                mtracker->allocate_device("gpu_node_map", (void**)&gpu_node_map,        mesh->points_size * sizeof(uint64_t));
				mtracker->allocate_device("gpu_node_map_keys", (void**)&gpu_node_map_keys,   global_node_to_local_node_map.size() * sizeof(uint64_t));
				mtracker->allocate_device("gpu_node_map_values", (void**)&gpu_node_map_values, global_node_to_local_node_map.size() * sizeof(uint64_t));
				mtracker->allocate_device("gpu_face_areas", (void**)&gpu_face_areas, face_areas_array_size);
				mtracker->allocate_device("gpu_face_fields", (void**)&gpu_face_fields, face_field_array_size);
				mtracker->allocate_device("gpu_particle_terms", (void**)&gpu_particle_terms, mesh->local_mesh_size * sizeof(particle_aos<T>));
				mtracker->allocate_device("gpu_halo_ranks", (void**)&gpu_halo_ranks, halo_ranks.size() * sizeof(int));
				mtracker->allocate_device("gpu_halo_sizes", (void**)&gpu_halo_sizes, halo_sizes.size() * sizeof(int));
				mtracker->allocate_device("gpu_halo_disps", (void**)&gpu_halo_disps, halo_disps.size() * sizeof(int));
				// mtracker->allocate_device("gpu_halo_mpi_double_datatypes", (void**)&gpu_halo_mpi_double_datatypes, halo_mpi_double_datatypes.size() * sizeof(MPI_Datatype));
				// mtracker->allocate_device("gpu_halo_mpi_vec_double_datatypes", (void**)&gpu_halo_mpi_vec_double_datatypes, halo_mpi_vec_double_datatypes.size() * sizeof(MPI_Datatype));

				mtracker->allocate_device("rows_ptr", (void**)&rows_ptr, sizeof(int) *(mesh->local_mesh_size+1));
        		mtracker->allocate_device("col_indices", (void**)&col_indices, sizeof(int64_t) * (mesh->local_mesh_size*7));
        		mtracker->allocate_device("values", (void**)&values, sizeof(T) * (mesh->local_mesh_size*7));
        		mtracker->allocate_device("nnz", (void**)&nnz, sizeof(int));

			
				cudaMemcpy(gpu_phi.U, phi.U, phi_array_size,  cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.V, phi.V, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.W, phi.W, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.P, phi.P, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.PP, phi.PP, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.TE, phi.TE, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.ED, phi.ED, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.TP, phi.TP, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.TEM, phi.TEM, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.FUL, phi.FUL, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.PRO, phi.PRO, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.VARF, phi.VARF, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi.VARP, phi.VARP, phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_A_phi.U, A_phi.U, source_phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_A_phi.V, A_phi.V, source_phi_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_A_phi.W, A_phi.W, source_phi_array_size, cudaMemcpyHostToDevice);


				cudaMemset(gpu_S_phi.U, 0.0, source_phi_array_size);
                cudaMemset(gpu_S_phi.V, 0.0, source_phi_array_size);
                cudaMemset(gpu_S_phi.W, 0.0, source_phi_array_size);
				cudaMemcpy(gpu_face_rlencos, face_rlencos, face_rlencos_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_mass_fluxes, face_mass_fluxes, face_mass_fluxes_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_cell_densities, cell_densities,  density_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_cell_volumes, cell_volumes, volume_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_boundary_types, mesh->boundary_types, 6 * sizeof(uint64_t), cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_lambdas, face_lambdas, face_lambdas_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_normals, face_normals, face_normals_array_size,  cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_areas, face_areas, face_areas_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_face_fields, face_fields, face_field_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_particle_terms, mesh->particle_terms,  mesh->local_mesh_size * sizeof(particle_aos<T>), cudaMemcpyHostToDevice);

				int* tmp = &halo_ranks[0];
				cudaMemcpy(gpu_halo_ranks, tmp, halo_ranks.size() * sizeof(int), cudaMemcpyHostToDevice);
				tmp = &halo_sizes[0];
				cudaMemcpy(gpu_halo_sizes, tmp, halo_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);
				tmp = &halo_disps[0];
				cudaMemcpy(gpu_halo_disps, tmp, halo_disps.size() * sizeof(int), cudaMemcpyHostToDevice);
				

                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.U, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.V, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.W, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.P, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.PP, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.TE, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.ED, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.TEM, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.FUL, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.PRO, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.VARF, phi_grad_array_size, (void*)0x4);
                mtracker->allocate_device("gpu_phi_grad", (void**)&gpu_phi_grad.VARP, phi_grad_array_size, (void*)0x4);
	
				cudaMemcpy(gpu_phi_grad.U, phi_grad.U, phi_grad_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi_grad.V, phi_grad.V, phi_grad_array_size, cudaMemcpyHostToDevice);
				cudaMemcpy(gpu_phi_grad.W, phi_grad.W, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.P, phi_grad.P, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.PP, phi_grad.PP, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.TE, phi_grad.TE, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.ED, phi_grad.ED, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.TEM, phi_grad.TEM, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.FUL, phi_grad.FUL, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.PRO, phi_grad.PRO, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.VARF, phi_grad.VARF, phi_grad_array_size, cudaMemcpyHostToDevice);
                cudaMemcpy(gpu_phi_grad.VARP, phi_grad.VARP, phi_grad_array_size, cudaMemcpyHostToDevice);

				mtracker->allocate_device("gpu_face_centers",    (void**)&gpu_face_centers,    mesh->faces_size         * sizeof(vec<T>));
                mtracker->allocate_device("gpu_cell_centers",    (void**)&gpu_cell_centers,   (mesh->local_mesh_size + nhalos) * sizeof(vec<T>));
                mtracker->allocate_device("gpu_local_nodes",     (void**)&gpu_local_nodes,     mesh->points_size        * sizeof(vec<T>));
                mtracker->allocate_device("gpu_cells_per_point", (void**)&gpu_cells_per_point, mesh->points_size        * sizeof(uint8_t));
                mtracker->allocate_device("gpu_local_cells",     (void**)&gpu_local_cells,    (mesh->local_mesh_size + nhalos) * mesh->cell_size * sizeof(uint64_t));

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

				gpuErrchk( cudaMemcpy(gpu_boundary_map_keys,   full_boundary_map_keys,   boundary_map.size() * sizeof(uint64_t),  cudaMemcpyHostToDevice));
				gpuErrchk( cudaMemcpy(gpu_boundary_map_values, full_boundary_map_values, boundary_map.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

                uint64_t *gpu_boundary_hash_map_keys, *gpu_boundary_hash_map_values;
                uint64_t next_pow2 = pow(2, ceil(log(boundary_map.size())/log(2)))*16;
                if (mpi_config->particle_flow_rank == 0) printf("Boundary map size: %lu real pow2 %lu\n", boundary_map.size(), next_pow2);
                mtracker->allocate_device("gpu_node_hash_map_keys",   (void**)&gpu_boundary_hash_map_keys,   next_pow2 * sizeof(uint64_t));
				mtracker->allocate_device("gpu_node_hash_map_values", (void**)&gpu_boundary_hash_map_values, next_pow2 * sizeof(uint64_t));
				mtracker->allocate_device("gpu_boundary_hash_map", (void**)&gpu_boundary_hash_map, sizeof(Hash_map));
                boundary_hash_map = new Hash_map(mpi_config->particle_flow_rank, next_pow2, gpu_boundary_hash_map_keys, gpu_boundary_hash_map_values);
                gpuErrchk( cudaMemcpy(gpu_boundary_hash_map, boundary_hash_map, sizeof(Hash_map), cudaMemcpyHostToDevice));
                



                if (boundary_map.size() != 0)
                {
                    int thread_count = min( (int) 32, (int)boundary_map.size());
                    int block_count  = max(1, (int) ceil((double) (boundary_map.size()) / (double) thread_count));
                    cudaDeviceSynchronize();
		            gpuErrchk( cudaPeekAtLastError() );
                    C_create_map(block_count, thread_count, gpu_boundary_hash_map, gpu_boundary_map_keys, gpu_boundary_map_values, boundary_map.size());
                    cudaDeviceSynchronize();
		            gpuErrchk( cudaPeekAtLastError() );
                }

                free(full_boundary_map_keys);
				free(full_boundary_map_values);

                //node map to gpu
				uint64_t * full_node_map_keys = (uint64_t *) malloc(global_node_to_local_node_map.size() * sizeof(uint64_t));
				uint64_t * full_node_map_values = (uint64_t *) malloc(global_node_to_local_node_map.size() * sizeof(uint64_t));

				map_index = 0;
				for(const std::pair<uint64_t, uint64_t>& n : global_node_to_local_node_map)
				{
					full_node_map_keys[map_index] = n.first;
					full_node_map_values[map_index] = n.second;
					map_index++;
				}
				gpuErrchk( cudaMemset(gpu_node_map, (int)MESH_BOUNDARY, global_node_to_local_node_map.size()  * sizeof(uint64_t)));
				gpuErrchk( cudaMemcpy(gpu_node_map_keys,   full_node_map_keys,  global_node_to_local_node_map.size()  * sizeof(uint64_t),  cudaMemcpyHostToDevice));
				gpuErrchk( cudaMemcpy(gpu_node_map_values, full_node_map_values, global_node_to_local_node_map.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
                
                uint64_t *gpu_node_hash_map_keys, *gpu_node_hash_map_values;
                next_pow2 = pow(2, ceil(log(global_node_to_local_node_map.size())/log(2)))*16;
                if (mpi_config->particle_flow_rank == 0) printf("Node map size: %lu  real pow2 %lu\n", global_node_to_local_node_map.size(), next_pow2);
                mtracker->allocate_device("gpu_node_hash_map_keys",   (void**)&gpu_node_hash_map_keys,   next_pow2 * sizeof(uint64_t));
				mtracker->allocate_device("gpu_node_hash_map_values", (void**)&gpu_node_hash_map_values, next_pow2 * sizeof(uint64_t));
                node_hash_map = new Hash_map(mpi_config->particle_flow_rank, next_pow2, gpu_node_hash_map_keys, gpu_node_hash_map_values);
				mtracker->allocate_device("gpu_node_hash_map", (void**)&gpu_node_hash_map, sizeof(Hash_map));
                gpuErrchk( cudaMemcpy(gpu_node_hash_map, node_hash_map, sizeof(Hash_map), cudaMemcpyHostToDevice));

                
                if (global_node_to_local_node_map.size() != 0)
                {
                    int thread_count = min( (int) 32, (int)global_node_to_local_node_map.size());
                    int block_count  = max(1, (int) ceil((double) (global_node_to_local_node_map.size()) / (double) thread_count));

                    C_create_map(block_count, thread_count, gpu_node_hash_map, gpu_node_map_keys, gpu_node_map_values, global_node_to_local_node_map.size());
                    gpuErrchk( cudaPeekAtLastError() );
                }
                cudaStreamSynchronize(0);
                // exit(1);
                gpuErrchk( cudaPeekAtLastError() );

                free(full_node_map_keys);
				free(full_node_map_values);

                cudaFree(gpu_boundary_map_keys);
                cudaFree(gpu_boundary_map_values);

                // gpuErrchk( cudaFree(gpu_node_map_keys));
                // gpuErrchk( cudaFree(gpu_node_map_values));

				//cell centers to gpu
				// full_cell_centers = (vec<T> *) malloc(mesh->mesh_size * sizeof(vec<T>));
				// for(uint64_t i = 0; i < mesh->mesh_size; i++)
				// {
				// 	const uint64_t cell = i - mesh->shmem_cell_disp;
				// 	full_cell_centers[i] = mesh->cell_centers[cell];
				// 	//printf("the center of cell %lu is (%3.8f,%3.8f,%3.8f)\n",i,full_cell_centers[i].x,full_cell_centers[i].y,full_cell_centers[i].z);
				// } 


                uint64_t *local_halo_cells        = (uint64_t *) malloc(nhalos * mesh->cell_size * sizeof(uint64_t)); 
                vec<T>   *local_halo_cell_centers = (vec<T>   *) malloc(nhalos * sizeof(vec<T>)); 

                // Iterate over halos
                for ( auto cell_halo_pair : boundary_map )
                {
                    uint64_t cell = cell_halo_pair.first;
                    uint64_t halo = cell_halo_pair.second - mesh->local_mesh_size;

                    local_halo_cell_centers[halo] = mesh->cell_centers[cell-mesh->shmem_cell_disp];

                    #pragma ivdep
                    for (uint64_t n = 0; n < mesh->cell_size; n++)
                    {
                        local_halo_cells[halo*mesh->cell_size + n] = mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + n]; 
                    }
                }


				gpuErrchk( cudaMemcpy(gpu_face_centers, face_centers, mesh->faces_size * sizeof(vec<T>),
						   cudaMemcpyHostToDevice));
				

                gpuErrchk( cudaMemcpy(gpu_local_nodes, &mesh->points[-mesh->shmem_point_disp],
							mesh->points_size * sizeof(vec<T>),
							cudaMemcpyHostToDevice));

                gpuErrchk( cudaMemcpy(gpu_cells_per_point, &mesh->cells_per_point[-mesh->shmem_point_disp],
							mesh->points_size * sizeof(uint8_t),
							cudaMemcpyHostToDevice));

                // Copy local cells across
                gpuErrchk( cudaMemcpy(gpu_local_cells, &mesh->cells[(mesh->local_cells_disp - mesh->shmem_cell_disp) * mesh->cell_size],
							mesh->local_mesh_size * mesh->cell_size * sizeof(uint64_t),
							cudaMemcpyHostToDevice));

                gpuErrchk( cudaMemcpy(gpu_cell_centers, &mesh->cell_centers[mesh->local_cells_disp - mesh->shmem_cell_disp], 
						   mesh->local_mesh_size * sizeof(vec<T>),
						   cudaMemcpyHostToDevice));

                // Copy halo cells across
                gpuErrchk( cudaMemcpy(gpu_local_cells + mesh->local_mesh_size * mesh->cell_size, local_halo_cells,
							nhalos * mesh->cell_size * sizeof(uint64_t),
							cudaMemcpyHostToDevice));

                gpuErrchk( cudaMemcpy(gpu_cell_centers + mesh->local_mesh_size, local_halo_cell_centers, 
						   nhalos * sizeof(vec<T>),
						   cudaMemcpyHostToDevice));
                
                free(local_halo_cell_centers);
                free(local_halo_cells);

				// free(full_cell_centers);

                if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Done cell data.\n", mpi_config->particle_flow_rank);

				//Create config for AMGX
				AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
				AMGX_SAFE_CALL(AMGX_install_signal_handler());
				AMGX_SAFE_CALL(AMGX_config_create_from_file(&pressure_cfg, "/lustre/fsw/coreai_devtech_all/hwaugh/repos/minicombust_app/AMGX_Solvers/PCGF_CLASSICAL_V_JACOBI.json"));
				AMGX_SAFE_CALL(AMGX_config_add_parameters(&pressure_cfg, "exception_handling=1"));
				
				AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, "//lustre/fsw/coreai_devtech_all/hwaugh/repos/minicombust_app/AMGX_Solvers/PBICGSTAB_NOPREC.json"));
                AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));	
				
				AMGX_SAFE_CALL(AMGX_resources_create(&main_rsrc, cfg,
									  &mpi_config->particle_flow_world, 1, &lrank));
				AMGX_SAFE_CALL(AMGX_resources_create(&pressure_rsrc, pressure_cfg,
                                      &mpi_config->particle_flow_world, 1, &lrank));	
				
				//Create matrix and vectors for AMGX
				AMGX_SAFE_CALL(AMGX_matrix_create(&A, main_rsrc, mode));
				AMGX_SAFE_CALL(AMGX_vector_create(&u, main_rsrc, mode));
				AMGX_SAFE_CALL(AMGX_vector_create(&b, main_rsrc, mode));

				AMGX_SAFE_CALL(AMGX_matrix_create(&pressure_A, pressure_rsrc, mode));
                AMGX_SAFE_CALL(AMGX_vector_create(&pressure_u, pressure_rsrc, mode));
                AMGX_SAFE_CALL(AMGX_vector_create(&pressure_b, pressure_rsrc, mode));
				
				//Create solvers for AMGX
				AMGX_SAFE_CALL(AMGX_solver_create(&solver, main_rsrc, mode, cfg));
				AMGX_SAFE_CALL(AMGX_solver_create(&pressure_solver, pressure_rsrc, mode, cfg));


				AMGX_SAFE_CALL(AMGX_config_get_default_number_of_rings(cfg, &nrings));

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

                mtracker->print_usage();

                MPI_Barrier(mpi_config->world);

                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);

                size_t free, total;
	            cudaMemGetInfo( &free, &total );

                if (mpi_config->particle_flow_rank == 0)
                {
                    printf("GPU memory %lu free of %lu", free, total);
                }
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
                int *buffer           = (int *)      malloc(buff_size * sizeof(int));
                uint64_t *uint_buffer = (uint64_t *) malloc(buff_size * sizeof(uint64_t));
                for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
                {
                    int num_indexes;
                    MPI_Probe (halo_ranks[r], 0, mpi_config->particle_flow_world, &status );
                    MPI_Get_count( &status, MPI_UINT64_T, &num_indexes );

                    if ( num_indexes > buff_size )
                    {
                        buff_size = num_indexes;
                        buffer      = (int *)      realloc(buffer, (buff_size + 1) * sizeof(int));
                        uint_buffer = (uint64_t *) realloc(uint_buffer, (buff_size + 1) * sizeof(uint64_t));
                    }

                    MPI_Recv( uint_buffer, num_indexes, MPI_UINT64_T, halo_ranks[r], 0, mpi_config->particle_flow_world, MPI_STATUS_IGNORE );

                    for (int i = 0; i < num_indexes; i++)
                    {
                        buffer[i]      = (int)(uint_buffer[i] - mesh->local_cells_disp); 
                        uint_buffer[i] = uint_buffer[i] - mesh->local_cells_disp; 
                    }
                    uint64_t *tmp;
                    mtracker->allocate_device("gpu_halo_indexes", (void**)&tmp, sizeof(uint64_t) * num_indexes, (void*)0x7);
                    cudaMemcpy(tmp, uint_buffer, num_indexes * sizeof(uint64_t), cudaMemcpyHostToDevice);
                    gpu_halo_indexes.push_back(tmp);

                    phi_vector<T> phi_tmp;
                    phi_vector<vec<T>> phi_grad_tmp;

                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.U,         sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.V,         sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.W,         sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.P,         sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.PP,        sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.TE,        sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.ED,        sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.TP,        sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.TEM,       sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.FUL,       sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.PRO,       sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.VARF,      sizeof(T) * num_indexes, (void*)0x5);
                    mtracker->allocate_device("phi_halo_buffer", (void**)&phi_tmp.VARP,      sizeof(T) * num_indexes, (void*)0x5);

                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.U,    sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.V,    sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.W,    sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.P,    sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.PP,   sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.TE,   sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.ED,   sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.TEM,  sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.FUL,  sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.PRO,  sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.VARF, sizeof(vec<T>) * num_indexes, (void*)0x6);
                    mtracker->allocate_device("phi_grad_halo_buffer", (void**)&phi_grad_tmp.VARP, sizeof(vec<T>) * num_indexes, (void*)0x6);

                    gpu_phi_send_buffers.push_back(phi_tmp);
                    gpu_phi_grad_send_buffers.push_back(phi_grad_tmp);

                    // GPU
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
                free(uint_buffer);
            }


            void resize_cell_particle (uint64_t elements, uint64_t index)
            {
                while ( cell_index_array_size[index] < ((size_t) elements * sizeof(uint64_t)) )
                {
                    cell_index_array_size[index]    *= 2;
                    cell_particle_array_size[index] *= 2;

                    neighbour_indexes[index] = (uint64_t *)       realloc(neighbour_indexes[index],  cell_index_array_size[index]);
                    cell_particle_aos[index] = (particle_aos<T> *)realloc(cell_particle_aos[index],  cell_particle_array_size[index]);

                    cudaFree(gpu_neighbour_indexes[index]);
                    cudaFree(gpu_cell_particle_aos[index]);
                    cudaMalloc(&gpu_neighbour_indexes[index], cell_index_array_size[index]);
                    cudaMalloc(&gpu_cell_particle_aos[index], cell_particle_array_size[index]);
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
				AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
				AMGX_SAFE_CALL(AMGX_solver_destroy(pressure_solver));

				AMGX_SAFE_CALL(AMGX_vector_destroy(u));
				AMGX_SAFE_CALL(AMGX_vector_destroy(b));
				AMGX_SAFE_CALL(AMGX_vector_destroy(pressure_b));
				AMGX_SAFE_CALL(AMGX_vector_destroy(pressure_u));

				AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
				AMGX_SAFE_CALL(AMGX_matrix_destroy(pressure_A));

				AMGX_SAFE_CALL(AMGX_resources_destroy(main_rsrc));
				AMGX_SAFE_CALL(AMGX_resources_destroy(pressure_rsrc));

				AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
				AMGX_SAFE_CALL(AMGX_config_destroy(pressure_cfg));
			}
			
            void resize_send_buffers_nodes_arrays (uint64_t elements)
            {
                while ( send_buffers_node_index_array_size < ((size_t) elements * sizeof(uint64_t)) )
                {
                    printf("Flow rank resizing send bufffers for %lu elements\n", elements);
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
