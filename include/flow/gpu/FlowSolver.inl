#include <stdio.h>
#include <limits.h>

#include "flow/gpu/FlowSolver.hpp"

#include <nvToolsExt.h>

#define CUDA_SYNC_DEBUG 0

using namespace std;

namespace minicombust::flow 
{

	template<class T>void FlowSolver<T>::output_data(uint64_t timestep)
    	{
		VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh, mpi_config);
		vtk_writer->write_flow("out/flow/minicombust", timestep, &phi);
        // 	vtk_writer->write_flow_velocities("out/minicombust", timestep, &phi);
		// vtk_writer->write_flow_pressure("out/minicombust", timestep, &phi);
    	}
	
	template<typename T> inline bool FlowSolver<T>::is_halo ( uint64_t cell )
    	{
		/*Return true if a cell is part of the halo between processes*/
        	return ( cell - mesh->local_cells_disp >= mesh->local_mesh_size );
    	}

	template<typename T> void FlowSolver<T>::exchange_cell_info_halos ()
	{
		/*Exchange constant cell values over halos*/
        	int num_requests = 2;

        	MPI_Request send_requests[halo_ranks.size() * num_requests];
        	MPI_Request recv_requests[halo_ranks.size() * num_requests];
        	for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        	{
            		MPI_Isend( cell_densities,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            		MPI_Isend( cell_volumes,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
        	}

        	for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        	{
            		MPI_Irecv( &cell_densities[mesh->local_mesh_size + halo_disps[r]],  halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            		MPI_Irecv( &cell_volumes[mesh->local_mesh_size + halo_disps[r]],    halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 1] );
        	}

		    MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        	MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    	}
	
	template<typename T> void FlowSolver<T>::exchange_single_phi_halo(T *phi_component)
	{
		nvtxRangePush(__FUNCTION__);

		/*Pass a single phi value over the halos*/
		int num_requests = 1;
		
		MPI_Request send_requests[halo_ranks.size() * num_requests];
		MPI_Request recv_requests[halo_ranks.size() * num_requests];

		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
            MPI_Irecv( &gpu_phi.PP[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
		}
	
		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			int bytes_size;
			MPI_Type_size(halo_mpi_double_datatypes[r], &bytes_size);

			int elements = halo_sizes[r];

			int thread_count = min( (int) 256, elements);
			int block_count = max(1, (int) ceil((double) (elements) / (double) thread_count));

			C_kernel_pack_PP_halo_buffer(block_count, thread_count, gpu_phi_send_buffers[r], gpu_phi, gpu_halo_indexes[r], (uint64_t)(elements));
		gpuErrchk( cudaPeekAtLastError() );

            MPI_Isend( gpu_phi_send_buffers[r].PP,    halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 3,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 0] );
		}
		
		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
		MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);	
		nvtxRangePop();
		
	}

	template<typename T> void FlowSolver<T>::exchange_single_grad_halo(vec<T> *phi_grad_component)
    	{
			nvtxRangePush(__FUNCTION__);

			/*Pass a single gradient vector over the halos*/
			int num_requests = 1;

			MPI_Request send_requests[halo_ranks.size() * num_requests];
			MPI_Request recv_requests[halo_ranks.size() * num_requests];

			for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
			{
				int bytes_size;
				MPI_Type_size(halo_mpi_vec_double_datatypes[r], &bytes_size);

				int elements = halo_sizes[r];

				int thread_count = min((int) 256, elements);
				int block_count  = max(1, (int) ceil((double) (elements) / (double) thread_count));
				C_kernel_pack_PP_grad_halo_buffer(block_count, thread_count, gpu_phi_grad_send_buffers[r], gpu_phi_grad, gpu_halo_indexes[r], (uint64_t)(elements));
				gpuErrchk( cudaPeekAtLastError() );

			}	

			for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
			{
				MPI_Irecv( &gpu_phi_grad.PP[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
			}

			gpuErrchk( cudaPeekAtLastError() );
			gpuErrchk(cudaStreamSynchronize(0));

			for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
			{
				MPI_Isend( gpu_phi_grad_send_buffers[r].PP,    3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 3,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 0] );
			}	


			MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
			MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
			nvtxRangePop();

    	}

	template<typename T> void FlowSolver<T>::exchange_grad_halos()
	{
		/*Group exchange of most phi gradient vectors over the halos*/
		int num_requests = 11;
		
		MPI_Request send_requests[halo_ranks.size() * num_requests];
		MPI_Request recv_requests[halo_ranks.size() * num_requests];

		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			int elements = halo_sizes[r];

			int thread_count = min((int) 256, elements);
			int block_count  = max(1, (int) ceil((double) (elements) / (double) thread_count));
			C_kernel_pack_phi_grad_halo_buffer(block_count, thread_count, gpu_phi_grad_send_buffers[r], gpu_phi_grad, gpu_halo_indexes[r], (uint64_t)(elements));
			gpuErrchk( cudaPeekAtLastError() );

		}
        
		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			MPI_Irecv( &gpu_phi_grad.U[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
			MPI_Irecv( &gpu_phi_grad.V[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 1] );
			MPI_Irecv( &gpu_phi_grad.W[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 2, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 2] );
			MPI_Irecv( &gpu_phi_grad.P[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 3] );
			MPI_Irecv( &gpu_phi_grad.TE[mesh->local_mesh_size + halo_disps[r]],  3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 4, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 4] );
			MPI_Irecv( &gpu_phi_grad.ED[mesh->local_mesh_size + halo_disps[r]],  3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 5, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 5] );
			MPI_Irecv( &gpu_phi_grad.TEM[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 6, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 6] );
			MPI_Irecv( &gpu_phi_grad.FUL[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 7, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 7] );
			MPI_Irecv( &gpu_phi_grad.PRO[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 8, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 8] );
			MPI_Irecv( &gpu_phi_grad.VARF[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 9, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 9] );
			MPI_Irecv( &gpu_phi_grad.VARP[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 10, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 10] );
		}

		cudaStreamSynchronize(0);
		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{

			// MPI_Isend( gpu_phi_grad.U,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
			// MPI_Isend( gpu_phi_grad.V,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
			// MPI_Isend( gpu_phi_grad.W,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 2, mpi_config->particle_flow_world, &send_requests[num_requests*r + 2] );
			// MPI_Isend( gpu_phi_grad.P,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 3, mpi_config->particle_flow_world, &send_requests[num_requests*r + 3] );
			// MPI_Isend( gpu_phi_grad.TE,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 4, mpi_config->particle_flow_world, &send_requests[num_requests*r + 4] );
			// MPI_Isend( gpu_phi_grad.ED,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 5, mpi_config->particle_flow_world, &send_requests[num_requests*r + 5] );
			// MPI_Isend( gpu_phi_grad.TEM, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 6, mpi_config->particle_flow_world, &send_requests[num_requests*r + 6] );
			// MPI_Isend( gpu_phi_grad.FUL, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 7, mpi_config->particle_flow_world, &send_requests[num_requests*r + 7] );
			// MPI_Isend( gpu_phi_grad.PRO, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 8, mpi_config->particle_flow_world, &send_requests[num_requests*r + 8] );
			// MPI_Isend( gpu_phi_grad.VARF, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 9, mpi_config->particle_flow_world, &send_requests[num_requests*r + 9] );
			// MPI_Isend( gpu_phi_grad.VARP, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 10, mpi_config->particle_flow_world, &send_requests[num_requests*r + 10] );	

			MPI_Isend( gpu_phi_grad_send_buffers[r].U,    3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 0,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 0] );
            MPI_Isend( gpu_phi_grad_send_buffers[r].V,    3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 1,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 1] );
            MPI_Isend( gpu_phi_grad_send_buffers[r].W,    3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 2,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 2] );
            MPI_Isend( gpu_phi_grad_send_buffers[r].P,    3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 3,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 3] );
			MPI_Isend( gpu_phi_grad_send_buffers[r].TE,   3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 4,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 4] );	
			MPI_Isend( gpu_phi_grad_send_buffers[r].ED,   3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 5,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 5] );
			MPI_Isend( gpu_phi_grad_send_buffers[r].TEM,  3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 6,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 6] );
			MPI_Isend( gpu_phi_grad_send_buffers[r].FUL,  3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 7,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 7] ); 
			MPI_Isend( gpu_phi_grad_send_buffers[r].PRO,  3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 8,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 8] );
			MPI_Isend( gpu_phi_grad_send_buffers[r].VARF, 3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 9,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 9] );
			MPI_Isend( gpu_phi_grad_send_buffers[r].VARP, 3*halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 10,  mpi_config->particle_flow_world,  &send_requests[num_requests*r + 10] );

			// MPI_Isend( gpu_phi_grad.U,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
			// MPI_Isend( gpu_phi_grad.V,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
			// MPI_Isend( gpu_phi_grad.W,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 2, mpi_config->particle_flow_world, &send_requests[num_requests*r + 2] );
			// MPI_Isend( gpu_phi_grad.P,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 3, mpi_config->particle_flow_world, &send_requests[num_requests*r + 3] );
			// MPI_Isend( gpu_phi_grad.TE,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 4, mpi_config->particle_flow_world, &send_requests[num_requests*r + 4] );
			// MPI_Isend( gpu_phi_grad.ED,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 5, mpi_config->particle_flow_world, &send_requests[num_requests*r + 5] );
			// MPI_Isend( gpu_phi_grad.TEM, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 6, mpi_config->particle_flow_world, &send_requests[num_requests*r + 6] );
			// MPI_Isend( gpu_phi_grad.FUL, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 7, mpi_config->particle_flow_world, &send_requests[num_requests*r + 7] );
			// MPI_Isend( gpu_phi_grad.PRO, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 8, mpi_config->particle_flow_world, &send_requests[num_requests*r + 8] );
			// MPI_Isend( gpu_phi_grad.VARF, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 9, mpi_config->particle_flow_world, &send_requests[num_requests*r + 9] );
			// MPI_Isend( gpu_phi_grad.VARP, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 10, mpi_config->particle_flow_world, &send_requests[num_requests*r + 10] );	
		}
		int thread_count1 = min((uint64_t) 32, mesh->points_size);
		int block_count1 = max(1,(int) ceil((double) mesh->points_size/(double) thread_count1));
		C_kernel_interpolate_init_boundaries(block_count1, thread_count1, gpu_phi_nodes, gpu_cells_per_point, global_node_to_local_node_map.size());
		gpuErrchk( cudaPeekAtLastError() );

		// MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
		MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
	}

	template<typename T> void FlowSolver<T>::exchange_phi_halos_cpu ()
	{
		/*Group exchange of most phi values over the halos*/
		int num_requests = 11;
		MPI_Request send_requests[halo_ranks.size() * num_requests];
		MPI_Request recv_requests[halo_ranks.size() * num_requests];
		
		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			MPI_Isend( phi.U,         1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 0] );
			MPI_Isend( phi.V,         1, halo_mpi_double_datatypes[r],     halo_ranks[r], 1, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 1] );
			MPI_Isend( phi.W,         1, halo_mpi_double_datatypes[r],     halo_ranks[r], 2, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 2] );
			MPI_Isend( phi.P,         1, halo_mpi_double_datatypes[r],     halo_ranks[r], 3, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 3] );
			MPI_Isend( phi.TE,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 4, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 4] );
			MPI_Isend( phi.ED,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 5, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 5] );
			MPI_Isend( phi.TEM,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 6, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 6] );
			MPI_Isend( phi.FUL,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 7, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 7] );
			MPI_Isend( phi.PRO,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 8, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 8] );
			MPI_Isend( phi.VARF,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 9, mpi_config->particle_flow_world,  &send_requests[num_requests*r + 9] );
			MPI_Isend( phi.VARP,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 10, mpi_config->particle_flow_world, &send_requests[num_requests*r + 10] );
		}

        	for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        	{
			MPI_Irecv( &phi.U[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 0] );
			MPI_Irecv( &phi.V[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 1] );
			MPI_Irecv( &phi.W[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 2, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 2] );
			MPI_Irecv( &phi.P[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 3] );
			MPI_Irecv( &phi.TE[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 4, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 4] );
			MPI_Irecv( &phi.ED[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 5, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 5] );
			MPI_Irecv( &phi.TEM[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 6, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 6] );
			MPI_Irecv( &phi.FUL[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 7, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 7] );
			MPI_Irecv( &phi.PRO[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 8, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 8] );
			MPI_Irecv( &phi.VARF[mesh->local_mesh_size + halo_disps[r]],     halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 9, mpi_config->particle_flow_world,  &recv_requests[num_requests*r + 9] );
			MPI_Irecv( &phi.VARP[mesh->local_mesh_size + halo_disps[r]],     halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 10, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 10] );
		}

        	MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        	MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    	}

    template<typename T> void FlowSolver<T>::exchange_phi_halos ()
    {
		/*Group exchange of most phi values over the halos*/
        int num_requests = 11;
        
		MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];

		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			int elements = halo_sizes[r];

			int thread_count = min( (int) 256, elements);
			int block_count = max(1, (int) ceil((double) (elements) / (double) thread_count));

			C_kernel_pack_phi_halo_buffer(block_count, thread_count, gpu_phi_send_buffers[r], gpu_phi, gpu_halo_indexes[r], (uint64_t)(elements));
			gpuErrchk( cudaPeekAtLastError() );
		}

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &gpu_phi.U[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            MPI_Irecv( &gpu_phi.V[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 1] );
            MPI_Irecv( &gpu_phi.W[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 2,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 2] );
            MPI_Irecv( &gpu_phi.P[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 3] );
			MPI_Irecv( &gpu_phi.TE[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 4,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 4] );
			MPI_Irecv( &gpu_phi.ED[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 5,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 5] );
			MPI_Irecv( &gpu_phi.TEM[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 6,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 6] );
			MPI_Irecv( &gpu_phi.FUL[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 7,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 7] );
			MPI_Irecv( &gpu_phi.PRO[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 8,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 8] );
			MPI_Irecv( &gpu_phi.VARF[mesh->local_mesh_size + halo_disps[r]],     halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 9,  mpi_config->particle_flow_world, &recv_requests[num_requests*r + 9] );
			MPI_Irecv( &gpu_phi.VARP[mesh->local_mesh_size + halo_disps[r]],     halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 10, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 10] );
		}

		cudaStreamSynchronize(0);
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            // MPI_Isend( gpu_phi.U,         1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            // MPI_Isend( gpu_phi.V,         1, halo_mpi_double_datatypes[r],     halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
            // MPI_Isend( gpu_phi.W,         1, halo_mpi_double_datatypes[r],     halo_ranks[r], 2, mpi_config->particle_flow_world, &send_requests[num_requests*r + 2] );
            // MPI_Isend( gpu_phi.P,         1, halo_mpi_double_datatypes[r],     halo_ranks[r], 3, mpi_config->particle_flow_world, &send_requests[num_requests*r + 3] );
			// MPI_Isend( gpu_phi.TE,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 4, mpi_config->particle_flow_world, &send_requests[num_requests*r + 4] );	
			// MPI_Isend( gpu_phi.ED,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 5, mpi_config->particle_flow_world, &send_requests[num_requests*r + 5] );
			// MPI_Isend( gpu_phi.TEM,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 6, mpi_config->particle_flow_world, &send_requests[num_requests*r + 6] );
			// MPI_Isend( gpu_phi.FUL,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 7, mpi_config->particle_flow_world, &send_requests[num_requests*r + 7] ); 
			// MPI_Isend( gpu_phi.PRO,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 8, mpi_config->particle_flow_world, &send_requests[num_requests*r + 8] );
			// MPI_Isend( gpu_phi.VARF,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 9, mpi_config->particle_flow_world, &send_requests[num_requests*r + 9] );
			// MPI_Isend( gpu_phi.VARP,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 10, mpi_config->particle_flow_world, &send_requests[num_requests*r + 10] );

			MPI_Isend( gpu_phi_send_buffers[r].U,    halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 0,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 0] );
            MPI_Isend( gpu_phi_send_buffers[r].V,    halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 1,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 1] );
            MPI_Isend( gpu_phi_send_buffers[r].W,    halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 2,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 2] );
            MPI_Isend( gpu_phi_send_buffers[r].P,    halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 3,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 3] );
			MPI_Isend( gpu_phi_send_buffers[r].TE,   halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 4,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 4] );	
			MPI_Isend( gpu_phi_send_buffers[r].ED,   halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 5,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 5] );
			MPI_Isend( gpu_phi_send_buffers[r].TEM,  halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 6,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 6] );
			MPI_Isend( gpu_phi_send_buffers[r].FUL,  halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 7,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 7] ); 
			MPI_Isend( gpu_phi_send_buffers[r].PRO,  halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 8,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 8] );
			MPI_Isend( gpu_phi_send_buffers[r].VARF, halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 9,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 9] );
			MPI_Isend( gpu_phi_send_buffers[r].VARP, halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 10,  mpi_config->particle_flow_world,  &send_requests[num_requests*r + 10] );
		}
		
		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::exchange_A_halos (T *A_phi_component)
    {
		nvtxRangePush(__FUNCTION__);

		/*Exchange a single A_phi value over the halos
		  Used to make sure 1/A values are consistent over processes*/
        int num_requests = 1;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &A_phi_component[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
        }

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
			int bytes_size;
			MPI_Type_size(halo_mpi_double_datatypes[r], &bytes_size);

			int elements = halo_sizes[r];

			int thread_count = min( (int) 256, elements);
			int block_count = max(1, (int) ceil((double) (elements) / (double) thread_count));

			C_kernel_pack_Aphi_halo_buffer(block_count, thread_count, gpu_phi_send_buffers[r], gpu_A_phi, gpu_halo_indexes[r], (uint64_t)(elements));
			gpuErrchk( cudaPeekAtLastError() );

			MPI_Isend( gpu_phi_send_buffers[r].U,    halo_sizes[r], MPI_DOUBLE,  halo_ranks[r], 0,   mpi_config->particle_flow_world,  &send_requests[num_requests*r + 0] );
        }

		
		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
		nvtxRangePop();

    }

    template<typename T> void FlowSolver<T>::update_flow_field()
    {
		nvtxRangePush("Waiting time: update_flow_field");

		//TODO: we should collect the correct temp values and maybe the grad values.
		/*Collect and send cell values to particle solve and receive
		  values from particle solve.*/
        int time_count = 0;
        time_stats[time_count]  -= MPI_Wtime(); //0
        unordered_neighbours_set[0].clear();
        node_to_position_map.clear();
        new_cells_set.clear();
        ranks.clear();
		send_buffer_disp = 0;
		// memset(mesh->particle_terms,         0, sizeof(particle_aos<T>)*mesh->local_mesh_size);
		cudaMemsetAsync(gpu_particle_terms, 0, sizeof(particle_aos<T>)*mesh->local_mesh_size, process_gpu_fields_stream);
		gpuErrchk(cudaMemsetAsync(gpu_seen_node,           0, global_node_to_local_node_map.size() * sizeof(int) , (cudaStream_t) 0 ));

        for (uint64_t i = 0; i < local_particle_node_sets.size(); i++)
        {
            local_particle_node_sets[i].clear();
        }
        
        performance_logger.my_papi_start();

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //1
        static double time0=0., time1=0., time2=0.;
        static double recv_time1=0., recv_time2=0., recv_time3=0.;

        int recvs_complete = 0;
        MPI_Ibcast(&recvs_complete, 1, MPI_INT, mpi_config->particle_flow_world_size, mpi_config->world, &bcast_request);       
 
        int message_waiting = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);

		bool first_msg_recv        = false;
        bool  all_processed        = false;
        bool *processed_neighbours = async_locks;
		uint64_t node_element_disp = 0;
		uint64_t cell_element_disp = 0;
		gpuErrchk(cudaMemsetAsync(gpu_node_buffer_disp,    0, sizeof(uint64_t), (cudaStream_t) 0 ));

		// for (uint64_t rank_slot = 0; rank_slot < 13; rank_slot++)
		// 	node_starts[rank_slot] = -1;

		// MPI_Barrier(mpi_config->world);

        while(!all_processed)
        {
            time0 -= MPI_Wtime(); //1
            if ( message_waiting )
            {

				if (!first_msg_recv)
				{
					nvtxRangePop();
					nvtxRangePush("nowait_update_flow_field ");
					nvtxRangePush("update_flow::loop1_recv_get_neighbours");
					first_msg_recv = true;
				}

                uint64_t rank_slot = ranks.size();
                ranks.push_back(statuses[rank_slot].MPI_SOURCE);
                MPI_Get_count( &statuses[rank_slot], MPI_UINT64_T, &elements[rank_slot] );

                logger.recieved_cells += elements[rank_slot];

				cell_starts[rank_slot] = cell_element_disp;
				cell_element_disp += elements[rank_slot];


				if (cell_element_disp > gpu_recv_buffer_elements)
				{
					fprintf(output_file, "Cell array recv buffer overflow disp %lu max %d\n", cell_element_disp, gpu_recv_buffer_elements);
					exit(1);
				}

                MPI_Irecv(recv_buffers_cell_indexes         + cell_starts[rank_slot], elements[rank_slot], MPI_UINT64_T,                       ranks[rank_slot], 0, mpi_config->world, &recv_requests[2*rank_slot]     );
                MPI_Irecv(recv_buffers_cell_particle_fields + cell_starts[rank_slot], elements[rank_slot], mpi_config->MPI_PARTICLE_STRUCTURE, ranks[rank_slot], 2, mpi_config->world, &recv_requests[2*rank_slot + 1] );
				
                processed_neighbours[rank_slot] = false;  // Invalid write

                if ( statuses.size() <= ranks.size() )
                {
                    statuses.push_back (empty_mpi_status);
                    recv_requests.push_back ( MPI_REQUEST_NULL );
                    recv_requests.push_back ( MPI_REQUEST_NULL );
                    send_requests.push_back ( MPI_REQUEST_NULL );
                    send_requests.push_back ( MPI_REQUEST_NULL );

                    cell_index_array_size.push_back(max_storage    * sizeof(uint64_t));
                    cell_particle_array_size.push_back(max_storage * sizeof(particle_aos<T>));

                    local_particle_node_sets.push_back(unordered_set<uint64_t>());
                }


                message_waiting = 0;
				
				processed_neighbours[rank_slot] = false;

                MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);
                continue;
			}

			all_processed = true;
			for ( uint64_t rank_slot = 0; rank_slot < ranks.size(); rank_slot++ )
            {
				int recieved_particle_indexes = 0;
                if (!processed_neighbours[rank_slot])  MPI_Test(&recv_requests[2*rank_slot], &recieved_particle_indexes, MPI_STATUS_IGNORE);

				if ( !processed_neighbours[rank_slot] && recieved_particle_indexes )
				{
					gpuErrchk(cudaMemcpyAsync(gpu_recv_buffers_cell_indexes + cell_starts[rank_slot], recv_buffers_cell_indexes + cell_starts[rank_slot], elements[rank_slot]*sizeof(uint64_t), cudaMemcpyHostToDevice, (cudaStream_t) 0 ));

					gpuErrchk(cudaMemcpyAsync(&node_starts[rank_slot], gpu_node_buffer_disp, sizeof(uint64_t), cudaMemcpyDeviceToHost, (cudaStream_t) 0 ));

					int thread_count = min( (int) 32, (int) elements[rank_slot]);
					int block_count = max(1, (int) ceil((double) (elements[rank_slot]) / (double) thread_count));
					
					C_kernel_get_node_buffers(block_count, thread_count, gpu_recv_buffers_cell_indexes + cell_starts[rank_slot], gpu_local_cells, gpu_seen_node, gpu_phi_nodes, gpu_send_buffers_interp_node_indexes, gpu_send_buffers_interp_node_flow_fields, gpu_node_hash_map, elements[rank_slot], global_node_to_local_node_map.size(), mesh->local_mesh_size, mesh->local_cells_disp, gpu_atomic_buffer_index, gpu_node_buffer_disp);
					gpuErrchk( cudaPeekAtLastError() );

					gpuErrchk(cudaMemcpyAsync(&node_elements[rank_slot], gpu_atomic_buffer_index, sizeof(uint32_t), cudaMemcpyDeviceToHost, (cudaStream_t) 0 ));
					gpuErrchk(cudaMemsetAsync(gpu_seen_node, 0, global_node_to_local_node_map.size() * sizeof(int) , (cudaStream_t) 0 ));

					processed_neighbours[rank_slot] = true;
				}
				
				all_processed &= processed_neighbours[rank_slot];
			}

            MPI_Test ( &bcast_request, &recvs_complete, MPI_STATUS_IGNORE );
            MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);

            if ( FLOW_SOLVER_DEBUG && recvs_complete ) if(recvs_complete) fprintf(output_file, "\tFlow block %d: Recieved broadcast signal. message_waiting %d recvs_complete %d all_processed %d\n", mpi_config->particle_flow_rank, message_waiting, recvs_complete, all_processed);
            all_processed &= !message_waiting & recvs_complete;
        }

		if (first_msg_recv)
		{
			nvtxRangePop();
		}

		// Send size of reduced neighbours of cells back to ranks.
        uint64_t neighbour_point_size = node_to_position_map.size();

        logger.sent_nodes += neighbour_point_size;

		nvtxRangePush("update_flow::pack_and_post_buffers");
		cudaStreamSynchronize(0);

		uint64_t ptr_disp = 0;
        bool *processed_cell_fields = async_locks;
        for (uint64_t p = 0; p < ranks.size(); p++)
        {
            uint64_t local_disp = 0;

			MPI_Isend ( &gpu_send_buffers_interp_node_indexes[node_starts[p]],     node_elements[p], MPI_UINT64_T,                   ranks[p], 0, mpi_config->world, &send_requests[p] );
            MPI_Isend ( &gpu_send_buffers_interp_node_flow_fields[node_starts[p]], node_elements[p], mpi_config->MPI_FLOW_STRUCTURE, ranks[p], 1, mpi_config->world, &send_requests[p + ranks.size()] );

			ptr_disp += node_elements[p];
			node_element_disp += node_elements[p];

            processed_cell_fields[p] = false; // Invalid write

            recv_time2  += MPI_Wtime();
        }

		
		if ( logger.min_cells_buf_size_per_timestep == 0 )
			logger.min_cells_buf_size_per_timestep = cell_element_disp;

		if ( logger.min_cells_buf_size_per_timestep == 0 )
			logger.min_nodes_buf_size_per_timestep = node_element_disp;

		logger.min_cells_buf_size_per_timestep = min(logger.min_cells_buf_size_per_timestep, (double)cell_element_disp); 
		logger.min_nodes_buf_size_per_timestep = min(logger.min_nodes_buf_size_per_timestep, (double)node_element_disp); 
		logger.max_cells_buf_size_per_timestep = max(logger.max_cells_buf_size_per_timestep, (double)cell_element_disp);
		logger.max_nodes_buf_size_per_timestep = max(logger.max_nodes_buf_size_per_timestep, (double)node_element_disp);

		if (mpi_config->particle_flow_rank == 0)
			fprintf(output_file, "Cell buffer used %lu (of %d) Node buffer used %lu (of %d) \n", cell_element_disp, gpu_send_buffer_elements, node_element_disp, gpu_recv_buffer_elements);


		nvtxRangePop();
        
        recv_time3  -= MPI_Wtime();


		MPI_Waitall(2*ranks.size(), recv_requests.data(), MPI_STATUSES_IGNORE);

		gpuErrchk(cudaMemcpyAsync(gpu_recv_buffers_cell_particle_fields, recv_buffers_cell_particle_fields, cell_element_disp*sizeof(particle_aos<T>), cudaMemcpyHostToDevice, process_gpu_fields_stream));

		int thread_count = min(32, max(1, (int)cell_element_disp));
		int block_count = max(1,(int) ceil((double) cell_element_disp / (double) thread_count));

		C_kernel_process_particle_fields(block_count, thread_count, gpu_recv_buffers_cell_indexes, gpu_recv_buffers_cell_particle_fields, gpu_particle_terms, cell_element_disp, mesh->local_cells_disp, process_gpu_fields_stream);
		gpuErrchk( cudaPeekAtLastError() );


        MPI_Barrier(mpi_config->particle_flow_world);

		nvtxRangePop();
        recv_time3 += MPI_Wtime();

        if ( FLOW_SOLVER_DEBUG )  fprintf(output_file, "\tFlow Rank %d: Processed cell particle fields .\n", mpi_config->rank);

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
        
        time_stats[time_count++] += MPI_Wtime();

        if (((timestep_count + 1) % TIMER_OUTPUT_INTERVAL == 0) 
				&& FLOW_SOLVER_FINE_TIME)
        {
            if ( mpi_config->particle_flow_rank == 0 )
            {
                for (int i = 0; i < time_count; i++)
                    MPI_Reduce(MPI_IN_PLACE, &time_stats[i], 1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);

                double total_time = 0.0;
                fprintf(output_file, "\nUpdate Flow Field Communuication Timings\n");

                for (int i = 0; i < time_count; i++)
                    total_time += time_stats[i];
                for (int i = 0; i < time_count; i++)
                    fprintf(output_file, "Time stats %d: %.3f (%.2f %%)\n", i, time_stats[i]  / mpi_config->particle_flow_world_size, 100 * time_stats[i] / total_time);
                fprintf(output_file, "Total time %f\n", total_time / mpi_config->particle_flow_world_size);

            }
            else{
                for (int i = 0; i < time_count; i++)
                    MPI_Reduce(&time_stats[i], nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);
            }
			for(int i = 0; i < time_count; i++)
			{
				time_stats[i] = 0.0;
			}
        }
    }

	template<typename T> void FlowSolver<T>::get_phi_gradients()
	{
		int thread_count = min((uint64_t) 32, mesh->local_mesh_size);
		int block_count = max(1,(int) ceil((double) mesh->local_mesh_size/ (double) thread_count));

		//Ensure memory is set to zero
        /*gpuErrchk(cudaMemset2D(full_data_A, A_pitch, 0, 9 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bU, bU_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bV, bV_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bW, bW_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bP, bP_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bTE, bTE_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bED, bED_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bT, bT_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bFU, bFU_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bPR, bPR_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bVFU, bVFU_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));
        gpuErrchk(cudaMemset2D(full_data_bVPR, bVPR_pitch, 0, 3 * sizeof(T), mesh->local_mesh_size));*/

		/*gpuErrchk(cudaMemset(full_data_A, 0.0, 9 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bU, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bV, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bW, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bP, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bTE, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bED, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bT, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bFU, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bPR, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bVFU, 0.0, 3 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bVPR, 0.0, 3 * sizeof(T)));*/



		//generate all the data arrays
		C_kernel_get_phi_gradients(block_count, thread_count, gpu_phi, gpu_phi_grad, mesh->local_mesh_size, mesh->local_cells_disp, mesh->faces_per_cell, (gpu_Face<uint64_t> *) gpu_faces, gpu_cell_faces, gpu_cell_centers, mesh->mesh_size, gpu_boundary_hash_map, gpu_boundary_map_values, boundary_map.size(), gpu_face_centers, nhalos);
		gpuErrchk( cudaPeekAtLastError() );
		//Wait for arrays to fill.
		// gpuErrchk(cudaDeviceSynchronize());
	}

	template<typename T> void FlowSolver<T>::calculate_UVW()
	{
		gpuErrchk(cudaMemset(gpu_A_phi.U, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_A_phi.V, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_A_phi.W, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		
		gpuErrchk(cudaMemset(gpu_S_phi.U, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_S_phi.V, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_S_phi.W, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));

		cudaStreamSynchronize(process_gpu_fields_stream);
		
		int thread_count = min((uint64_t) 32,mesh->faces_size);
		int block_count = max(1,(int) ceil((double) mesh->faces_size/ (double) thread_count));
		
		C_kernel_calculate_flux_UVW(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, mesh->local_mesh_size, boundary_map.size(), gpu_boundary_hash_map, gpu_boundary_map_values, gpu_phi_grad, gpu_cell_centers, gpu_face_centers, gpu_phi, gpu_A_phi, gpu_face_mass_fluxes, gpu_face_lambdas, gpu_face_normals, (gpu_Face<T> *) gpu_face_fields, gpu_S_phi, nhalos, gpu_boundary_types, mesh->dummy_gas_vel, effective_viscosity, gpu_face_rlencos, inlet_effective_viscosity, gpu_face_areas);
		gpuErrchk( cudaPeekAtLastError() );

		//sort out the forces
		thread_count = min((uint64_t) 32,mesh->local_mesh_size);
		block_count = max(1,(int) ceil((double) mesh->local_mesh_size/(double) thread_count));
		C_kernel_apply_forces(block_count, thread_count, mesh->local_mesh_size, gpu_cell_densities, gpu_cell_volumes, gpu_phi, gpu_S_phi, gpu_phi_grad, delta, gpu_A_phi, gpu_particle_terms);
		gpuErrchk( cudaPeekAtLastError() );
		
		//define variables for
		const double UVW_URFactor = 0.5;
		gpuErrchk(cudaMemset(nnz, 0, sizeof(int)));
		gpuErrchk(cudaMemset(values, 0.0, sizeof(T) * (mesh->local_mesh_size*7)));
		gpuErrchk(cudaMemset(col_indices, 0, sizeof(int64_t) * (mesh->local_mesh_size*7)));
		gpuErrchk(cudaMemset(rows_ptr, 0, sizeof(int) * (mesh->local_mesh_size+1)));

		//Solve for U
		C_kernel_setup_sparse_matrix(block_count, thread_count, UVW_URFactor, mesh->local_mesh_size, rows_ptr, col_indices, mesh->local_cells_disp, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_hash_map, gpu_boundary_map_values, gpu_A_phi.U, (gpu_Face<T> *) gpu_face_fields, values, gpu_S_phi.U, gpu_phi.U, mesh->mesh_size, mesh->faces_per_cell, gpu_cell_faces, nnz);
		gpuErrchk( cudaPeekAtLastError() );
		
		//wait for matrix values
		int cpu_nnz = mesh->local_mesh_size*(mesh->faces_per_cell+1);
		gpuErrchk(cudaMemcpy(nnz, &cpu_nnz, sizeof(int),
		                  cudaMemcpyHostToDevice));


		// for(int i = 0; i < mpi_config->particle_flow_world_size; i++){
		// 	if(mpi_config->particle_flow_rank == i){
		// 		//C_kernel_test_values(nnz, values, rows_ptr, col_indices, mesh->local_mesh_size, gpu_S_phi.U, gpu_phi.U);
		// 	}
		// 	MPI_Barrier (mpi_config->particle_flow_world);
		// }
		// MPI_Barrier (mpi_config->particle_flow_world);
		//int cpu_nnz = 0;
		//gpuErrchk(cudaMemcpy(nnz, &cpu_nnz, sizeof(int),cudaMemcpyHostToDevice));
		//C_kernel_test_values(nnz, values, rows_ptr, col_indices, mesh->local_mesh_size, gpu_S_phi.U, gpu_phi.U);
		//gpuErrchk(cudaMemcpy(&cpu_nnz, nnz, sizeof(int),
		//			cudaMemcpyDeviceToHost));
	        //static bool first_mat=true;	
		if(first_mat)
		{
			//find partition vector
			uint64_t *row_sizes = (uint64_t*)malloc(mpi_config->particle_flow_world_size * sizeof(uint64_t));
		
			MPI_Allgather(&(mesh->local_mesh_size), 1, MPI_UINT64_T, row_sizes, 1, MPI_UINT64_T, mpi_config->particle_flow_world);
			
			int count = 0;
			for(int i = 0; i < mpi_config->particle_flow_world_size; i++)
			{
				for(int j = 0; j < row_sizes[i]; j++)
        		{
					partition_vector[count] = i;
					count++;
				}
        	}

			first_mat = false;
			AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(A, mesh->mesh_size, mesh->local_mesh_size, cpu_nnz, 1, 1, rows_ptr, col_indices, values, NULL, 1, 1, partition_vector));

			AMGX_SAFE_CALL(AMGX_vector_bind(u, A));
			AMGX_SAFE_CALL(AMGX_vector_bind(b, A));
			AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.U));

			
			AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		}
		else
		{
			AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.U));
			AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
			AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		}

        AMGX_SAFE_CALL(AMGX_vector_set_zero(u, (mesh->local_mesh_size + nhalos), 1));

		nvtxRangePush("solveU");
		AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, u));
		nvtxRangePop();
		AMGX_SAFE_CALL(AMGX_vector_download(u, gpu_phi.U));

		/*for(int i = 0; i < mpi_config->particle_flow_world_size; i++){
            if(mpi_config->particle_flow_rank == i){
                //C_kernel_test_values(nnz, values, rows_ptr, col_indices, mesh->local_mesh_size, gpu_S_phi.U, gpu_phi.U);
                            }
                                        MPI_Barrier (mpi_config->particle_flow_world);
                                                }
                                                        MPI_Barrier (mpi_config->particle_flow_world);
*/
		//Solve for V
		thread_count = min((uint64_t) 32,mesh->faces_size);
        	block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));
		
		C_kernel_update_sparse_matrix(block_count, thread_count, UVW_URFactor, mesh->local_mesh_size, gpu_A_phi.V, values, rows_ptr, gpu_S_phi.V, gpu_phi.V, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_hash_map, gpu_boundary_map_values, mesh->local_cells_disp, (gpu_Face<double> *) gpu_face_fields, mesh->mesh_size);
		gpuErrchk( cudaPeekAtLastError() );

		//wait for matrix values
		// gpuErrchk(cudaDeviceSynchronize());

		AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
		AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size +  nhalos), 1, gpu_S_phi.V));
		AMGX_SAFE_CALL(AMGX_vector_set_zero(u, (mesh->local_mesh_size + nhalos), 1));
		AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		
		nvtxRangePush("solveV");
		AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, u));
		nvtxRangePop();
		AMGX_SAFE_CALL(AMGX_vector_download(u, gpu_phi.V));

		//Solve for W
		C_kernel_update_sparse_matrix(block_count, thread_count, UVW_URFactor, mesh->local_mesh_size, gpu_A_phi.W, values, rows_ptr, gpu_S_phi.W, gpu_phi.W, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_hash_map, gpu_boundary_map_values, mesh->local_cells_disp, (gpu_Face<double> *) gpu_face_fields, mesh->mesh_size);
		gpuErrchk( cudaPeekAtLastError() );

		//wait for matrix values
		// gpuErrchk(cudaDeviceSynchronize());

		AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
		AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.W));
		AMGX_SAFE_CALL(AMGX_vector_set_zero(u, (mesh->local_mesh_size + nhalos), 1));
		AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		
		nvtxRangePush("solveW");
		AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, u));
		nvtxRangePop();
		AMGX_SAFE_CALL(AMGX_vector_download(u, gpu_phi.W));
	}

	template<typename T> void FlowSolver<T>::precomp_AU()
	{
		int thread_count = min((uint64_t) 32, mesh->faces_size);
		int block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));

		gpuErrchk(cudaMemset(gpu_A_phi.U, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));

		C_kernel_precomp_AU(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, gpu_boundary_types, effective_viscosity, gpu_face_rlencos, gpu_face_mass_fluxes, gpu_A_phi, mesh->local_mesh_size, delta, gpu_cell_densities, gpu_cell_volumes);
		gpuErrchk( cudaPeekAtLastError() );

	}

	template<typename T> void FlowSolver<T>::get_phi_gradient(T *phi_component, vec<T> *phi_grad_component)
	{
		int thread_count = min((uint64_t) 32, mesh->local_mesh_size);
		int block_count = max(1,(int) ceil((double) mesh->local_mesh_size/(double) thread_count));
		
		//Ensure memory is set to zero
		/*gpuErrchk(cudaMemset2D(full_data_A, A_pitch, 0.0, 9 * sizeof(T), mesh->local_mesh_size));
		gpuErrchk(cudaMemset2D(full_data_bU, bU_pitch, 0.0, 3 * sizeof(T), mesh->local_mesh_size));*/

		/*gpuErrchk(cudaMemset(full_data_A, 0.0, 9 * sizeof(T)));
        gpuErrchk(cudaMemset(full_data_bU, 0.0, 3 * sizeof(T)));*/

		//generate all the data arrays
		C_kernel_get_phi_gradient(block_count, thread_count, phi_component, mesh->local_mesh_size, mesh->local_cells_disp, mesh->faces_per_cell, (gpu_Face<uint64_t> *) gpu_faces, gpu_cell_faces, gpu_cell_centers, mesh->mesh_size, gpu_boundary_hash_map, gpu_boundary_map_values, boundary_map.size(), gpu_face_centers, nhalos, phi_grad_component);
		gpuErrchk( cudaPeekAtLastError() );
		
	}

	template<typename T> void FlowSolver<T>::calculate_mass_flux()
	{
		int thread_count = min((uint64_t) 32,mesh->faces_size);
		int block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));

		C_kernel_calculate_mass_flux(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, mesh->local_mesh_size, boundary_map.size(), gpu_boundary_hash_map, gpu_boundary_map_values, gpu_phi_grad, gpu_cell_centers, gpu_face_centers, gpu_phi, gpu_cell_densities, gpu_A_phi, gpu_cell_volumes, gpu_face_mass_fluxes, gpu_face_lambdas, gpu_face_normals, gpu_face_areas, (gpu_Face<T> *) gpu_face_fields, gpu_S_phi, nhalos, gpu_boundary_types, mesh->dummy_gas_vel);
		
		//Wait for cuda work
		T *FlowIn;
		T *FlowOut;
		T *areaout;
		T *FlowFact;
		int *count_out;

		gpuErrchk(cudaMalloc(&FlowIn, sizeof(T)));
		gpuErrchk(cudaMalloc(&FlowOut, sizeof(T)));
		gpuErrchk(cudaMalloc(&areaout, sizeof(T)));
		gpuErrchk(cudaMalloc(&FlowFact, sizeof(T)));
		gpuErrchk(cudaMalloc(&count_out, sizeof(int)));

		gpuErrchk(cudaMemset(FlowFact, 0.0, sizeof(T)));
		gpuErrchk(cudaMemset(FlowIn, 0.0, sizeof(T)));
		gpuErrchk(cudaMemset(FlowOut, 0.0, sizeof(T)));
		gpuErrchk(cudaMemset(areaout, 0.0, sizeof(T)));
		gpuErrchk(cudaMemset(count_out, 0, sizeof(int)));

		C_kernel_compute_flow_correction(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->mesh_size, gpu_boundary_types, FlowOut, FlowIn, areaout, count_out, gpu_face_mass_fluxes, gpu_face_areas);
		gpuErrchk( cudaPeekAtLastError() );

		
		MPI_Allreduce(MPI_IN_PLACE, areaout, 1, MPI_DOUBLE, MPI_SUM, mpi_config->particle_flow_world);
		MPI_Allreduce(MPI_IN_PLACE, FlowIn, 1, MPI_DOUBLE, MPI_SUM, mpi_config->particle_flow_world);
		MPI_Allreduce(MPI_IN_PLACE, FlowOut, 1, MPI_DOUBLE, MPI_SUM, mpi_config->particle_flow_world);

		C_kernel_correct_flow(block_count, thread_count, count_out, FlowOut, FlowIn, areaout, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->mesh_size, gpu_boundary_types, gpu_face_mass_fluxes, gpu_face_areas, gpu_cell_densities, gpu_phi, mesh->local_mesh_size, nhalos, gpu_face_normals, mesh->local_cells_disp, gpu_S_phi, FlowFact);
		gpuErrchk( cudaPeekAtLastError() );


		C_kernel_correct_flow2(block_count, thread_count, count_out, FlowOut, FlowIn, areaout, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->mesh_size, gpu_boundary_types, gpu_face_mass_fluxes, gpu_face_areas, gpu_cell_densities, gpu_phi, mesh->local_mesh_size, nhalos, gpu_face_normals, mesh->local_cells_disp, gpu_S_phi, FlowFact);
		
		gpuErrchk( cudaPeekAtLastError() );
	}

	template<typename T> void FlowSolver<T>::calculate_pressure(int global_timestep)
	{
		int Loop_num = 0;
		bool Loop_continue = true;
		T *gpu_Pressure_correction_max;
		T cpu_Pressure_correction_max = 0.0;
		T Pressure_correction_ref = 0.0;

		gpuErrchk(cudaMalloc(&gpu_Pressure_correction_max, sizeof(T)));

		gpuErrchk(cudaMemset(gpu_A_phi.V, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_S_phi.U, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_face_mass_fluxes, 0.0, mesh->faces_size * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_face_fields, 0.0, mesh->faces_size * sizeof(Face<T>)));

		exchange_A_halos(gpu_A_phi.U); //exchange A_phi.

		if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
		if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "exchange_A_halos\n");
		if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
		if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());

		calculate_mass_flux();
		//wait for flux happens in C kernel.
		size_t free_mem, total;
		cudaMemGetInfo( &free_mem, &total );
		if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0)
		{
			fprintf(output_file, "GPU memory %lu free of %lu\n", free_mem, total);
		}
		if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
		if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "calculate_mass_flux\n");
		if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
		if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());
		if (mpi_config->particle_flow_rank == 0)
		


		//C_kernel_print(gpu_S_phi.U, mesh->local_mesh_size);

		//gpuErrchk(cudaDeviceSynchronize());	

		gpuErrchk(cudaMemset(nnz, 0, sizeof(int)));
        gpuErrchk(cudaMemset(values, 0.0, sizeof(T) * (mesh->local_mesh_size*7)));
        gpuErrchk(cudaMemset(col_indices, 0, sizeof(int64_t) * (mesh->local_mesh_size*7)));
        gpuErrchk(cudaMemset(rows_ptr, 0, sizeof(int) * (mesh->local_mesh_size+1)));

		int thread_count = min((uint64_t) 32,mesh->local_mesh_size);
        int block_count = max(1,(int) ceil((double) mesh->local_mesh_size/(double) thread_count));

		C_kernel_setup_pressure_matrix(block_count, thread_count, mesh->local_mesh_size, rows_ptr, col_indices, mesh->local_cells_disp, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_hash_map, gpu_boundary_map_values, (gpu_Face<T> *) gpu_face_fields, values, mesh->mesh_size, mesh->faces_per_cell, gpu_cell_faces, nnz, gpu_face_mass_fluxes, gpu_A_phi, gpu_S_phi);
		gpuErrchk( cudaPeekAtLastError() );

		cudaMemGetInfo( &free_mem, &total );
		if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0)
		{
			fprintf(output_file, "GPU memory %lu free of %lu\n", free_mem, total);
		}
		if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
		if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "C_kernel_setup_pressure_matrix\n");
		if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
		if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());
	
		//wait for pressure matrix
		int cpu_nnz = mesh->local_mesh_size*(mesh->faces_per_cell+1);
		//int cpu_nnz = 0;
        //gpuErrchk(cudaMemcpy(nnz, &cpu_nnz, sizeof(int),
        //            cudaMemcpyHostToDevice));

		if(first_press)
		{
			AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(pressure_A, mesh->mesh_size, mesh->local_mesh_size, cpu_nnz, 1, 1, rows_ptr, col_indices, values, NULL, 1, 1, partition_vector));


			free(partition_vector);

			AMGX_SAFE_CALL(AMGX_vector_bind(pressure_u, pressure_A));
			AMGX_SAFE_CALL(AMGX_vector_bind(pressure_b, pressure_A));
	
		}
		else
		{
			AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
			// AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(pressure_A, mesh->local_mesh_size, cpu_nnz, values, NULL));
		}
		AMGX_SAFE_CALL(AMGX_solver_setup(pressure_solver, pressure_A));

		thread_count = min((uint64_t) 32,mesh->faces_size);
		block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));

		while(Loop_continue)
		{
			Loop_num++;
			cpu_Pressure_correction_max = 0.0;
			
			//Solve first pressure update
			if (first_press)
			{
				AMGX_SAFE_CALL(AMGX_vector_upload(pressure_b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.U));
				first_press = false;
			}
			else
			{
				AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.U));
			}

			AMGX_SAFE_CALL(AMGX_vector_set_zero(pressure_u, (mesh->local_mesh_size + nhalos), 1));
		
			nvtxRangePush("solveP");
			AMGX_SAFE_CALL(AMGX_solver_solve(pressure_solver, pressure_b, pressure_u));
			nvtxRangePop();
			AMGX_SAFE_CALL(AMGX_vector_download(pressure_u, gpu_phi.PP));

			C_kernel_find_pressure_correction_max(1, 1, &cpu_Pressure_correction_max, gpu_phi.PP, mesh->local_mesh_size);
			gpuErrchk( cudaPeekAtLastError() );

			MPI_Allreduce(MPI_IN_PLACE, &cpu_Pressure_correction_max, 1, MPI_DOUBLE, MPI_MAX, mpi_config->particle_flow_world);
			//MPI_Allreduce(MPI_IN_PLACE, gpu_Pressure_correction_max, 1, MPI_DOUBLE, MPI_MAX, mpi_config->particle_flow_world);

			//gpuErrchk(cudaMemcpy(&cpu_Pressure_correction_max, gpu_Pressure_correction_max, sizeof(T), cudaMemcpyDeviceToHost));
			
			if(Loop_num == 1)
			{
				Pressure_correction_ref = cpu_Pressure_correction_max;
			}

			exchange_single_phi_halo(gpu_phi.PP);

			thread_count = min((uint64_t) 32,mesh->faces_size);
        	block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));

			C_kernel_Update_P_at_boundaries(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, mesh->local_mesh_size, nhalos, gpu_phi.PP);
		gpuErrchk( cudaPeekAtLastError() );

			//wait for update.
			get_phi_gradient(gpu_phi.PP, gpu_phi_grad.PP);
			//wait for grad done in C kernel.

			exchange_single_grad_halo(gpu_phi_grad.PP);

			C_kernel_update_vel_and_flux(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->local_mesh_size, nhalos, (gpu_Face<T> *) gpu_face_fields, mesh->mesh_size, boundary_map.size(), gpu_boundary_hash_map, gpu_boundary_map_values, gpu_face_mass_fluxes, gpu_A_phi, gpu_phi, gpu_cell_volumes, gpu_phi_grad, global_timestep);
		gpuErrchk( cudaPeekAtLastError() );


			gpuErrchk(cudaMemset(gpu_S_phi.U, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));

			C_kernel_update_mass_flux(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->local_mesh_size, mesh->mesh_size, gpu_boundary_hash_map, gpu_boundary_map_values, boundary_map.size(), gpu_face_centers, gpu_cell_centers, (gpu_Face<double> *) gpu_face_fields, gpu_phi_grad, gpu_face_mass_fluxes, gpu_S_phi, gpu_face_normals);
		gpuErrchk( cudaPeekAtLastError() );

			gpuErrchk(cudaMemset(gpu_phi.PP, 0.0, (mesh->local_mesh_size + nhalos + mesh->boundary_cells_size) * sizeof(T)));
			if(Loop_num >= 4 or (cpu_Pressure_correction_max <= (0.25 * Pressure_correction_ref))) Loop_continue = false;
		}

		C_kernel_Update_P_at_boundaries(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, mesh->local_mesh_size, nhalos, gpu_phi.P);


		get_phi_gradient(gpu_phi.P, gpu_phi_grad.P);
		gpuErrchk( cudaPeekAtLastError() );

		C_kernel_Update_P(block_count, thread_count, mesh->faces_size, mesh->local_mesh_size, nhalos, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, gpu_cell_centers, gpu_face_centers, gpu_boundary_types, gpu_phi.P, gpu_phi_grad.P);
		gpuErrchk( cudaPeekAtLastError() );

		cudaMemGetInfo( &free_mem, &total );
		if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0)
		{
			fprintf(output_file, "After pressure GPU memory %lu free of %lu\n", free_mem, total);
		}
	}

	template<typename T> void FlowSolver<T>::Scalar_solve(int type, T *phi_component, vec<T> *phi_grad_component)
	{
		gpuErrchk(cudaMemset(gpu_A_phi.V, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_S_phi.U, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));

		int thread_count = min((uint64_t) 32,mesh->faces_size);
		int block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));

		//TODO;If not one of our types exit
		C_kernel_flux_scalar(block_count, thread_count, type, mesh->faces_size, mesh->local_mesh_size, nhalos, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, gpu_cell_centers, gpu_face_centers, gpu_boundary_types, gpu_boundary_hash_map, gpu_boundary_map_values, boundary_map.size(), phi_component, gpu_A_phi, gpu_S_phi, phi_grad_component, gpu_face_lambdas, effective_viscosity, gpu_face_rlencos, gpu_face_mass_fluxes, gpu_face_normals, inlet_effective_viscosity, (gpu_Face<T> *) gpu_face_fields, mesh->dummy_gas_vel, mesh->dummy_gas_tem, mesh->dummy_gas_fuel);
		gpuErrchk( cudaPeekAtLastError() );

		thread_count = min((uint64_t) 32,mesh->local_mesh_size);
		block_count = max(1,(int) ceil((double) mesh->local_mesh_size/(double) thread_count));

		C_kernel_apply_pres_forces(block_count, thread_count, type, mesh->local_mesh_size, delta, gpu_cell_densities, gpu_cell_volumes, phi_component, gpu_A_phi, gpu_S_phi, gpu_particle_terms);
		gpuErrchk( cudaPeekAtLastError() );

		if(type == TERBTE or type == TERBED)
		{
			C_kernel_solve_turb_models_cell(block_count, thread_count, type, mesh->local_mesh_size, gpu_A_phi, gpu_S_phi, gpu_phi_grad, gpu_phi, effective_viscosity, gpu_cell_densities, gpu_cell_volumes);
			gpuErrchk( cudaPeekAtLastError() );

			thread_count = min((uint64_t) 32,mesh->faces_size);
			block_count = max(1,(int) ceil((double) mesh->faces_size/ (double) thread_count));


			C_kernel_solve_turb_models_face(block_count, thread_count, type, mesh->faces_size, mesh->mesh_size, gpu_cell_volumes, effective_viscosity, gpu_face_centers, gpu_cell_centers, gpu_face_normals, gpu_cell_densities, mesh->local_mesh_size, nhalos, mesh->local_cells_disp, gpu_A_phi, gpu_S_phi, gpu_phi, (gpu_Face<T> *) gpu_face_fields, (gpu_Face<uint64_t> *) gpu_faces, gpu_boundary_types, mesh->faces_per_cell, gpu_cell_faces);
			gpuErrchk( cudaPeekAtLastError() );

		}

		double URFactor = 0.0;
		if(type == TERBTE or type == TERBED)
		{
			URFactor = 0.5;
		}
		else if(type == TEMP)
		{
			URFactor = 0.95;
		}
		else
		{
			URFactor = 0.95;
		}

		gpuErrchk(cudaMemset(nnz, 0, sizeof(int)));
        gpuErrchk(cudaMemset(values, 0.0, sizeof(T) * (mesh->local_mesh_size*7)));
        gpuErrchk(cudaMemset(col_indices, 0, sizeof(int64_t) * (mesh->local_mesh_size*7)));
        gpuErrchk(cudaMemset(rows_ptr, 0, sizeof(int) * (mesh->local_mesh_size+1)));

		thread_count = min((uint64_t) 32,mesh->faces_size);
        block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));

		C_kernel_setup_sparse_matrix(block_count, thread_count, URFactor, mesh->local_mesh_size, rows_ptr, col_indices, mesh->local_cells_disp, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_hash_map, gpu_boundary_map_values, gpu_A_phi.V, (gpu_Face<double> *) gpu_face_fields, values, gpu_S_phi.U, phi_component, mesh->mesh_size, mesh->faces_per_cell, gpu_cell_faces, nnz);
		gpuErrchk( cudaPeekAtLastError() );
	

		int cpu_nnz = mesh->local_mesh_size*(mesh->faces_per_cell+1);
		//int cpu_nnz = 0;
        //gpuErrchk(cudaMemcpy(&cpu_nnz, nnz, sizeof(int),
        //            cudaMemcpyDeviceToHost));

		AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
		gpuErrchk( cudaPeekAtLastError() );
	    AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.U));
		gpuErrchk( cudaPeekAtLastError() );
		AMGX_SAFE_CALL(AMGX_vector_set_zero(u, (mesh->local_mesh_size + nhalos), 1));
		gpuErrchk( cudaPeekAtLastError() );
	    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		gpuErrchk( cudaPeekAtLastError() );

		nvtxRangePush("solveScalar");
		AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, u));
		nvtxRangePop();
		gpuErrchk( cudaPeekAtLastError() );
		AMGX_SAFE_CALL(AMGX_vector_download(u, phi_component));
		gpuErrchk( cudaPeekAtLastError() );
	}

	template<typename T> void FlowSolver<T>::set_up_field()
    {
        /*We need inital values for mass_flux and AU for the first iteration*/
        if (FLOW_SOLVER_DEBUG)  fprintf(output_file, "\tRank %d: Running function set_up_field.\n", mpi_config->rank);

		precomp_AU();
        exchange_A_halos(gpu_A_phi.U);
		calculate_mass_flux();
	}

	template<typename T> void FlowSolver<T>::set_up_fgm_table()
	{
		int thread_count = 256;
		int block_count = max(1,(int) ceil((double) 100*100*100*100/ (double) thread_count));

		gpuErrchk(cudaMalloc(&gpu_fgm_table, sizeof(double)*100*100*100*100));

		C_kernel_set_up_fgm_table(block_count, thread_count, gpu_fgm_table, time(NULL));
		gpuErrchk( cudaPeekAtLastError() );

	}

	template<typename T> void FlowSolver<T>::FGM_look_up()
	{
		int thread_count = min((uint64_t) 32, mesh->local_mesh_size);
		int block_count = max(1,(int) ceil((double) mesh->local_mesh_size/(double) thread_count));
		C_kernel_fgm_look_up(block_count, thread_count, gpu_fgm_table, gpu_S_phi, gpu_phi, mesh->local_mesh_size);
		
		if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
		if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "C_kernel_fgm_look_up\n");
		if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
		if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());
	}

	template<class T> void FlowSolver<T>::print_logger_stats(uint64_t timesteps, double runtime)
    {
		/*Print out some statistics at about the flow solver*/
        Flow_Logger loggers[mpi_config->particle_flow_world_size];
        MPI_Gather(&logger, sizeof(Flow_Logger), MPI_BYTE, &loggers, sizeof(Flow_Logger), MPI_BYTE, 0, mpi_config->particle_flow_world);

        double max_red_cells = loggers[0].reduced_recieved_cells;
        double min_red_cells = loggers[0].reduced_recieved_cells;
        double max_cells     = loggers[0].recieved_cells;
        double min_cells     = loggers[0].recieved_cells;
        double min_nodes = loggers[0].sent_nodes;
        double max_nodes = loggers[0].sent_nodes;


		double min_min_cells_buf_size_per_timestep = loggers[0].min_cells_buf_size_per_timestep;
		double min_min_nodes_buf_size_per_timestep = loggers[0].min_nodes_buf_size_per_timestep;
		double min_max_cells_buf_size_per_timestep = loggers[0].max_cells_buf_size_per_timestep;
		double min_max_nodes_buf_size_per_timestep = loggers[0].max_nodes_buf_size_per_timestep;

		double max_min_cells_buf_size_per_timestep = loggers[0].min_cells_buf_size_per_timestep;
		double max_min_nodes_buf_size_per_timestep = loggers[0].min_nodes_buf_size_per_timestep;
		double max_max_cells_buf_size_per_timestep = loggers[0].max_cells_buf_size_per_timestep;
		double max_max_nodes_buf_size_per_timestep = loggers[0].max_nodes_buf_size_per_timestep;

        double non_zero_blocks      = 0;
        double total_cells_recieved = 0;
        double total_reduced_cells_recieves = 0;
        if (mpi_config->particle_flow_rank == 0)
        {
            memset(&logger,           0, sizeof(Flow_Logger));
            for (int rank = 0; rank < mpi_config->particle_flow_world_size; rank++)
            {
                total_cells_recieved          += loggers[rank].recieved_cells;
                total_reduced_cells_recieves  += loggers[rank].reduced_recieved_cells;
                logger.reduced_recieved_cells += loggers[rank].reduced_recieved_cells;
                logger.recieved_cells         += loggers[rank].recieved_cells;
                logger.sent_nodes             += loggers[rank].sent_nodes;

				logger.min_cells_buf_size_per_timestep += loggers[rank].min_cells_buf_size_per_timestep;
                logger.min_nodes_buf_size_per_timestep += loggers[rank].min_nodes_buf_size_per_timestep;
                logger.max_cells_buf_size_per_timestep += loggers[rank].max_cells_buf_size_per_timestep;
                logger.max_nodes_buf_size_per_timestep += loggers[rank].max_nodes_buf_size_per_timestep;

				if ( min_min_cells_buf_size_per_timestep > loggers[rank].min_cells_buf_size_per_timestep ) min_min_cells_buf_size_per_timestep = loggers[rank].min_cells_buf_size_per_timestep;
                if ( min_min_nodes_buf_size_per_timestep > loggers[rank].min_nodes_buf_size_per_timestep ) min_min_nodes_buf_size_per_timestep = loggers[rank].min_nodes_buf_size_per_timestep;
                if ( min_max_cells_buf_size_per_timestep > loggers[rank].max_cells_buf_size_per_timestep ) min_max_cells_buf_size_per_timestep = loggers[rank].max_cells_buf_size_per_timestep;
                if ( min_max_nodes_buf_size_per_timestep > loggers[rank].max_nodes_buf_size_per_timestep ) min_max_nodes_buf_size_per_timestep = loggers[rank].max_nodes_buf_size_per_timestep;

				if ( max_min_cells_buf_size_per_timestep < loggers[rank].min_cells_buf_size_per_timestep ) max_min_cells_buf_size_per_timestep = loggers[rank].min_cells_buf_size_per_timestep;
				if ( max_min_nodes_buf_size_per_timestep < loggers[rank].min_nodes_buf_size_per_timestep ) max_min_nodes_buf_size_per_timestep = loggers[rank].min_nodes_buf_size_per_timestep;
				if ( max_max_cells_buf_size_per_timestep < loggers[rank].max_cells_buf_size_per_timestep ) max_max_cells_buf_size_per_timestep = loggers[rank].max_cells_buf_size_per_timestep;
				if ( max_max_nodes_buf_size_per_timestep < loggers[rank].max_nodes_buf_size_per_timestep ) max_max_nodes_buf_size_per_timestep = loggers[rank].max_nodes_buf_size_per_timestep;


                if ( min_cells > loggers[rank].recieved_cells )  min_cells = loggers[rank].recieved_cells ;
                if ( max_cells < loggers[rank].recieved_cells )  max_cells = loggers[rank].recieved_cells ;

                if ( min_red_cells > loggers[rank].reduced_recieved_cells )  min_red_cells = loggers[rank].reduced_recieved_cells ;
                if ( max_red_cells < loggers[rank].reduced_recieved_cells )  max_red_cells = loggers[rank].reduced_recieved_cells ;

                if ( min_nodes > loggers[rank].sent_nodes )  min_nodes = loggers[rank].sent_nodes ;
                if ( max_nodes < loggers[rank].sent_nodes )  max_nodes = loggers[rank].sent_nodes ;
            }
            
            for (int rank = 0; rank < mpi_config->particle_flow_world_size; rank++) 
                non_zero_blocks += loggers[rank].recieved_cells > (0.01 * max_cells) ;

            logger.reduced_recieved_cells /= non_zero_blocks;
            logger.recieved_cells /= non_zero_blocks;
            logger.sent_nodes     /= non_zero_blocks;

			logger.min_cells_buf_size_per_timestep /= mpi_config->particle_flow_world_size;
			logger.min_nodes_buf_size_per_timestep /= mpi_config->particle_flow_world_size;
			logger.max_cells_buf_size_per_timestep /= mpi_config->particle_flow_world_size;
			logger.max_nodes_buf_size_per_timestep /= mpi_config->particle_flow_world_size;
            
            fprintf(output_file, "Flow Solver Stats:\t                            AVG       MIN       MAX\n");
            fprintf(output_file, "\tmin_cells_recv    ( per rank )         : %9.0f %9.0f %9.0f\n", logger.min_cells_buf_size_per_timestep, min_min_cells_buf_size_per_timestep, max_min_cells_buf_size_per_timestep);
            fprintf(output_file, "\tmax_cells_recv    ( per rank )         : %9.0f %9.0f %9.0f\n", logger.max_cells_buf_size_per_timestep, min_max_cells_buf_size_per_timestep, max_max_cells_buf_size_per_timestep);
			fprintf(output_file, "\tmin_nodes_recv    ( per rank )         : %9.0f %9.0f %9.0f\n", logger.min_cells_buf_size_per_timestep, min_min_cells_buf_size_per_timestep, max_min_cells_buf_size_per_timestep);
            fprintf(output_file, "\tmax_nodes_recv    ( per rank )         : %9.0f %9.0f %9.0f\n", logger.max_cells_buf_size_per_timestep, min_max_cells_buf_size_per_timestep, max_max_cells_buf_size_per_timestep);
           
		   
		   
		    fprintf(output_file, "\tFlow blocks with <1%% max droplets  : %d\n", mpi_config->particle_flow_world_size - (int)non_zero_blocks); 
            fprintf(output_file, "\tAvg Cells with droplets             : %.2f%%\n", 100 * total_cells_recieved / (timesteps * mesh->mesh_size));
            fprintf(output_file, "\tCell copies across particle ranks   : %.2f%%\n", 100.*(1 - total_reduced_cells_recieves / total_cells_recieved ));




            
            MPI_Barrier (mpi_config->particle_flow_world);
            cout << endl;
        }
        else
        {
            MPI_Barrier (mpi_config->particle_flow_world);
        }

        MPI_Barrier(mpi_config->world);


        performance_logger.print_counters(mpi_config->rank, mpi_config->world_size, runtime);

    }

    template<typename T> void FlowSolver<T>::timestep()
    {
		nvtxRangePush(__FUNCTION__);
		/*High level function to advance the flow solver one timestep.*/
        if (FLOW_SOLVER_DEBUG)  fprintf(output_file, "\tFlow Rank %d: Start flow timestep.\n", mpi_config->rank);

        int comms_timestep = 3;
		if ( mpi_config->particle_flow_rank == 0 )
			fprintf(output_file, "timestep %lu\n",timestep_count + 1);


		for (int subiter = 0; subiter < 3; subiter++)
		{
			// mtracker->print_usage();

			size_t free_mem, total;
			cudaMemGetInfo( &free_mem, &total );

			if (mpi_config->particle_flow_rank == 0)
			{
				fprintf(output_file, "GPU memory %lu free of %lu\n", free_mem, total);
			}
			
			if (((timestep_count + 1) % 100) == 0)
			{
				double arr_usage  = ((double)get_array_memory_usage()) / 1.e9;
				double stl_usage  = ((double)get_stl_memory_usage())   / 1.e9 ;
				double mesh_usage = ((double)mesh->get_memory_usage()) / 1.e9 ;
				double arr_usage_total, stl_usage_total, mesh_usage_total;

				MPI_Reduce(&arr_usage,  &arr_usage_total,  1, MPI_DOUBLE, 
							MPI_SUM, 0, mpi_config->particle_flow_world);
				MPI_Reduce(&stl_usage,  &stl_usage_total,  1, MPI_DOUBLE, 
							MPI_SUM, 0, mpi_config->particle_flow_world);
				MPI_Reduce(&mesh_usage, &mesh_usage_total, 1, MPI_DOUBLE, 
							MPI_SUM, 0, mpi_config->particle_flow_world);

				if ( mpi_config->particle_flow_rank == 0 )
				{
					fprintf(output_file, "Timestep %6lu Flow     mem (TOTAL %8.3f GB)" 
							"(AVG %8.3f GB) \n", timestep_count + 1, 
							(arr_usage_total + stl_usage_total + mesh_usage_total), 
							(arr_usage_total + stl_usage_total + mesh_usage_total) / 
							mpi_config->particle_flow_world_size);

				}
			}
			

			//NOTE: comparing the parallel and serial version 
			//We get idential A and b with UVW but not idential results
			//This is probably due to parallel solvers??
			//We can track through all the differences in the pressure
			//Solve to these differences.
			
			// Note: After pointer swap, last iterations phi is now in phi.
			//TODO: Should we be doing this?

			nvtxRangePush("exchange_phi_halos");

			compute_time -= MPI_Wtime();
			exchange_phi_halos();
			nvtxRangePop();


			if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
			if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "exchange_A_halos\n");
			if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
			if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());

			nvtxRangePush("get_phi_gradients");

			flow_timings[0] -= MPI_Wtime();
			get_phi_gradients();
			flow_timings[0] += MPI_Wtime();
			nvtxRangePop();


			if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
			if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "get_phi_gradients\n");
			if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
			if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());

			//C_kernel_vec_print(gpu_phi_grad.U, mesh->local_mesh_size);

	//		if(FLOW_SOLVER_LIMIT_GRAD)
	//			limit_phi_gradients();
			nvtxRangePush("exchange_grad_halos");

			exchange_grad_halos();
			nvtxRangePop();

		
			nvtxRangePush("set_up_field");

			if(timestep_count == 0)
			{
				set_up_field();
				set_up_fgm_table();
			}
			nvtxRangePop();

			compute_time += MPI_Wtime();
		
			flow_timings[1] -= MPI_Wtime();

			if (((timestep_count+subiter) % comms_timestep) == 0)
			{

				int thread_count = min((uint64_t) 32, (mesh->local_mesh_size + nhalos));
				int block_count = max(1,(int) ceil((double) (mesh->local_mesh_size + nhalos)/(double) thread_count));
				// int thread_count = min((uint64_t) 32, (mesh->local_mesh_size));
				// int block_count = max(1,(int) ceil((double) (mesh->local_mesh_size)/(double) thread_count));

				// C_kernel_interpolate_init_boundaries(block_count1, thread_count1, gpu_phi_nodes, gpu_cells_per_point, global_node_to_local_node_map.size());
				// gpuErrchk( cudaPeekAtLastError() );

				if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "C_kernel_interpolate_phi_to_nodes mesh_size + halos %lu mesh_size %lu halos %lu\n", mesh->local_mesh_size + nhalos, mesh->local_mesh_size , nhalos);

				C_kernel_interpolate_phi_to_nodes(block_count, thread_count, gpu_phi, gpu_phi_grad, gpu_phi_nodes, gpu_local_nodes, gpu_node_hash_map, gpu_boundary_hash_map, gpu_cells_per_point, gpu_local_cells, gpu_cell_centers, mesh->local_mesh_size, mesh->local_cells_disp, global_node_to_local_node_map.size(), nhalos);
				gpuErrchk( cudaPeekAtLastError() );



			if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
			if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "C_kernel_interpolate_phi_to_nodes\n");
			if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
			if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());

				// gpuErrchk(cudaMemcpyAsync(phi.U, gpu_phi.U,     phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0)); // Asynchronous memcpy to default stream
				// gpuErrchk(cudaMemcpyAsync(phi.V, gpu_phi.V,     phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
				// gpuErrchk(cudaMemcpyAsync(phi.W, gpu_phi.W,     phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
				// gpuErrchk(cudaMemcpyAsync(phi.P, gpu_phi.P,     phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
				// gpuErrchk(cudaMemcpyAsync(phi.TEM, gpu_phi.TEM, phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));  
				// gpuErrchk( cudaPeekAtLastError() );


				// gpuErrchk(cudaMemcpyAsync(phi_nodes.U,   gpu_phi_nodes.U,   phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0)); // Asynchronous memcpy to default stream
				// gpuErrchk(cudaMemcpyAsync(phi_nodes.V,   gpu_phi_nodes.V,   phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
				// gpuErrchk(cudaMemcpyAsync(phi_nodes.W,   gpu_phi_nodes.W,   phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
				// gpuErrchk(cudaMemcpyAsync(phi_nodes.P,   gpu_phi_nodes.P,   phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
				// gpuErrchk(cudaMemcpyAsync(phi_nodes.TEM, gpu_phi_nodes.TEM, phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));  
				update_flow_field();

				
				
			}

			if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
			if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "update_flow_field\n");
			if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
			if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());

			flow_timings[1] += MPI_Wtime();
			compute_time -= MPI_Wtime();

			flow_timings[2] -= MPI_Wtime();
			nvtxRangePush("calculate_UVW");

			calculate_UVW();
			nvtxRangePop();
			flow_timings[2] += MPI_Wtime();

			nvtxRangePush("exchange_phi_halos");


			if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
			if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "calculate_UVW\n");
			if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
			if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());

			exchange_phi_halos(); //exchange new UVW values.
			nvtxRangePop();

		
			if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
			if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "exchange_phi_halos\n");
			if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
			if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());

			nvtxRangePush("calculate_pressure");

			flow_timings[3] -= MPI_Wtime();
			calculate_pressure(timestep_count + subiter); 
			flow_timings[3] += MPI_Wtime();

			if (CUDA_SYNC_DEBUG) MPI_Barrier(mpi_config->particle_flow_world);
			if (CUDA_SYNC_DEBUG && mpi_config->particle_flow_rank == 0) fprintf(output_file, "calculate_pressure\n");
			if (CUDA_SYNC_DEBUG) gpuErrchk( cudaPeekAtLastError() );
			if (CUDA_SYNC_DEBUG) gpuErrchk(cudaDeviceSynchronize());


			nvtxRangePop();

			nvtxRangePush("Scalar_solve");

			flow_timings[4] -= MPI_Wtime();
			//Turbulence solve
			Scalar_solve(TERBTE, gpu_phi.TE, gpu_phi_grad.TE);
			flow_timings[4] += MPI_Wtime();
			flow_timings[5] -= MPI_Wtime();
			Scalar_solve(TERBED, gpu_phi.ED, gpu_phi_grad.ED);
			flow_timings[5] += MPI_Wtime();		

			flow_timings[6] -= MPI_Wtime();
			//temperature solve
			Scalar_solve(TEMP, gpu_phi.TEM, gpu_phi_grad.TEM);
			flow_timings[6] += MPI_Wtime();

			flow_timings[7] -= MPI_Wtime();
			//fuel mixture fraction solve
			Scalar_solve(FUEL, gpu_phi.FUL, gpu_phi_grad.FUL);
			flow_timings[7] += MPI_Wtime();		

			flow_timings[8] -= MPI_Wtime();
			//rection progression solve
			Scalar_solve(PROG, gpu_phi.PRO, gpu_phi_grad.PRO);
			flow_timings[8] += MPI_Wtime();

			flow_timings[9] -= MPI_Wtime();
			//Solve Variance of mixture fraction as transport equ
			Scalar_solve(VARFU, gpu_phi.VARF, gpu_phi_grad.VARF);
			flow_timings[9] += MPI_Wtime();

			flow_timings[10] -= MPI_Wtime();
			//Solve Variance of progression as trasnport equ
			Scalar_solve(VARPR, gpu_phi.VARP, gpu_phi_grad.VARP);
			flow_timings[10] += MPI_Wtime();		

			nvtxRangePop();

			nvtxRangePush("FGM_look_up");

			fgm_lookup_time -= MPI_Wtime();
			//Look up results from the FGM look-up table
			FGM_look_up();
			fgm_lookup_time += MPI_Wtime();
			compute_time += MPI_Wtime();

			nvtxRangePop();

			if(((timestep_count + 1) % TIMER_OUTPUT_INTERVAL == 0) && FLOW_SOLVER_TIME)
			{
				if(mpi_config->particle_flow_rank == 0)
				{
					MPI_Reduce(MPI_IN_PLACE, &fgm_lookup_time, 1, MPI_DOUBLE,
								MPI_SUM, 0, mpi_config->particle_flow_world);
					double total_time = vel_total_time + pres_total_time + 
										fgm_lookup_time;
					double scalar_time = 0.0;
					for(int i = 0; i < 8; i++)
					{
						total_time += sca_total_time[i];
						scalar_time += sca_total_time[i];
					} 

					fprintf(output_file, "\nTotal Flow Solver Timings:\n");
					fprintf(output_file, "Compute Velocity time: %.3f (%.2f %%)\n",
							vel_total_time / mpi_config->particle_flow_world_size,
							100 * vel_total_time / total_time);
					fprintf(output_file, "Compute Pressure time: %.3f (%.2f %%)\n",
							pres_total_time / mpi_config->particle_flow_world_size,
							100 * pres_total_time / total_time);
					fprintf(output_file, "Compute 8 Scalars time %.3f (%.2f %%)\n",
							scalar_time / mpi_config->particle_flow_world_size,
							100 * scalar_time / total_time);
					fprintf(output_file, "FGM Table Lookup time %.3f (%.2f %%)\n",
							fgm_lookup_time / mpi_config->particle_flow_world_size,
							100 * fgm_lookup_time / total_time);
					fprintf(output_file, "Total time: %f\n\n", 
							total_time / mpi_config->particle_flow_world_size);
				}
				else
				{
					MPI_Reduce(&fgm_lookup_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
								0, mpi_config->particle_flow_world);
				}
				vel_total_time = 0.0;
				pres_total_time = 0.0;
				fgm_lookup_time = 0.0;
				for(int i = 0; i < 8; i++)
				{
					sca_total_time[i] = 0.0;
				}
			}

			if(timestep_count + 1 == 5)
			{
				if(mpi_config->particle_flow_rank == 0)
				{
					MPI_Reduce(MPI_IN_PLACE, flow_timings, 11, MPI_DOUBLE, MPI_SUM,
							0, mpi_config->particle_flow_world);
					for(int i = 0; i < 11; i++)
					{
						flow_timings[i] /= mpi_config->particle_flow_world_size;
					}
					fprintf(output_file, "\nFlow Timing: \nCalc gradients: %f\nCalc update particles: %f\nCalc velocity: %f\nCalc Pressure: %f\nCalc Turb TE: %f\nCalc Turb ED: %f\nCalc Heat: %f\nCalc PROG: %f\nCalc FUEL: %f\nCalc VAR PROG: %f\nCalc VAR FUEL: %f\n",flow_timings[0],flow_timings[1],flow_timings[2],flow_timings[3],flow_timings[4],flow_timings[5],flow_timings[6],flow_timings[7],flow_timings[8],flow_timings[9],flow_timings[10]);
				}
				else
				{
					MPI_Reduce(flow_timings, nullptr, 11, MPI_DOUBLE, MPI_SUM,
							0, mpi_config->particle_flow_world);
				}
			}		
			
			if ( FLOW_SOLVER_DEBUG )  fprintf(output_file, "\tFlow Rank %d: Stop flow timestep.\n", mpi_config->rank);

		}


        timestep_count++;

		nvtxRangePop();

    }
}
