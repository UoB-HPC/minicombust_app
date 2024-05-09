#include <stdio.h>
#include <limits.h>

#include "flow/gpu/FlowSolver.hpp"

#include <nvToolsExt.h>

using namespace std;

namespace minicombust::flow 
{

	template<class T>void FlowSolver<T>::output_data(uint64_t timestep)
    	{
		VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh, mpi_config);
        	vtk_writer->write_flow_velocities("out/minicombust", timestep, &phi);
		vtk_writer->write_flow_pressure("out/minicombust", timestep, &phi);
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
				MPI_Irecv( &gpu_phi_grad.PP[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
			}


			for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
			{
				int bytes_size;
				MPI_Type_size(halo_mpi_vec_double_datatypes[r], &bytes_size);

				int elements = halo_sizes[r];

				int thread_count = min((int) 256, elements);
				int block_count  = max(1, (int) ceil((double) (elements) / (double) thread_count));
				C_kernel_pack_PP_grad_halo_buffer(block_count, thread_count, gpu_phi_grad_send_buffers[r], gpu_phi_grad, gpu_halo_indexes[r], (uint64_t)(elements));
		gpuErrchk( cudaPeekAtLastError() );

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

		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			int bytes_size;
			MPI_Type_size(halo_mpi_vec_double_datatypes[r], &bytes_size);

			int elements = halo_sizes[r];

			int thread_count = min((int) 256, elements);
			int block_count  = max(1, (int) ceil((double) (elements) / (double) thread_count));
			C_kernel_pack_phi_grad_halo_buffer(block_count, thread_count, gpu_phi_grad_send_buffers[r], gpu_phi_grad, gpu_halo_indexes[r], (uint64_t)(elements));
		gpuErrchk( cudaPeekAtLastError() );

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


		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
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

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
			int bytes_size;
			MPI_Type_size(halo_mpi_double_datatypes[r], &bytes_size);

			int elements = halo_sizes[r];

			int thread_count = min( (int) 256, elements);
			int block_count = max(1, (int) ceil((double) (elements) / (double) thread_count));

			C_kernel_pack_phi_halo_buffer(block_count, thread_count, gpu_phi_send_buffers[r], gpu_phi, gpu_halo_indexes[r], (uint64_t)(elements));
			gpuErrchk( cudaPeekAtLastError() );

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

    template<typename T> void FlowSolver<T>::get_neighbour_cells ( const uint64_t recv_id )
    {
		/*Find all the cells which are neighbours of the current cell.
		  Used to find the halos*/
        double node_neighbours   = 8;
        const uint64_t cell_size = mesh->cell_size;

        resize_nodes_arrays(node_to_position_map.size() + elements[recv_id] * cell_size + 1 );

		int local_disp = 0;
		// resize_send_buffers_nodes_arrays (send_buffer_disp + elements[recv_id] * cell_size + 1);

        #pragma ivdep
        for (int i = 0; i < elements[recv_id]; i++)
        {
            uint64_t cell = neighbour_indexes[recv_id][i];

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id      = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];
				if (!local_particle_node_sets[recv_id].contains(node_id))
				{
                	local_particle_node_sets[recv_id].insert(node_id);
					send_buffers_interp_node_indexes[send_buffer_disp + local_disp] = node_id;
					local_disp++;
				}
            }

			if ( new_cells_set.contains(cell) )  continue;

            new_cells_set.insert(cell);
            unordered_neighbours_set[0].insert(cell);   
            

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id      = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];

                if (!node_to_position_map.count(node_id))
                {
                    const T boundary_neighbours = node_neighbours - mesh->cells_per_point[node_id - mesh->shmem_point_disp];

                    flow_aos<T> temp_term;
                    temp_term.vel      = mesh->dummy_gas_vel * (boundary_neighbours / node_neighbours);
                    temp_term.pressure = mesh->dummy_gas_pre * (boundary_neighbours / node_neighbours);
                    temp_term.temp     = mesh->dummy_gas_tem * (boundary_neighbours / node_neighbours);

                    const uint64_t position = node_to_position_map.size();
                    interp_node_indexes[position]     = node_id;
                    interp_node_flow_fields[position] = temp_term; 
                    node_to_position_map[node_id]     = position;
                }
            }

            // Get 6 immediate neighbours
            const uint64_t below_neighbour                = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + DOWN_FACE];
            const uint64_t above_neighbour                = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + UP_FACE];
            const uint64_t around_left_neighbour          = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + LEFT_FACE];
            const uint64_t around_right_neighbour         = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + RIGHT_FACE];
            const uint64_t around_front_neighbour         = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + FRONT_FACE];
            const uint64_t around_back_neighbour          = mesh->cell_neighbours[ (cell - mesh->shmem_cell_disp) * mesh->faces_per_cell + BACK_FACE];

            unordered_neighbours_set[0].insert(below_neighbour);             // Immediate neighbour cell indexes are correct   
            unordered_neighbours_set[0].insert(above_neighbour);             // Immediate neighbour cell indexes are correct  
            unordered_neighbours_set[0].insert(around_left_neighbour);       // Immediate neighbour cell indexes are correct   
            unordered_neighbours_set[0].insert(around_right_neighbour);      // Immediate neighbour cell indexes are correct   
            unordered_neighbours_set[0].insert(around_front_neighbour);      // Immediate neighbour cell indexes are correct   
            unordered_neighbours_set[0].insert(around_back_neighbour);       // Immediate neighbour cell indexes are correct   

            // Get 8 cells neighbours around
            if ( around_left_neighbour != MESH_BOUNDARY  )   // If neighbour isn't edge of mesh and isn't a halo cell
            {
                const uint64_t around_left_front_neighbour    = mesh->cell_neighbours[ (around_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                const uint64_t around_left_back_neighbour     = mesh->cell_neighbours[ (around_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(around_left_front_neighbour);    
                unordered_neighbours_set[0].insert(around_left_back_neighbour);     
            }
            if ( around_right_neighbour != MESH_BOUNDARY )
            {
                const uint64_t around_right_front_neighbour   = mesh->cell_neighbours[ (around_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell + FRONT_FACE] ;
                const uint64_t around_right_back_neighbour    = mesh->cell_neighbours[ (around_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(around_right_front_neighbour);   
                unordered_neighbours_set[0].insert(around_right_back_neighbour); 
            }
            if ( below_neighbour != MESH_BOUNDARY )
            {
                // Get 8 cells around below cell
                const uint64_t below_left_neighbour           = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + LEFT_FACE]  ;
                const uint64_t below_right_neighbour          = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + RIGHT_FACE] ;
                const uint64_t below_front_neighbour          = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + FRONT_FACE] ;
                const uint64_t below_back_neighbour           = mesh->cell_neighbours[ (below_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(below_left_neighbour);           
                unordered_neighbours_set[0].insert(below_right_neighbour);          
                unordered_neighbours_set[0].insert(below_front_neighbour);          
                unordered_neighbours_set[0].insert(below_back_neighbour);           
                if ( below_left_neighbour != MESH_BOUNDARY )
                {
                    const uint64_t below_left_front_neighbour     = mesh->cell_neighbours[ (below_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + FRONT_FACE] ;
                    const uint64_t below_left_back_neighbour      = mesh->cell_neighbours[ (below_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(below_left_front_neighbour);     
                    unordered_neighbours_set[0].insert(below_left_back_neighbour);      
                }
                if ( below_right_neighbour != MESH_BOUNDARY )
                {
                    const uint64_t below_right_front_neighbour    = mesh->cell_neighbours[ (below_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                    const uint64_t below_right_back_neighbour     = mesh->cell_neighbours[ (below_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(below_right_front_neighbour);    
                    unordered_neighbours_set[0].insert(below_right_back_neighbour); 
                }
            }
            if ( above_neighbour != MESH_BOUNDARY )
            {
                // Get 8 cells neighbours above
                const uint64_t above_left_neighbour           = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + LEFT_FACE]  ;
                const uint64_t above_right_neighbour          = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + RIGHT_FACE] ;
                const uint64_t above_front_neighbour          = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + FRONT_FACE] ;
                const uint64_t above_back_neighbour           = mesh->cell_neighbours[ (above_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell        + BACK_FACE]  ;
                unordered_neighbours_set[0].insert(above_left_neighbour);           
                unordered_neighbours_set[0].insert(above_right_neighbour);          
                unordered_neighbours_set[0].insert(above_front_neighbour);          
                unordered_neighbours_set[0].insert(above_back_neighbour);           
                if ( above_left_neighbour != MESH_BOUNDARY )
                {
                    const uint64_t above_left_front_neighbour     = mesh->cell_neighbours[ (above_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + FRONT_FACE] ;
                    const uint64_t above_left_back_neighbour      = mesh->cell_neighbours[ (above_left_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell   + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(above_left_front_neighbour);     
                    unordered_neighbours_set[0].insert(above_left_back_neighbour);      
                }
                if ( above_right_neighbour != MESH_BOUNDARY )
                {
                    const uint64_t above_right_front_neighbour    = mesh->cell_neighbours[ (above_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + FRONT_FACE] ;
                    const uint64_t above_right_back_neighbour     = mesh->cell_neighbours[ (above_right_neighbour - mesh->shmem_cell_disp) * mesh->faces_per_cell  + BACK_FACE]  ;
                    unordered_neighbours_set[0].insert(above_right_front_neighbour);    
                    unordered_neighbours_set[0].insert(above_right_back_neighbour);     
                }
            }
        }


		if (local_disp != 0){
			


			int thread_count = min( (int) 32, (int)local_particle_node_sets[recv_id].size());
			int block_count = max(1, (int) ceil((double) (local_particle_node_sets[recv_id].size()) / (double) thread_count));

			// printf("Block size %lu, thread count %lu\n", block_count, thread_count);
			// gpuErrchk( cudaPeekAtLastError() );

			if ((send_buffer_disp + local_particle_node_sets[recv_id].size()) * sizeof(uint64_t) >= gpu_send_buffers_node_index_array_size )
			{
				printf("GPU BUFFER OVERFLOW : %lu > \n", (send_buffer_disp + local_particle_node_sets[recv_id].size()) * sizeof(uint64_t), gpu_send_buffers_node_index_array_size );

			}

			gpuErrchk(cudaMemcpyAsync(&gpu_send_buffers_interp_node_indexes[send_buffer_disp], &send_buffers_interp_node_indexes[send_buffer_disp], local_particle_node_sets[recv_id].size()*sizeof(uint64_t), cudaMemcpyHostToDevice, (cudaStream_t) 0));
			C_kernel_pack_flow_field_buffer(block_count, thread_count, &gpu_send_buffers_interp_node_indexes[send_buffer_disp] , gpu_phi_nodes, &gpu_send_buffers_interp_node_flow_fields[send_buffer_disp], gpu_node_map, local_particle_node_sets[recv_id].size(), global_node_to_local_node_map.size());
			// gpuErrchk(cudaMemcpyAsync(&send_buffers_interp_node_flow_fields[send_buffer_disp], &gpu_send_buffers_interp_node_flow_fields[send_buffer_disp], local_particle_node_sets[recv_id].size()*sizeof(flow_aos<T>), cudaMemcpyDeviceToHost, (cudaStream_t) 0));
			
			gpuErrchk( cudaPeekAtLastError() );

			send_buffer_disp += local_particle_node_sets[recv_id].size();

		}
		
        unordered_neighbours_set[0].erase(MESH_BOUNDARY);
    }

    template<typename T> void FlowSolver<T>::interpolate_to_nodes ()
    {
		/*Interpolate values from particle side to flow grid*/
        const uint64_t cell_size = mesh->cell_size;
        double node_neighbours   = 8;

        // Process the allocation of cell fields (NOTE: Imperfect solution near edges. Fix by doing interpolation on flow side.)
        // #pragma ivdep
        for ( uint64_t cell : unordered_neighbours_set[0] )
        {
            const uint64_t block_cell      = cell - mesh->local_cells_disp;
            const uint64_t shmem_cell      = cell - mesh->shmem_cell_disp;

            flow_aos<T> flow_term;      
            flow_aos<T> flow_grad_term; 

			//TODO: Now all our terms work we should probably sort this out for all the versions

            if (is_halo(cell)) 
            {
                // flow_term.temp          = mesh->dummy_gas_tem;      
                // flow_grad_term.temp     = 0.0;  

                // flow_term.pressure      = phi.P[boundary_map[cell]];      
                // flow_grad_term.pressure = 0.0; 

                // flow_term.vel.x         = phi.U[boundary_map[cell]]; 
                // flow_term.vel.y         = phi.V[boundary_map[cell]]; 
                // flow_term.vel.z         = phi.W[boundary_map[cell]]; 
                // flow_grad_term.vel = { 0.0, 0.0, 0.0 }; 

				continue;
            }
            else
            {

                flow_term.vel.x    = phi.U[block_cell];
                flow_term.vel.y    = phi.V[block_cell];
                flow_term.vel.z    = phi.W[block_cell];
                flow_term.pressure = phi.P[block_cell];
                flow_term.temp     = phi.TEM[block_cell];

                flow_grad_term = mesh->flow_grad_terms[block_cell]; 
            }

            // cout << "vel " << print_vec(flow_term.vel) << " pressure " << flow_term.pressure << " temperature " << flow_term.temp << endl;


            const vec<T> cell_centre         = mesh->cell_centers[shmem_cell];

            // check_flow_field_exit ( "INTERP NODAL ERROR: Flow value",      &flow_term,      &mesh->dummy_flow_field,      cell );
            // check_flow_field_exit ( "INTERP NODAL ERROR: Flow grad value", &flow_grad_term, &mesh->dummy_flow_field_grad, cell );

            // if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL ERROR: Flow value",      &flow_term,      &mesh->dummy_flow_field,      cell );
            // if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL ERROR: Flow grad value", &flow_grad_term, &mesh->dummy_flow_field_grad, cell );

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id = mesh->cells[shmem_cell*mesh->cell_size + n];

                if (node_to_position_map.count(node_id))
                {
                    const vec<T> direction      = mesh->points[node_id - mesh->shmem_point_disp] - cell_centre;

                    // interp_node_flow_fields[node_to_position_map[node_id]].temp     += (flow_term.temp     + dot_product(flow_grad_term.temp,     direction)) / node_neighbours;
                    // interp_node_flow_fields[node_to_position_map[node_id]].pressure += (flow_term.pressure + dot_product(flow_grad_term.pressure, direction)) / node_neighbours;
                    // interp_node_flow_fields[node_to_position_map[node_id]].vel      += (flow_term.vel      + dot_product(flow_grad_term.vel,      direction)) / node_neighbours;

					interp_node_flow_fields[node_to_position_map[node_id]]     = mesh->dummy_flow_field;
                }
            }
        }
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
		cudaMemset(gpu_particle_terms, 0, sizeof(particle_aos<T>)*mesh->local_mesh_size);
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
        MPI_Ibcast(&recvs_complete, 1, MPI_INT, 0, mpi_config->world, &bcast_request);       
 
        int message_waiting = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);

		bool first_msg_recv        = false;
        bool  all_processed        = false;
        bool *processed_neighbours = async_locks;
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

                resize_cell_particle(elements[rank_slot], rank_slot);
                if ( FLOW_SOLVER_DEBUG ) printf("\tFlow block %d: Recieving %d indexes from %d (slot %lu). Max element size %lu. neighbour index rank size %ld array_pointer %p \n", mpi_config->particle_flow_rank, elements[rank_slot], ranks.back(), rank_slot, cell_index_array_size[rank_slot] / sizeof(uint64_t), neighbour_indexes.size(), neighbour_indexes[rank_slot]);

                logger.recieved_cells += elements[rank_slot];

                MPI_Irecv(neighbour_indexes[rank_slot], elements[rank_slot], MPI_UINT64_T,                       ranks[rank_slot], 0, mpi_config->world, &recv_requests[2*rank_slot]     );
                MPI_Irecv(cell_particle_aos[rank_slot], elements[rank_slot], mpi_config->MPI_PARTICLE_STRUCTURE, ranks[rank_slot], 2, mpi_config->world, &recv_requests[2*rank_slot + 1] );

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

                    neighbour_indexes.push_back((uint64_t*)         malloc(cell_index_array_size.back()));
                    cell_particle_aos.push_back((particle_aos<T> * )malloc(cell_particle_array_size.back()));

					uint64_t *cuda_pointer_tmp;
					particle_aos<T> *cuda_pointer_tmp2;
					cudaMalloc(&cuda_pointer_tmp,  cell_index_array_size.back());
					cudaMalloc(&cuda_pointer_tmp2, cell_particle_array_size.back());
					gpu_neighbour_indexes.push_back(cuda_pointer_tmp);
					gpu_cell_particle_aos.push_back(cuda_pointer_tmp2);

                    local_particle_node_sets.push_back(unordered_set<uint64_t>());
                }
                message_waiting = 0;
                MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);
                continue;
			}

            time0 += MPI_Wtime(); //1
            time1 -= MPI_Wtime(); //1
            
            all_processed = true;
            for ( uint64_t p = 0; p < ranks.size(); p++ )
            {
                int recieved_indexes = 0;
                MPI_Test(&recv_requests[2*p], &recieved_indexes, MPI_STATUS_IGNORE);

                if ( recieved_indexes && !processed_neighbours[p] )  // Invalid read
                {
                    if ( FLOW_SOLVER_DEBUG ) printf("\tFlow block %d: Processing %d indexes from %d. Local set size %lu (%lu of %lu sets)\n", mpi_config->particle_flow_rank, elements[p], ranks[p], local_particle_node_sets[p].size(), p, local_particle_node_sets.size());
                    
                    get_neighbour_cells (p);
                    processed_neighbours[p] = true;  // Invalid write
                }
                all_processed &= processed_neighbours[p]; //Invalid read
            }

            time1 += MPI_Wtime(); //1
            time2 -= MPI_Wtime(); //1

            MPI_Test ( &bcast_request, &recvs_complete, MPI_STATUS_IGNORE );
            MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);

            if ( FLOW_SOLVER_DEBUG && recvs_complete ) if(recvs_complete) printf("\tFlow block %d: Recieved broadcast signal. message_waiting %d recvs_complete %d all_processed %d\n", mpi_config->particle_flow_rank, message_waiting, recvs_complete, all_processed);
            all_processed = all_processed & !message_waiting & recvs_complete;
			time2 += MPI_Wtime(); //1
        }
		if (first_msg_recv)
		{
			nvtxRangePop();
		}
		nvtxRangePush("update_flow::finish_memcpy");


        logger.reduced_recieved_cells += new_cells_set.size();

        if ( FLOW_SOLVER_DEBUG ) printf("\tFlow Rank %d: Recieved index sizes.\n", mpi_config->rank);

        uint64_t max_send_buffer_size = 0;
        for (uint64_t p = 0; p < ranks.size(); p++)
        {
            max_send_buffer_size += local_particle_node_sets[p].size();
        }
		printf("Flow rank requires %lu items in buffer\n", max_send_buffer_size);

		// Synchronize with default stream to make sure phi data is on CPU
		cudaStreamSynchronize(0);
		nvtxRangePop();
		nvtxRangePush("update_flow::interpolate_to_nodes");
        interpolate_to_nodes ();
		nvtxRangePop();

		// Send size of reduced neighbours of cells back to ranks.
        uint64_t neighbour_point_size = node_to_position_map.size();

        logger.sent_nodes += neighbour_point_size;

		nvtxRangePush("update_flow::pack_and_post_buffers");

        uint64_t ptr_disp = 0;
        bool *processed_cell_fields = async_locks;
        for (uint64_t p = 0; p < ranks.size(); p++)
        {
            uint64_t local_disp = 0;

			gpuErrchk(cudaMemcpy(&check_send_buffers_interp_node_flow_fields[ptr_disp], &gpu_send_buffers_interp_node_flow_fields[ptr_disp], local_particle_node_sets[p].size()*sizeof(flow_aos<T>), cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();

            #pragma ivdep
            for ( uint64_t node : local_particle_node_sets[p] )
            // for ( uint64_t i = 0; i < local_particle_node_sets[p].size(); i++ )
            {
				// uint64_t node = send_buffers_interp_node_indexes[ptr_disp + local_disp];


                send_buffers_interp_node_indexes[ptr_disp     + local_disp] = interp_node_indexes[node_to_position_map[node]];
                send_buffers_interp_node_flow_fields[ptr_disp + local_disp] = interp_node_flow_fields[node_to_position_map[node]];
				// send_buffers_interp_node_indexes[ptr_disp     + local_disp] = node;


                // send_buffers_interp_node_flow_fields[ptr_disp + local_disp].vel.x    = phi_nodes.U[global_node_to_local_node_map[node]];
                // send_buffers_interp_node_flow_fields[ptr_disp + local_disp].vel.y    = phi_nodes.V[global_node_to_local_node_map[node]];
                // send_buffers_interp_node_flow_fields[ptr_disp + local_disp].vel.z    = phi_nodes.W[global_node_to_local_node_map[node]];
                // send_buffers_interp_node_flow_fields[ptr_disp + local_disp].pressure = phi_nodes.P[global_node_to_local_node_map[node]];
                // send_buffers_interp_node_flow_fields[ptr_disp + local_disp].temp     = phi_nodes.TEM[global_node_to_local_node_map[node]];

				// check_flow_field_exit("CPU_GPU_BUFFER WRONG", &check_send_buffers_interp_node_flow_fields[ptr_disp + local_disp], &send_buffers_interp_node_flow_fields[ptr_disp + local_disp], ptr_disp+local_disp);

                local_disp++;
            }

			// // gpuErrchk(cudaMemcpy(&send_buffers_interp_node_indexes[ptr_disp],     &gpu_send_buffers_interp_node_indexes[ptr_disp],     local_particle_node_sets[p].size()*sizeof(uint64_t),    cudaMemcpyDeviceToHost));
			// // gpuErrchk(cudaMemcpy(&send_buffers_interp_node_flow_fields[ptr_disp], &gpu_send_buffers_interp_node_flow_fields[ptr_disp], local_particle_node_sets[p].size()*sizeof(flow_aos<T>), cudaMemcpyDeviceToHost));
			

			MPI_Isend ( &send_buffers_interp_node_indexes[ptr_disp],      local_particle_node_sets[p].size(), MPI_UINT64_T,                   ranks[p], 0, mpi_config->world, &send_requests[p] );
            MPI_Isend ( &send_buffers_interp_node_flow_fields[ptr_disp],  local_particle_node_sets[p].size(), mpi_config->MPI_FLOW_STRUCTURE, ranks[p], 1, mpi_config->world, &send_requests[p + ranks.size()] );
            // MPI_Isend ( &check_send_buffers_interp_node_flow_fields[ptr_disp], local_particle_node_sets[p].size(), mpi_config->MPI_FLOW_STRUCTURE, ranks[p], 1, mpi_config->world, &send_requests[p + ranks.size()] );

            // MPI_Isend ( &gpu_send_buffers_interp_node_indexes[ptr_disp],     local_particle_node_sets[p].size(), MPI_UINT64_T,                   ranks[p], 0, mpi_config->world, &send_requests[p] );
            // MPI_Isend ( &gpu_send_buffers_interp_node_flow_fields[ptr_disp], local_particle_node_sets[p].size(), mpi_config->MPI_FLOW_STRUCTURE, ranks[p], 1, mpi_config->world, &send_requests[p + ranks.size()] );
            
			ptr_disp += local_particle_node_sets[p].size();

            processed_cell_fields[p] = false; // Invalid write

            recv_time2  += MPI_Wtime();
        }
		nvtxRangePop();
        
        recv_time3  -= MPI_Wtime();

        if ( FLOW_SOLVER_DEBUG && mpi_config->particle_flow_rank == 0 )  printf("\tFlow Rank %d: Posted sends.\n", mpi_config->rank);

        all_processed = false;
        while ( !all_processed )
        {
            all_processed = true;

            for ( uint64_t p = 0; p < ranks.size(); p++ )
            {
				//printf("p is %lu\n",p);
                int recieved_indexes = 0;
                MPI_Test(&recv_requests[2*p + 1], &recieved_indexes, MPI_STATUS_IGNORE);

                if ( recieved_indexes && !processed_neighbours[p] )
                {
            		gpuErrchk(cudaMemcpyAsync(gpu_neighbour_indexes[p], neighbour_indexes[p], elements[p]*sizeof(uint64_t),        cudaMemcpyHostToDevice, (cudaStream_t) 0));
            		gpuErrchk(cudaMemcpyAsync(gpu_cell_particle_aos[p], cell_particle_aos[p], elements[p]*sizeof(particle_aos<T>), cudaMemcpyHostToDevice, (cudaStream_t) 0));
					

					int thread_count = min(32, max(1, elements[p]));
					int block_count = max(1,(int) ceil((double) elements[p] / (double) thread_count));

					C_kernel_process_particle_fields(block_count, thread_count, gpu_neighbour_indexes[p], gpu_cell_particle_aos[p], gpu_particle_terms, elements[p], mesh->local_cells_disp);
					gpuErrchk( cudaPeekAtLastError() );

                    // if ( FLOW_SOLVER_DEBUG )  printf("\tFlow block %d: Processing %d cell fields from %d .\n", mpi_config->particle_flow_rank, elements[p], ranks[p]);
                    // for (int i = 0; i < elements[p]; i++)
                    // {
                    //     mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].momentum += cell_particle_aos[p][i].momentum;
                    //     mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].energy   += cell_particle_aos[p][i].energy;
                    //     mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].fuel     += cell_particle_aos[p][i].fuel;
                    // }
					//printf("p is %lu\n",p);

                    processed_cell_fields[p] = true;
                }
				//printf("p is %lu\n",p);
                all_processed &= processed_cell_fields[p];
            }
            if ( FLOW_SOLVER_DEBUG && all_processed )  printf("\tFlow block %d: all_processed %d\n", mpi_config->particle_flow_rank, all_processed);
        }

		// memcpys back to GPU.
		// gpuErrchk(cudaMemcpy(gpu_particle_terms, mesh->particle_terms,
        //                mesh->local_mesh_size * sizeof(particle_aos<T>),
        //                cudaMemcpyHostToDevice));

        MPI_Barrier(mpi_config->particle_flow_world);

		nvtxRangePop();
        recv_time3 += MPI_Wtime();

        // MPI_Waitall(send_requests.size() - 2, send_requests.data(), MPI_STATUSES_IGNORE); // Check field values later on!

        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Processed cell particle fields .\n", mpi_config->rank);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //6

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //7

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //8

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
                printf("\nUpdate Flow Field Communuication Timings\n");

                for (int i = 0; i < time_count; i++)
                    total_time += time_stats[i];
                for (int i = 0; i < time_count; i++)
                    printf("Time stats %d: %.3f (%.2f %%)\n", i, time_stats[i]  / mpi_config->particle_flow_world_size, 100 * time_stats[i] / total_time);
                printf("Total time %f\n", total_time / mpi_config->particle_flow_world_size);

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
		C_kernel_get_phi_gradients(block_count, thread_count, gpu_phi, gpu_phi_grad, mesh->local_mesh_size, mesh->local_cells_disp, mesh->faces_per_cell, (gpu_Face<uint64_t> *) gpu_faces, gpu_cell_faces, gpu_cell_centers, mesh->mesh_size, gpu_boundary_map, gpu_boundary_map_values, boundary_map.size(), gpu_face_centers, nhalos);
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
		
		int thread_count = min((uint64_t) 32,mesh->faces_size);
		int block_count = max(1,(int) ceil((double) mesh->faces_size/ (double) thread_count));
		
		C_kernel_calculate_flux_UVW(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, mesh->local_mesh_size, boundary_map.size(), gpu_boundary_map, gpu_boundary_map_values, gpu_phi_grad, gpu_cell_centers, gpu_face_centers, gpu_phi, gpu_A_phi, gpu_face_mass_fluxes, gpu_face_lambdas, gpu_face_normals, (gpu_Face<T> *) gpu_face_fields, gpu_S_phi, nhalos, gpu_boundary_types, mesh->dummy_gas_vel, effective_viscosity, gpu_face_rlencos, inlet_effective_viscosity, gpu_face_areas);
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
		C_kernel_setup_sparse_matrix(block_count, thread_count, UVW_URFactor, mesh->local_mesh_size, rows_ptr, col_indices, mesh->local_cells_disp, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_map, gpu_boundary_map_values, gpu_A_phi.U, (gpu_Face<T> *) gpu_face_fields, values, gpu_S_phi.U, gpu_phi.U, mesh->mesh_size, mesh->faces_per_cell, gpu_cell_faces, nnz);
		gpuErrchk( cudaPeekAtLastError() );
		
		//wait for matrix values
		int cpu_nnz = mesh->local_mesh_size*(mesh->faces_per_cell+1);
		gpuErrchk(cudaMemcpy(nnz, &cpu_nnz, sizeof(int),
		                  cudaMemcpyHostToDevice));
		for(int i = 0; i < mpi_config->particle_flow_world_size; i++){
			if(mpi_config->particle_flow_rank == i){
				//C_kernel_test_values(nnz, values, rows_ptr, col_indices, mesh->local_mesh_size, gpu_S_phi.U, gpu_phi.U);
			}
			MPI_Barrier (mpi_config->particle_flow_world);
		}
		MPI_Barrier (mpi_config->particle_flow_world);
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

	/*		for(int i = 0; i < mesh->mesh_size; i++)
            {
                printf("%d, ",partition_vector[i]);
            }
            printf("\n");
	*/		
			first_mat = false;
			AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(A, mesh->mesh_size, mesh->local_mesh_size, cpu_nnz, 1, 1, rows_ptr, col_indices, values, NULL, 1, 1, partition_vector));

			/*for(int i = 0; i < mesh->mesh_size; i++)
			{
				printf("%d, ",partition_vector[i]);
			}
			printf("\n");	
	*/
			AMGX_SAFE_CALL(AMGX_vector_bind(u, A));
			AMGX_SAFE_CALL(AMGX_vector_bind(b, A));
			
			AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		}
		else
		{
			AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
			AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		}

		AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.U));
        AMGX_SAFE_CALL(AMGX_vector_set_zero(u, (mesh->local_mesh_size + nhalos), 1));

		AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, u));
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
		
		C_kernel_update_sparse_matrix(block_count, thread_count, UVW_URFactor, mesh->local_mesh_size, gpu_A_phi.V, values, rows_ptr, gpu_S_phi.V, gpu_phi.V, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_map, gpu_boundary_map_values, mesh->local_cells_disp, (gpu_Face<double> *) gpu_face_fields, mesh->mesh_size);
		gpuErrchk( cudaPeekAtLastError() );

		//wait for matrix values
		gpuErrchk(cudaDeviceSynchronize());

		AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
		AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size +  nhalos), 1, gpu_S_phi.V));
		AMGX_SAFE_CALL(AMGX_vector_set_zero(u, (mesh->local_mesh_size + nhalos), 1));
		AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		
		AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, u));
		AMGX_SAFE_CALL(AMGX_vector_download(u, gpu_phi.V));

		//Solve for W
		C_kernel_update_sparse_matrix(block_count, thread_count, UVW_URFactor, mesh->local_mesh_size, gpu_A_phi.W, values, rows_ptr, gpu_S_phi.W, gpu_phi.W, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_map, gpu_boundary_map_values, mesh->local_cells_disp, (gpu_Face<double> *) gpu_face_fields, mesh->mesh_size);
		gpuErrchk( cudaPeekAtLastError() );

		//wait for matrix values
		gpuErrchk(cudaDeviceSynchronize());

		AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
	    	AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.W));
		AMGX_SAFE_CALL(AMGX_vector_set_zero(u, (mesh->local_mesh_size + nhalos), 1));
	    	AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
		
		AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, u));
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
		C_kernel_get_phi_gradient(block_count, thread_count, phi_component, mesh->local_mesh_size, mesh->local_cells_disp, mesh->faces_per_cell, (gpu_Face<uint64_t> *) gpu_faces, gpu_cell_faces, gpu_cell_centers, mesh->mesh_size, gpu_boundary_map, gpu_boundary_map_values, boundary_map.size(), gpu_face_centers, nhalos, phi_grad_component);
		gpuErrchk( cudaPeekAtLastError() );
		
	}

	template<typename T> void FlowSolver<T>::calculate_mass_flux()
	{
		int thread_count = min((uint64_t) 32,mesh->faces_size);
		int block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));

		C_kernel_calculate_mass_flux(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, mesh->local_mesh_size, boundary_map.size(), gpu_boundary_map, gpu_boundary_map_values, gpu_phi_grad, gpu_cell_centers, gpu_face_centers, gpu_phi, gpu_cell_densities, gpu_A_phi, gpu_cell_volumes, gpu_face_mass_fluxes, gpu_face_lambdas, gpu_face_normals, gpu_face_areas, (gpu_Face<T> *) gpu_face_fields, gpu_S_phi, nhalos, gpu_boundary_types, mesh->dummy_gas_vel);
		
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

	template<typename T> void FlowSolver<T>::calculate_pressure()
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

		calculate_mass_flux();
		//wait for flux happens in C kernel.

		//C_kernel_print(gpu_S_phi.U, mesh->local_mesh_size);

		//gpuErrchk(cudaDeviceSynchronize());	

		gpuErrchk(cudaMemset(nnz, 0, sizeof(int)));
        gpuErrchk(cudaMemset(values, 0.0, sizeof(T) * (mesh->local_mesh_size*7)));
        gpuErrchk(cudaMemset(col_indices, 0, sizeof(int64_t) * (mesh->local_mesh_size*7)));
        gpuErrchk(cudaMemset(rows_ptr, 0, sizeof(int) * (mesh->local_mesh_size+1)));

		int thread_count = min((uint64_t) 32,mesh->local_mesh_size);
        int block_count = max(1,(int) ceil((double) mesh->local_mesh_size/(double) thread_count));

		C_kernel_setup_pressure_matrix(block_count, thread_count, mesh->local_mesh_size, rows_ptr, col_indices, mesh->local_cells_disp, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_map, gpu_boundary_map_values, (gpu_Face<T> *) gpu_face_fields, values, mesh->mesh_size, mesh->faces_per_cell, gpu_cell_faces, nnz, gpu_face_mass_fluxes, gpu_A_phi, gpu_S_phi);
		gpuErrchk( cudaPeekAtLastError() );
	
		//wait for pressure matrix
		int cpu_nnz = mesh->local_mesh_size*(mesh->faces_per_cell+1);
		//int cpu_nnz = 0;
        //gpuErrchk(cudaMemcpy(nnz, &cpu_nnz, sizeof(int),
        //            cudaMemcpyHostToDevice));
		
		if(first_press)
		{
			first_press = false;
			AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(pressure_A, mesh->mesh_size, mesh->local_mesh_size, cpu_nnz, 1, 1, rows_ptr, col_indices, values, NULL, 1, 1, partition_vector));

			free(partition_vector);

			AMGX_SAFE_CALL(AMGX_vector_bind(pressure_u, pressure_A));
			AMGX_SAFE_CALL(AMGX_vector_bind(pressure_b, pressure_A));
	
		}
		else
		{
			AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(pressure_A, mesh->local_mesh_size, cpu_nnz, values, NULL));
		}
		AMGX_SAFE_CALL(AMGX_solver_setup(pressure_solver, pressure_A));

		thread_count = min((uint64_t) 32,mesh->faces_size);
		block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));
	
		while(Loop_continue)
		{
			Loop_num++;
			cpu_Pressure_correction_max = 0.0;
			
			//Solve first pressure update
			AMGX_SAFE_CALL(AMGX_vector_upload(pressure_b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.U));
			AMGX_SAFE_CALL(AMGX_vector_set_zero(pressure_u, (mesh->local_mesh_size + nhalos), 1));
		
			AMGX_SAFE_CALL(AMGX_solver_solve(pressure_solver, pressure_b, pressure_u));
			AMGX_SAFE_CALL(AMGX_vector_download(pressure_u, gpu_phi.PP));

			C_kernel_find_pressure_correction_max(1, 1, &cpu_Pressure_correction_max, gpu_phi.PP, mesh->local_mesh_size);
			//printf("max is %f\n",cpu_Pressure_correction_max);
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

			C_kernel_update_vel_and_flux(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->local_mesh_size, nhalos, (gpu_Face<T> *) gpu_face_fields, mesh->mesh_size, boundary_map.size(), gpu_boundary_map, gpu_boundary_map_values, gpu_face_mass_fluxes, gpu_A_phi, gpu_phi, gpu_cell_volumes, gpu_phi_grad, timestep_count);
		gpuErrchk( cudaPeekAtLastError() );


			gpuErrchk(cudaMemset(gpu_S_phi.U, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));

			C_kernel_update_mass_flux(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->local_mesh_size, mesh->mesh_size, gpu_boundary_map, gpu_boundary_map_values, boundary_map.size(), gpu_face_centers, gpu_cell_centers, (gpu_Face<double> *) gpu_face_fields, gpu_phi_grad, gpu_face_mass_fluxes, gpu_S_phi, gpu_face_normals);
		gpuErrchk( cudaPeekAtLastError() );

			gpuErrchk(cudaMemset(gpu_phi.PP, 0.0, (mesh->local_mesh_size + nhalos + mesh->boundary_cells_size) * sizeof(T)));
			if(Loop_num >= 4 or (cpu_Pressure_correction_max <= (0.25 * Pressure_correction_ref))) Loop_continue = false;
		}
		C_kernel_Update_P_at_boundaries(block_count, thread_count, mesh->faces_size, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, mesh->local_mesh_size, nhalos, gpu_phi.P);

		
		get_phi_gradient(gpu_phi.P, gpu_phi_grad.P);
		//wait done in C kernel.
		
		C_kernel_Update_P(block_count, thread_count, mesh->faces_size, mesh->local_mesh_size, nhalos, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, gpu_cell_centers, gpu_face_centers, gpu_boundary_types, gpu_phi.P, gpu_phi_grad.P);

		gpuErrchk( cudaPeekAtLastError() );
	}

	template<typename T> void FlowSolver<T>::Scalar_solve(int type, T *phi_component, vec<T> *phi_grad_component)
	{
		gpuErrchk(cudaMemset(gpu_A_phi.V, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));
		gpuErrchk(cudaMemset(gpu_S_phi.U, 0.0, (mesh->local_mesh_size + nhalos) * sizeof(T)));

		int thread_count = min((uint64_t) 32,mesh->faces_size);
		int block_count = max(1,(int) ceil((double) mesh->faces_size/(double) thread_count));
		
		//TODO;If not one of our types exit
		C_kernel_flux_scalar(block_count, thread_count, type, mesh->faces_size, mesh->local_mesh_size, nhalos, (gpu_Face<uint64_t> *) gpu_faces, mesh->local_cells_disp, mesh->mesh_size, gpu_cell_centers, gpu_face_centers, gpu_boundary_types, gpu_boundary_map, gpu_boundary_map_values, boundary_map.size(), phi_component, gpu_A_phi, gpu_S_phi, phi_grad_component, gpu_face_lambdas, effective_viscosity, gpu_face_rlencos, gpu_face_mass_fluxes, gpu_face_normals, inlet_effective_viscosity, (gpu_Face<T> *) gpu_face_fields, mesh->dummy_gas_vel, mesh->dummy_gas_tem, mesh->dummy_gas_fuel);
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

		C_kernel_setup_sparse_matrix(block_count, thread_count, URFactor, mesh->local_mesh_size, rows_ptr, col_indices, mesh->local_cells_disp, (gpu_Face<uint64_t> *) gpu_faces, boundary_map.size(), gpu_boundary_map, gpu_boundary_map_values, gpu_A_phi.V, (gpu_Face<double> *) gpu_face_fields, values, gpu_S_phi.U, phi_component, mesh->mesh_size, mesh->faces_per_cell, gpu_cell_faces, nnz);
		gpuErrchk( cudaPeekAtLastError() );
	

		int cpu_nnz = mesh->local_mesh_size*(mesh->faces_per_cell+1);
		//int cpu_nnz = 0;
        //gpuErrchk(cudaMemcpy(&cpu_nnz, nnz, sizeof(int),
        //            cudaMemcpyDeviceToHost));

		AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, mesh->local_mesh_size, cpu_nnz, values, NULL));
	    AMGX_SAFE_CALL(AMGX_vector_upload(b, (mesh->local_mesh_size + nhalos), 1, gpu_S_phi.U));
		AMGX_SAFE_CALL(AMGX_vector_set_zero(u, (mesh->local_mesh_size + nhalos), 1));
	    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));

		AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, u));
		AMGX_SAFE_CALL(AMGX_vector_download(u, phi_component));
	}

	template<typename T> void FlowSolver<T>::set_up_field()
    {
        /*We need inital values for mass_flux and AU for the first iteration*/
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function set_up_field.\n", mpi_config->rank);

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

		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk( cudaPeekAtLastError() );

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
            
            printf("Flow Solver Stats:\t                            AVG       MIN       MAX\n");
            printf("\tReduced Recieved Cells ( per rank ) : %9.0f %9.0f %9.0f\n", round(logger.reduced_recieved_cells / timesteps), round(min_red_cells / timesteps), round(max_red_cells / timesteps));
            printf("\tRecieved Cells ( per rank )         : %9.0f %9.0f %9.0f\n", round(logger.recieved_cells / timesteps), round(min_cells / timesteps), round(max_cells / timesteps));
            printf("\tSent Nodes     ( per rank )         : %9.0f %9.0f %9.0f\n", round(logger.sent_nodes     / timesteps), round(min_nodes / timesteps), round(max_nodes / timesteps));
            printf("\tFlow blocks with <1%% max droplets  : %d\n", mpi_config->particle_flow_world_size - (int)non_zero_blocks); 
            printf("\tAvg Cells with droplets             : %.2f%%\n", 100 * total_cells_recieved / (timesteps * mesh->mesh_size));
            printf("\tCell copies across particle ranks   : %.2f%%\n", 100.*(1 - total_reduced_cells_recieves / total_cells_recieved ));

            
            MPI_Barrier (mpi_config->particle_flow_world);

            // printf("\tFlow Rank %4d: Recieved Cells %7.0f Sent Nodes %7.0f\n", mpi_config->particle_flow_rank, round(loggers[mpi_config->particle_flow_rank].recieved_cells / timesteps), round(loggers[mpi_config->particle_flow_rank].sent_nodes / timesteps));

            // MPI_Barrier (mpi_config->particle_flow_world);
            cout << endl;
        }
        else
        {
            MPI_Barrier (mpi_config->particle_flow_world);
            // printf("\tFlow Rank %4d: Recieved Cells %7.0f Sent Nodes %7.0f\n", mpi_config->particle_flow_rank, round(logger.recieved_cells / timesteps), round(logger.sent_nodes / timesteps));
            // MPI_Barrier (mpi_config->particle_flow_world);
        }

        MPI_Barrier(mpi_config->world);


        performance_logger.print_counters(mpi_config->rank, mpi_config->world_size, runtime);

    }

    template<typename T> void FlowSolver<T>::timestep()
    {
		nvtxRangePush(__FUNCTION__);
		/*High level function to advance the flow solver one timestep.*/
        if (FLOW_SOLVER_DEBUG)  printf("\tFlow Rank %d: Start flow timestep.\n", mpi_config->rank);

        int comms_timestep = 1;
		if ( mpi_config->particle_flow_rank == 0 )
			printf("\ntimestep %lu\n",timestep_count + 1);
		
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
                printf("Timestep %6lu Flow     mem (TOTAL %8.3f GB)" 
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

		nvtxRangePush("get_phi_gradients");

		flow_timings[0] -= MPI_Wtime();
		get_phi_gradients();
		flow_timings[0] += MPI_Wtime();
		nvtxRangePop();

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

		
		if ((timestep_count % comms_timestep) == 0)
		{
			int thread_count1 = min((uint64_t) 32, mesh->points_size);
			int block_count1 = max(1,(int) ceil((double) mesh->points_size/(double) thread_count1));

			int thread_count = min((uint64_t) 32, (mesh->local_mesh_size + nhalos));
			int block_count = max(1,(int) ceil((double) (mesh->local_mesh_size + nhalos)/(double) thread_count));
			// int thread_count = min((uint64_t) 32, (mesh->local_mesh_size));
			// int block_count = max(1,(int) ceil((double) (mesh->local_mesh_size)/(double) thread_count));

  			C_kernel_interpolate_init_boundaries(block_count1, thread_count1, gpu_phi_nodes, gpu_cells_per_point, global_node_to_local_node_map.size());
			gpuErrchk( cudaPeekAtLastError() );
			C_kernel_interpolate_phi_to_nodes(block_count, thread_count, gpu_phi, gpu_phi_grad, gpu_phi_nodes, gpu_local_nodes, gpu_node_map, gpu_cells_per_point, gpu_local_cells, gpu_cell_centers, mesh->local_mesh_size, mesh->local_cells_disp, global_node_to_local_node_map.size(), nhalos);
			gpuErrchk( cudaPeekAtLastError() );

			gpuErrchk(cudaMemcpyAsync(phi.U, gpu_phi.U,     phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0)); // Asynchronous memcpy to default stream
            gpuErrchk(cudaMemcpyAsync(phi.V, gpu_phi.V,     phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
            gpuErrchk(cudaMemcpyAsync(phi.W, gpu_phi.W,     phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
            gpuErrchk(cudaMemcpyAsync(phi.P, gpu_phi.P,     phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
            gpuErrchk(cudaMemcpyAsync(phi.TEM, gpu_phi.TEM, phi_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));  
			gpuErrchk( cudaPeekAtLastError() );


			gpuErrchk(cudaMemcpyAsync(phi_nodes.U,   gpu_phi_nodes.U,   phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0)); // Asynchronous memcpy to default stream
            gpuErrchk(cudaMemcpyAsync(phi_nodes.V,   gpu_phi_nodes.V,   phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
            gpuErrchk(cudaMemcpyAsync(phi_nodes.W,   gpu_phi_nodes.W,   phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
            gpuErrchk(cudaMemcpyAsync(phi_nodes.P,   gpu_phi_nodes.P,   phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));
            gpuErrchk(cudaMemcpyAsync(phi_nodes.TEM, gpu_phi_nodes.TEM, phi_nodes_array_size, cudaMemcpyDeviceToHost, (cudaStream_t) 0));  
            update_flow_field();
			
		}

		flow_timings[1] += MPI_Wtime();
		compute_time -= MPI_Wtime();

		flow_timings[2] -= MPI_Wtime();
		nvtxRangePush("calculate_UVW");

		calculate_UVW();
		nvtxRangePop();
		flow_timings[2] += MPI_Wtime();

		nvtxRangePush("exchange_phi_halos");


		exchange_phi_halos(); //exchange new UVW values.
		nvtxRangePop();

		//printf("We finish pressure\n");
		MPI_Barrier(mpi_config->particle_flow_world);
/*
		if(((timestep_count + 1) % 1) == 0)
        {
            gpuErrchk(cudaMemcpy(phi.U, gpu_phi.U, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.V, gpu_phi.V, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.W, gpu_phi.W, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.P, gpu_phi.P, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.TE, gpu_phi.TE, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.ED, gpu_phi.ED, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.TEM, gpu_phi.TEM, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.FUL, gpu_phi.FUL, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.PRO, gpu_phi.PRO, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.VARF, gpu_phi.VARF, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.VARP, gpu_phi.VARP, phi_array_size,
                       cudaMemcpyDeviceToHost));
            if(mpi_config->particle_flow_rank == 0)
            {
                printf("Result is:\n");
            }
            for(int i = 0; i < mpi_config->particle_flow_world_size; i++)
            {
                if(i == mpi_config->particle_flow_rank)
                {
                    for(uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
                    {
                        const uint64_t cell = block_cell + mesh->local_cells_disp;
                        printf("locate (%4.18f,%4.18f,%4.18f)\n", mesh->cell_centers[cell-mesh->shmem_cell_disp].x,mesh->cell_centers[cell-mesh->shmem_cell_disp].y,mesh->cell_centers[cell-mesh->shmem_cell_disp].z);
                        printf("Variables are pressure %4.18f \nvel (%4.18f,%4.18f,%4.18f) \nTerb (%4.18f,%4.18f) \ntemerature %4.18f fuel mix %4.18f \nand progression %.6f\n var mix %4.18f\n var pro %4.18f\n\n", phi.P[block_cell], phi.U[block_cell], phi.V[block_cell], phi.W[block_cell], phi.TE[block_cell], phi.ED[block_cell], phi.TEM[block_cell], phi.FUL[block_cell], phi.PRO[block_cell], phi.VARF[block_cell], phi.VARP[block_cell]);
                    }
                }
                MPI_Barrier(mpi_config->particle_flow_world);
            }
        }
*/

		nvtxRangePush("calculate_pressure");

		flow_timings[3] -= MPI_Wtime();
		calculate_pressure(); 
		flow_timings[3] += MPI_Wtime();

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



	/*	if(((timestep_count + 1) % 1) == 0)
		{
			gpuErrchk(cudaMemcpy(phi.U, gpu_phi.U, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.V, gpu_phi.V, phi_array_size,
                       cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(phi.W, gpu_phi.W, phi_array_size,
                       cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(phi.P, gpu_phi.P, phi_array_size,
                       cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(phi.TE, gpu_phi.TE, phi_array_size,
                       cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(phi.ED, gpu_phi.ED, phi_array_size,
                       cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(phi.TEM, gpu_phi.TEM, phi_array_size,
                       cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(phi.FUL, gpu_phi.FUL, phi_array_size,
                       cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(phi.PRO, gpu_phi.PRO, phi_array_size,
                       cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(phi.VARF, gpu_phi.VARF, phi_array_size,
                       cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(phi.VARP, gpu_phi.VARP, phi_array_size,
                       cudaMemcpyDeviceToHost));
			if(mpi_config->particle_flow_rank == 0)
			{
				printf("Result is:\n");
			}
			for(int i = 0; i < mpi_config->particle_flow_world_size; i++)
			{
				if(i == mpi_config->particle_flow_rank)
				{
					for(uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
					{
						const uint64_t cell = block_cell + mesh->local_cells_disp;
						printf("locate (%4.18f,%4.18f,%4.18f)\n", mesh->cell_centers[cell-mesh->shmem_cell_disp].x,mesh->cell_centers[cell-mesh->shmem_cell_disp].y,mesh->cell_centers[cell-mesh->shmem_cell_disp].z);
						printf("Variables are pressure %4.18f \nvel (%4.18f,%4.18f,%4.18f) \nTerb (%4.18f,%4.18f) \ntemerature %4.18f fuel mix %4.18f \nand progression %.6f\n var mix %4.18f\n var pro %4.18f\n\n", phi.P[block_cell], phi.U[block_cell], phi.V[block_cell], phi.W[block_cell], phi.TE[block_cell], phi.ED[block_cell], phi.TEM[block_cell], phi.FUL[block_cell], phi.PRO[block_cell], phi.VARF[block_cell], phi.VARP[block_cell]);
					}
				}
				MPI_Barrier(mpi_config->particle_flow_world);
			}
		}
*/
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

                printf("\nTotal Flow Solver Timings:\n");
                printf("Compute Velocity time: %.3f (%.2f %%)\n",
                        vel_total_time / mpi_config->particle_flow_world_size,
                        100 * vel_total_time / total_time);
                printf("Compute Pressure time: %.3f (%.2f %%)\n",
                        pres_total_time / mpi_config->particle_flow_world_size,
                        100 * pres_total_time / total_time);
				printf("Compute 8 Scalars time %.3f (%.2f %%)\n",
						scalar_time / mpi_config->particle_flow_world_size,
						100 * scalar_time / total_time);
				printf("FGM Table Lookup time %.3f (%.2f %%)\n",
						fgm_lookup_time / mpi_config->particle_flow_world_size,
						100 * fgm_lookup_time / total_time);
                printf("Total time: %f\n\n", 
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
                printf("\nFlow Timing: \nCalc gradients: %f\nCalc update particles: %f\nCalc velocity: %f\nCalc Pressure: %f\nCalc Turb TE: %f\nCalc Turb ED: %f\nCalc Heat: %f\nCalc PROG: %f\nCalc FUEL: %f\nCalc VAR PROG: %f\nCalc VAR FUEL: %f\n",flow_timings[0],flow_timings[1],flow_timings[2],flow_timings[3],flow_timings[4],flow_timings[5],flow_timings[6],flow_timings[7],flow_timings[8],flow_timings[9],flow_timings[10]);
            }
            else
            {
                MPI_Reduce(flow_timings, nullptr, 11, MPI_DOUBLE, MPI_SUM,
                           0, mpi_config->particle_flow_world);
            }
        }		
        
		if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Stop flow timestep.\n", mpi_config->rank);
        timestep_count++;

		nvtxRangePop();

    }
}
