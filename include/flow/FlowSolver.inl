#include <stdio.h>
#include <limits.h>

#include "flow/FlowSolver.hpp"

#define TERBTE 0
#define TERBED 1
#define TEMP 2
#define FUEL 3
#define PROG  4
#define VARFU 5
#define VARPR 6
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
		/*Pass a single phi value over the halos*/
		int num_requests = 1;
		
		MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
	
		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			MPI_Isend(phi_component, 1, halo_mpi_double_datatypes[r], halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
		}
		
		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			MPI_Irecv(&phi_component[mesh->local_mesh_size + halo_disps[r]], halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );	
		}
		
		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);	
	}

	template<typename T> void FlowSolver<T>::exchange_single_grad_halo(vec<T> *phi_grad_component)
    {
		/*Pass a single gradient vector over the halos*/
        int num_requests = 1;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
			MPI_Isend(phi_grad_component, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0]);
        }

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
			MPI_Irecv(&phi_grad_component[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0]);
        }

        MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

	template<typename T> void FlowSolver<T>::exchange_grad_halos()
	{
		/*Group exchange of most phi gradient vectors over the halos*/
		int num_requests = 9;
		
		MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        
		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
		{
			MPI_Isend( phi_grad.U,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            MPI_Isend( phi_grad.V,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
            MPI_Isend( phi_grad.W,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 2, mpi_config->particle_flow_world, &send_requests[num_requests*r + 2] );
            MPI_Isend( phi_grad.P,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 3, mpi_config->particle_flow_world, &send_requests[num_requests*r + 3] );
            MPI_Isend( phi_grad.TE,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 4, mpi_config->particle_flow_world, &send_requests[num_requests*r + 4] );
            MPI_Isend( phi_grad.ED,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 5, mpi_config->particle_flow_world, &send_requests[num_requests*r + 5] );
            MPI_Isend( phi_grad.TEM, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 6, mpi_config->particle_flow_world, &send_requests[num_requests*r + 6] );
            MPI_Isend( phi_grad.FUL, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 7, mpi_config->particle_flow_world, &send_requests[num_requests*r + 7] );
            MPI_Isend( phi_grad.PRO, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 8, mpi_config->particle_flow_world, &send_requests[num_requests*r + 8] );
		}

		for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
			MPI_Irecv( &phi_grad.U[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            MPI_Irecv( &phi_grad.V[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 1] );
            MPI_Irecv( &phi_grad.W[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 2, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 2] );
            MPI_Irecv( &phi_grad.P[mesh->local_mesh_size + halo_disps[r]],   3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 3] );
            MPI_Irecv( &phi_grad.TE[mesh->local_mesh_size + halo_disps[r]],  3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 4, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 4] );
            MPI_Irecv( &phi_grad.ED[mesh->local_mesh_size + halo_disps[r]],  3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 5, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 5] );
            MPI_Irecv( &phi_grad.TEM[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 6, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 6] );
            MPI_Irecv( &phi_grad.FUL[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 7, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 7] );
            MPI_Irecv( &phi_grad.PRO[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 8, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 8] );
		}

		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
	}

    template<typename T> void FlowSolver<T>::exchange_phi_halos ()
    {
		/*Group exchange of most phi values over the halos*/
        int num_requests = 9;
        
		MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Isend( phi.U,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            MPI_Isend( phi.V,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
            MPI_Isend( phi.W,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 2, mpi_config->particle_flow_world, &send_requests[num_requests*r + 2] );
            MPI_Isend( phi.P,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 3, mpi_config->particle_flow_world, &send_requests[num_requests*r + 3] );
			MPI_Isend( phi.TE,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 4, mpi_config->particle_flow_world, &send_requests[num_requests*r + 4] );	
			MPI_Isend( phi.ED,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 5, mpi_config->particle_flow_world, &send_requests[num_requests*r + 5] );
			MPI_Isend( phi.TEM,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 6, mpi_config->particle_flow_world, &send_requests[num_requests*r + 6] );
			MPI_Isend( phi.FUL,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 7, mpi_config->particle_flow_world, &send_requests[num_requests*r + 7] ); 
			MPI_Isend( phi.PRO,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 8, mpi_config->particle_flow_world, &send_requests[num_requests*r + 8] );
		}

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &phi.U[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            MPI_Irecv( &phi.V[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 1] );
            MPI_Irecv( &phi.W[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 2, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 2] );
            MPI_Irecv( &phi.P[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 3] );
			MPI_Irecv( &phi.TE[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 4, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 4] );
			MPI_Irecv( &phi.ED[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 5, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 5] );
			MPI_Irecv( &phi.TEM[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 6, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 6] );
			MPI_Irecv( &phi.FUL[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 7, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 7] );
			MPI_Irecv( &phi.PRO[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 8, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 8] );
		}
		
		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::exchange_A_halos (T *A_phi_component)
    {
		/*Exchange a single A_phi value over the halos
		  Used to make sure 1/A values are consistent over processes*/
        int num_requests = 1;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Isend( A_phi_component,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
        }

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &A_phi_component[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
        }
		
		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::get_neighbour_cells ( const uint64_t recv_id )
    {
		/*Find all the cells which are neighbours of the current cell.
		  Used to find the halos*/
        double node_neighbours   = 8;
        const uint64_t cell_size = mesh->cell_size;

        resize_nodes_arrays(node_to_position_map.size() + elements[recv_id] * cell_size + 1 );

        #pragma ivdep
        for (int i = 0; i < elements[recv_id]; i++)
        {
            uint64_t cell = neighbour_indexes[recv_id][i];

            #pragma ivdep
            for (uint64_t n = 0; n < cell_size; n++)
            {
                const uint64_t node_id      = mesh->cells[(cell - mesh->shmem_cell_disp) * mesh->cell_size + n];
                local_particle_node_sets[recv_id].insert(node_id);
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

            if (is_halo(cell)) 
            {
                flow_term.temp          = mesh->dummy_gas_tem;      
                flow_grad_term.temp     = 0.0;  

                flow_term.pressure      = phi.P[boundary_map[cell]];      
                flow_grad_term.pressure = 0.0; 

                // if (!boundary_map.contains(cell)) exit(1);
                
                flow_term.vel.x         = phi.U[boundary_map[cell]]; 
                flow_term.vel.y         = phi.V[boundary_map[cell]]; 
                flow_term.vel.z         = phi.W[boundary_map[cell]]; 
                // flow_term.vel      = mesh->dummy_gas_vel; 
                flow_grad_term.vel = { 0.0, 0.0, 0.0 }; 
            }
            else
            {

                flow_term.vel.x    = phi.U[block_cell];
                flow_term.vel.y    = phi.V[block_cell];
                flow_term.vel.z    = phi.W[block_cell];
                flow_term.pressure = phi.P[block_cell];
                flow_term.temp     = mesh->flow_terms[block_cell].temp;

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

                    interp_node_flow_fields[node_to_position_map[node_id]].temp     += (flow_term.temp     + dot_product(flow_grad_term.temp,     direction)) / node_neighbours;
                    interp_node_flow_fields[node_to_position_map[node_id]].pressure += (flow_term.pressure + dot_product(flow_grad_term.pressure, direction)) / node_neighbours;
                    interp_node_flow_fields[node_to_position_map[node_id]].vel      += (flow_term.vel      + dot_product(flow_grad_term.vel,      direction)) / node_neighbours;
                }
            }
        }
    }

    template<typename T> void FlowSolver<T>::update_flow_field()
    {
		/*Collect and send cell values to particle solve and receive
		  values from particle solve.*/
        int time_count = 0;
        time_stats[time_count]  -= MPI_Wtime(); //0
        unordered_neighbours_set[0].clear();
        node_to_position_map.clear();
        new_cells_set.clear();
        ranks.clear();

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

        bool  all_processed        = false;
        bool *processed_neighbours = async_locks;
        while(!all_processed)
        {
            time0 -= MPI_Wtime(); //1
            if ( message_waiting )
            {
                uint64_t rank_slot = ranks.size();
                ranks.push_back(statuses[rank_slot].MPI_SOURCE);
                MPI_Get_count( &statuses[rank_slot], MPI_UINT64_T, &elements[rank_slot] );

                resize_cell_particle(elements[rank_slot], rank_slot);
                if ( FLOW_SOLVER_DEBUG ) printf("\tFlow block %d: Recieving %d indexes from %d (slot %lu). Max element size %lu. neighbour index rank size %ld array_pointer %p \n", mpi_config->particle_flow_rank, elements[rank_slot], ranks.back(), rank_slot, cell_index_array_size[rank_slot] / sizeof(uint64_t), neighbour_indexes.size(), neighbour_indexes[rank_slot]);

                logger.recieved_cells += elements[rank_slot];

                MPI_Irecv(neighbour_indexes[rank_slot], elements[rank_slot], MPI_UINT64_T,                       ranks[rank_slot], 0, mpi_config->world, &recv_requests[2*rank_slot]     );
                MPI_Irecv(cell_particle_aos[rank_slot], elements[rank_slot], mpi_config->MPI_PARTICLE_STRUCTURE, ranks[rank_slot], 2, mpi_config->world, &recv_requests[2*rank_slot + 1] );

                processed_neighbours[rank_slot] = false;

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

                if ( recieved_indexes && !processed_neighbours[p] )
                {
                    if ( FLOW_SOLVER_DEBUG ) printf("\tFlow block %d: Processing %d indexes from %d. Local set size %lu (%lu of %lu sets)\n", mpi_config->particle_flow_rank, elements[p], ranks[p], local_particle_node_sets[p].size(), p, local_particle_node_sets.size());
                    
                    get_neighbour_cells (p);
                    processed_neighbours[p] = true;

                }

                all_processed &= processed_neighbours[p];
            }

            time1 += MPI_Wtime(); //1
            time2 -= MPI_Wtime(); //1

            MPI_Test ( &bcast_request, &recvs_complete, MPI_STATUS_IGNORE );
            MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);

            if ( FLOW_SOLVER_DEBUG && recvs_complete ) if(recvs_complete) printf("\tFlow block %d: Recieved broadcast signal. message_waiting %d recvs_complete %d all_processed %d\n", mpi_config->particle_flow_rank, message_waiting, recvs_complete, all_processed);
            all_processed = all_processed & !message_waiting & recvs_complete;
			time2 += MPI_Wtime(); //1
        }

        logger.reduced_recieved_cells += new_cells_set.size();

        if ( FLOW_SOLVER_DEBUG ) printf("\tFlow Rank %d: Recieved index sizes.\n", mpi_config->rank);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //2


        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //3
        

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //4
        
        interpolate_to_nodes ();

        // Send size of reduced neighbours of cells back to ranks.
        uint64_t neighbour_point_size = node_to_position_map.size();

        logger.sent_nodes += neighbour_point_size;

        uint64_t max_send_buffer_size = 0;
        for (uint64_t p = 0; p < ranks.size(); p++)
        {
            max_send_buffer_size += local_particle_node_sets[p].size();
        }
        resize_send_buffers_nodes_arrays (max_send_buffer_size);

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //5

        uint64_t ptr_disp = 0;
        bool *processed_cell_fields = async_locks;
        for (uint64_t p = 0; p < ranks.size(); p++)
        {
            recv_time1  -= MPI_Wtime();
            uint64_t local_disp = 0;
            #pragma ivdep
            for ( uint64_t node : local_particle_node_sets[p] )
            {
                send_buffers_interp_node_indexes[ptr_disp     + local_disp] = interp_node_indexes[node_to_position_map[node]];
                send_buffers_interp_node_flow_fields[ptr_disp + local_disp] = interp_node_flow_fields[node_to_position_map[node]];
                local_disp++;
                
            }
            recv_time1  += MPI_Wtime();
            recv_time2  -= MPI_Wtime();

            MPI_Isend ( send_buffers_interp_node_indexes + ptr_disp,     local_disp, MPI_UINT64_T,                   ranks[p], 0, mpi_config->world, &send_requests[p] );
            MPI_Isend ( send_buffers_interp_node_flow_fields + ptr_disp, local_disp, mpi_config->MPI_FLOW_STRUCTURE, ranks[p], 1, mpi_config->world, &send_requests[p + ranks.size()] );
            ptr_disp += local_disp;

            processed_cell_fields[p] = false;

            recv_time2  += MPI_Wtime();
        }
        
        recv_time3  -= MPI_Wtime();

        if ( FLOW_SOLVER_DEBUG && mpi_config->particle_flow_rank == 0 )  printf("\tFlow Rank %d: Posted sends.\n", mpi_config->rank);

        all_processed = false;
        while ( !all_processed )
        {
            all_processed = true;

            for ( uint64_t p = 0; p < ranks.size(); p++ )
            {
                int recieved_indexes = 0;
                MPI_Test(&recv_requests[2*p + 1], &recieved_indexes, MPI_STATUS_IGNORE);

                if ( recieved_indexes && !processed_neighbours[p] )
                {
                    if ( FLOW_SOLVER_DEBUG )  printf("\tFlow block %d: Processing %d cell fields from %d .\n", mpi_config->particle_flow_rank, elements[p], ranks[p]);
                    for (int i = 0; i < elements[p]; i++)
                    {
                        mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].momentum += cell_particle_aos[p][i].momentum;
                        mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].energy   += cell_particle_aos[p][i].energy;
                        mesh->particle_terms[neighbour_indexes[p][i] - mesh->local_cells_disp].fuel     += cell_particle_aos[p][i].fuel;
                    }

                    processed_cell_fields[p] = true;
                }
                all_processed &= processed_cell_fields[p];
            }
            if ( FLOW_SOLVER_DEBUG && all_processed )  printf("\tFlow block %d: all_processed %d\n", mpi_config->particle_flow_rank, all_processed);
        }

        recv_time3 += MPI_Wtime();

        MPI_Waitall(send_requests.size() - 2, send_requests.data(), MPI_STATUSES_IGNORE); // Check field values later on!

        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Processed cell particle fields .\n", mpi_config->rank);

        MPI_Barrier(mpi_config->particle_flow_world);
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

	template<typename T> void FlowSolver<T>::get_phi_gradient ( T *phi_component, vec<T> *phi_grad_component, bool pressure )
	{
		/*Use the Least Squares method to find the gradient of a phi component*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function get_phi_gradient.\n", mpi_config->rank);

        for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
        {
            const uint64_t cell = block_cell + mesh->local_cells_disp;
			MatZeroEntries(grad_A);
			VecZeroEntries(grad_b);
            for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
            {
                const uint64_t face  = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];

                const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
                const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

                const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;
				T dPhi;
                vec<T> dX;
                if ( mesh->faces[face].cell1 < mesh->mesh_size )  // Inner cell
                {
                    const uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                    const uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
                    const T mask = ( mesh->faces[face].cell0 == cell ) ? 1. : -1.;

                    dPhi = mask * ( phi_component[phi_index1] - phi_component[phi_index0] );
                    dX = mask * ( mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0] );
                    // Note: ADD code for porous cells here
                }
                else // Boundary face
                {
                    const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
					
					if(pressure)
					{
						dPhi = 0.0; //dolfyn inforces this after the first calc of pressure
					}
					else
					{
						dPhi = phi_component[mesh->local_mesh_size + nhalos + boundary_cell] - phi_component[block_cell0];
                    }
					dX = face_centers[face] - mesh->cell_centers[shmem_cell0];
                }
			
				MatSetValue(grad_A, 0, 0, (dX.x * dX.x), ADD_VALUES);
				MatSetValue(grad_A, 1, 0, (dX.x * dX.y), ADD_VALUES);
				MatSetValue(grad_A, 2, 0, (dX.x * dX.z), ADD_VALUES);

				MatSetValue(grad_A, 0, 1, (dX.y * dX.x), ADD_VALUES);
				MatSetValue(grad_A, 1, 1, (dX.y * dX.y), ADD_VALUES);
				MatSetValue(grad_A, 2, 1, (dX.y * dX.z) ,ADD_VALUES);

				MatSetValue(grad_A, 0, 2, (dX.z * dX.x), ADD_VALUES);
				MatSetValue(grad_A, 1, 2, (dX.z * dX.y), ADD_VALUES);
				MatSetValue(grad_A, 2, 2, (dX.z * dX.z), ADD_VALUES);

				VecSetValue(grad_b, 0, (dX.x * dPhi), ADD_VALUES);
				VecSetValue(grad_b, 1, (dX.y * dPhi), ADD_VALUES);
				VecSetValue(grad_b, 2, (dX.z * dPhi), ADD_VALUES);
            }

			T time = 0;
		
			time -=	MPI_Wtime(); 
			MatAssemblyBegin(grad_A, MAT_FINAL_ASSEMBLY);
			MatAssemblyEnd(grad_A, MAT_FINAL_ASSEMBLY);

			VecAssemblyBegin(grad_b);
			VecAssemblyEnd(grad_b);

			KSPSolve(grad_ksp, grad_b, grad_u);
			PetscInt indx[3] = {0, 1, 2};
			VecGetValues(grad_u, 3, indx, &phi_grad_component[block_cell].x);
			time += MPI_Wtime();
        }
	}

	template<typename T> void FlowSolver<T>::limit_phi_gradient(T *phi_component, vec<T> *phi_grad_component)
	{
		/*Use the Venkatakrishnan method for gradient limiting from
          On the accuracy of limiters and convergence to steady state solutions,
          V.Venkatakrishnan, 1993.*/
		T dmax, dmin;
        T deltamax, deltamin;

        for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
        {
            const uint64_t cell = block_cell + mesh->local_cells_disp;
            const uint64_t shmem_cell = cell - mesh->shmem_cell_disp;

            dmax = phi_component[block_cell];
            dmin = phi_component[block_cell];
            for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
            {
				const uint64_t face  = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];
                const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
                const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

                if ( mesh->faces[face].cell1 < mesh->mesh_size )  // Inner cell
                {
                    const uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                    const uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
                    if(mesh->faces[face].cell0 == cell)
                    {
                        dmax = max(dmax,phi_component[phi_index1]);
                        dmin = min(dmin,phi_component[phi_index1]);
                    }
                    else
                    {
                        dmax = max(dmax,phi_component[phi_index0]);
                        dmin = min(dmin,phi_component[phi_index0]);
                    }
                }
                else //Boundary cell
                {
                    const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                    dmax = max(dmax,phi_component[mesh->local_mesh_size + nhalos + boundary_cell]);
                    dmin = min(dmin,phi_component[mesh->local_mesh_size + nhalos + boundary_cell]);
                }
            }
			deltamax = dmax - phi_component[block_cell];
            deltamin = dmin - phi_component[block_cell];

            T alpha= 1.0;
            T r = 0.0;
            vec<T> ds = {0.0, 0.0, 0.0};
            for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
            {
                const uint64_t face  = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];
                ds = face_centers[face] - mesh->cell_centers[shmem_cell];
                T delta_face = dot_product(phi_grad_component[block_cell],ds);

                if(abs(delta_face) < 0.000006)
                {
                    r = 1000.0;
                }
                else if(delta_face > 0.0)
                {
                    r = deltamax/delta_face;
                }
                else
                {
                    r = deltamin/delta_face;
                }
				alpha = min(alpha, (pow(r,2) + 2.0 * r)/(pow(r,2) + r + 2.0));
            }
            phi_grad_component[block_cell] = alpha * phi_grad_component[block_cell];
        }
	}

	template<typename T> void FlowSolver<T>::limit_phi_gradients ()
    {
		/*High level function to limit the gradient of variables*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function limit_phi_gradients.\n", mpi_config->rank);
		printf("coming here\n");
		limit_phi_gradient(phi.U, phi_grad.U);
		limit_phi_gradient(phi.V, phi_grad.V);
		limit_phi_gradient(phi.W, phi_grad.W);
		limit_phi_gradient(phi.P, phi_grad.P);
		limit_phi_gradient(phi.TE, phi_grad.TE);
		limit_phi_gradient(phi.ED, phi_grad.ED);
		limit_phi_gradient(phi.TEM, phi_grad.TEM);
		limit_phi_gradient(phi.FUL, phi_grad.FUL);
		limit_phi_gradient(phi.PRO, phi_grad.PRO);
		limit_phi_gradient(phi.VARF, phi_grad.VARF);
		limit_phi_gradient(phi.VARP, phi_grad.VARP);
	}

    template<typename T> void FlowSolver<T>::get_phi_gradients ()
    {
		/*Use the Least Squares method to find the gradient of most phi components*/
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function get_phi_gradients.\n", mpi_config->rank);
        // NOTE: Currently Least squares is the only method supported

        for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
        {
            const uint64_t cell = block_cell + mesh->local_cells_disp;
            MatZeroEntries(grad_A);
            VecZeroEntries(grad_bU);
			VecZeroEntries(grad_bV);
			VecZeroEntries(grad_bW);
			VecZeroEntries(grad_bP);
			VecZeroEntries(grad_bTE);
			VecZeroEntries(grad_bED);
			VecZeroEntries(grad_bT);
			VecZeroEntries(grad_bFU);
			VecZeroEntries(grad_bPR);
			VecZeroEntries(grad_bVFU);
			VecZeroEntries(grad_bVPR);

            for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
            {
                const uint64_t face  = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];
                
				const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
                const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

                const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;
				
                T dU, dV, dW, dP, dTE, dED, dT, dFU, dPR, dVFU, dVPR;
                vec<T> dX;
                if ( mesh->faces[face].cell1 < mesh->mesh_size )  // Inner cell
                {
                    const uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                    const uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
                    const T mask = ( mesh->faces[face].cell0 == cell ) ? 1. : -1.;

                    dU =   mask * ( phi.U[phi_index1]   - phi.U[phi_index0] );
                    dV =   mask * ( phi.V[phi_index1]   - phi.V[phi_index0] );
                    dW =   mask * ( phi.W[phi_index1]   - phi.W[phi_index0] );
                    dP =   mask * ( phi.P[phi_index1]   - phi.P[phi_index0] );
					dTE =  mask * ( phi.TE[phi_index1]  - phi.TE[phi_index0] );
					dED =  mask * ( phi.ED[phi_index1]  - phi.ED[phi_index0] );
					dT =   mask * ( phi.TEM[phi_index1] - phi.TEM[phi_index0] );
					dFU =  mask * ( phi.FUL[phi_index1] - phi.FUL[phi_index0] );
					dPR =  mask * ( phi.PRO[phi_index1] - phi.PRO[phi_index0] );
					dVFU = mask * (phi.VARF[phi_index1] - phi.VARF[phi_index0] );
					dVPR = mask * (phi.VARP[phi_index1] - phi.VARP[phi_index0] );
                    
					dX = mask * ( mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0] );
                    // Note: ADD code for porous cells here
                } 
                else // Boundary face
                {
					const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;

                    dU = phi.U[mesh->local_mesh_size + nhalos + boundary_cell] - phi.U[block_cell0];
                    dV = phi.V[mesh->local_mesh_size + nhalos + boundary_cell] - phi.V[block_cell0];
                    dW = phi.W[mesh->local_mesh_size + nhalos + boundary_cell] - phi.W[block_cell0];
                    dP = 0.0;//dolfyn also enforces dp = 0.0 over boundary
					dTE = phi.TE[mesh->local_mesh_size + nhalos + boundary_cell] - phi.TE[block_cell0];
					dED = phi.ED[mesh->local_mesh_size + nhalos + boundary_cell] - phi.ED[block_cell0];
					dT = phi.TEM[mesh->local_mesh_size + nhalos + boundary_cell] - phi.TEM[block_cell0];
					dFU = phi.FUL[mesh->local_mesh_size + nhalos + boundary_cell] - phi.FUL[block_cell0];
					dPR = phi.PRO[mesh->local_mesh_size + nhalos + boundary_cell] - phi.PRO[block_cell0];
					dVFU = phi.VARF[mesh->local_mesh_size + nhalos + boundary_cell] - phi.VARF[block_cell0];
					dVPR = phi.VARP[mesh->local_mesh_size + nhalos + boundary_cell] - phi.VARP[block_cell0];

                    dX = face_centers[face] - mesh->cell_centers[shmem_cell0];
					//NOTE: This is required to prevent really small dX due to machine precision.
					if(abs(dX.x) < 0.0000000000000003)
					{
						dX.x = 0.0;
					}
					if(abs(dX.y) < 0.0000000000000003)
                    {
                        dX.y = 0.0;
                    }
					if(abs(dX.z) < 0.0000000000000003)
                    {
                        dX.z = 0.0;
                    }	
                }
				
				MatSetValue(grad_A, 0, 0, (dX.x * dX.x), ADD_VALUES);
                MatSetValue(grad_A, 1, 0, (dX.x * dX.y), ADD_VALUES);
                MatSetValue(grad_A, 2, 0, (dX.x * dX.z), ADD_VALUES);

                MatSetValue(grad_A, 0, 1, (dX.y * dX.x), ADD_VALUES);
                MatSetValue(grad_A, 1, 1, (dX.y * dX.y), ADD_VALUES);
                MatSetValue(grad_A, 2, 1, (dX.y * dX.z) ,ADD_VALUES);

                MatSetValue(grad_A, 0, 2, (dX.z * dX.x), ADD_VALUES);
                MatSetValue(grad_A, 1, 2, (dX.z * dX.y), ADD_VALUES);
                MatSetValue(grad_A, 2, 2, (dX.z * dX.z), ADD_VALUES);	

                VecSetValue(grad_bU, 0, (dX.x * dU), ADD_VALUES);
                VecSetValue(grad_bU, 1, (dX.y * dU), ADD_VALUES);
                VecSetValue(grad_bU, 2, (dX.z * dU), ADD_VALUES);

				VecSetValue(grad_bV, 0, (dX.x * dV), ADD_VALUES);
                VecSetValue(grad_bV, 1, (dX.y * dV), ADD_VALUES);
                VecSetValue(grad_bV, 2, (dX.z * dV), ADD_VALUES);
		
				VecSetValue(grad_bW, 0, (dX.x * dW), ADD_VALUES);
                VecSetValue(grad_bW, 1, (dX.y * dW), ADD_VALUES);
                VecSetValue(grad_bW, 2, (dX.z * dW), ADD_VALUES);

				VecSetValue(grad_bP, 0, (dX.x * dP), ADD_VALUES);
                VecSetValue(grad_bP, 1, (dX.y * dP), ADD_VALUES);
                VecSetValue(grad_bP, 2, (dX.z * dP), ADD_VALUES);

				VecSetValue(grad_bTE, 0, (dX.x * dTE), ADD_VALUES);
                VecSetValue(grad_bTE, 1, (dX.y * dTE), ADD_VALUES);
                VecSetValue(grad_bTE, 2, (dX.z * dTE), ADD_VALUES);

				VecSetValue(grad_bED, 0, (dX.x * dED), ADD_VALUES);
                VecSetValue(grad_bED, 1, (dX.y * dED), ADD_VALUES);
                VecSetValue(grad_bED, 2, (dX.z * dED), ADD_VALUES);

				VecSetValue(grad_bT, 0, (dX.x * dT), ADD_VALUES);
                VecSetValue(grad_bT, 1, (dX.y * dT), ADD_VALUES);
                VecSetValue(grad_bT, 2, (dX.z * dT), ADD_VALUES);

				VecSetValue(grad_bFU, 0, (dX.x * dFU), ADD_VALUES);
                VecSetValue(grad_bFU, 1, (dX.y * dFU), ADD_VALUES);
                VecSetValue(grad_bFU, 2, (dX.z * dFU), ADD_VALUES);

				VecSetValue(grad_bPR, 0, (dX.x * dPR), ADD_VALUES);
                VecSetValue(grad_bPR, 1, (dX.y * dPR), ADD_VALUES);
                VecSetValue(grad_bPR, 2, (dX.z * dPR), ADD_VALUES);

				VecSetValue(grad_bVFU, 0, (dX.x * dVFU), ADD_VALUES);
                VecSetValue(grad_bVFU, 1, (dX.y * dVFU), ADD_VALUES);
                VecSetValue(grad_bVFU, 2, (dX.z * dVFU), ADD_VALUES);

				VecSetValue(grad_bVPR, 0, (dX.x * dVPR), ADD_VALUES);
                VecSetValue(grad_bVPR, 1, (dX.y * dVPR), ADD_VALUES);
                VecSetValue(grad_bVPR, 2, (dX.z * dVPR), ADD_VALUES);
            }


			MatAssemblyBegin(grad_A, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(grad_A, MAT_FINAL_ASSEMBLY);

			VecAssemblyBegin(grad_bU);
            VecAssemblyEnd(grad_bU);
		
			VecAssemblyBegin(grad_bV);
            VecAssemblyEnd(grad_bV);

			VecAssemblyBegin(grad_bW);
            VecAssemblyEnd(grad_bW);

			VecAssemblyBegin(grad_bP);
            VecAssemblyEnd(grad_bP);

			VecAssemblyBegin(grad_bTE);
            VecAssemblyEnd(grad_bTE);

			VecAssemblyBegin(grad_bED);
            VecAssemblyEnd(grad_bED);

			VecAssemblyBegin(grad_bT);
            VecAssemblyEnd(grad_bT);

			VecAssemblyBegin(grad_bFU);
            VecAssemblyEnd(grad_bFU);

			VecAssemblyBegin(grad_bPR);
            VecAssemblyEnd(grad_bPR);

			VecAssemblyBegin(grad_bVFU);
            VecAssemblyEnd(grad_bVFU);

			VecAssemblyBegin(grad_bVPR);
            VecAssemblyEnd(grad_bVPR);

			PetscInt indx[3] = {0, 1, 2};

			KSPSolve(grad_ksp, grad_bU, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.U[block_cell].x);

			KSPSolve(grad_ksp, grad_bV, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.V[block_cell].x);

			KSPSolve(grad_ksp, grad_bW, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.W[block_cell].x);

			KSPSolve(grad_ksp, grad_bP, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.P[block_cell].x);

			KSPSolve(grad_ksp, grad_bTE, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.TE[block_cell].x);

			KSPSolve(grad_ksp, grad_bED, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.ED[block_cell].x);

			KSPSolve(grad_ksp, grad_bT, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.TEM[block_cell].x);

			KSPSolve(grad_ksp, grad_bFU, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.FUL[block_cell].x);

			KSPSolve(grad_ksp, grad_bPR, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.PRO[block_cell].x);

			KSPSolve(grad_ksp, grad_bVFU, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.VARF[block_cell].x);

			KSPSolve(grad_ksp, grad_bVPR, grad_u);
            VecGetValues(grad_u, 3, indx, &phi_grad.VARP[block_cell].x);
        }
    }

	template<typename T> void FlowSolver<T>::precomp_mass_flux()
	{
		/*we need the value of mass flux at the inlets and outlets to caculate AU*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function precomp_mass_flux.\n", mpi_config->rank);
		for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;

            if ( mesh->faces[face].cell1 < mesh->mesh_size ) continue;
			//only need the boundary cells
			const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
            const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
			if ( boundary_type == INLET )
            {
                // Constant inlet values for velocities and densities. Add custom regions laters
                const vec<T> vel_inward = mesh->dummy_gas_vel;
                const T Din = 1.2;
                face_mass_fluxes[face] = Din * dot_product( vel_inward, face_normals[face] );
            }
            else if( boundary_type == OUTLET )
            {
				const vec<T> vel_outward = { phi.U[block_cell0],
                                             phi.V[block_cell0],
                                             phi.W[block_cell0] };
                const T Din = 1.2;

                face_mass_fluxes[face] = Din * dot_product(vel_outward, face_normals[face]);
				// !
                // ! For an outlet face_mass_fluxes must be 0.0 or positive
                // !
                if( face_mass_fluxes[face] < 0.0 )
                {
                    printf("PRECOMP NEGATIVE OUTFLOW %3.18f\n", face_mass_fluxes[face]);
                    face_mass_fluxes[face] = 1e-15;

					phi.TE[mesh->local_mesh_size + nhalos + boundary_cell] = 
							phi.TE[block_cell0];
					phi.ED[mesh->local_mesh_size + nhalos + boundary_cell] =
							phi.ED[block_cell0];
					phi.TEM[mesh->local_mesh_size + nhalos + boundary_cell] =
							phi.TEM[block_cell0];
                }
            }
		}
		//Conserve mass if the flow has not reached the outflow yet.
        T FlowOut = 0.0;
        T FlowIn = 0.0;
        T areaout= 0.0;
        int count_out = 0;
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            if ( mesh->faces[face].cell1 < mesh->mesh_size )  continue;
            //Boundary only
            const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
            const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
            if(boundary_type == INLET)
            {
                FlowIn += face_mass_fluxes[face];
            }
            else if(boundary_type == OUTLET)
            {
                FlowOut += face_mass_fluxes[face];
                count_out++;
                areaout += face_areas[face];
            }
        }
        T FlowFact[count_out];
        int step = 0;
        for(int i = 0; i < count_out; i++){
            if(FlowOut == 0.0)
            {
                //This protects against NaN
                FlowFact[i] = 0.0;
            }
            else
            {
                FlowFact[i] = -FlowIn/FlowOut;
            }
        }
		MPI_Allreduce(MPI_IN_PLACE, &FlowIn, 1, MPI_DOUBLE, MPI_SUM, mpi_config->particle_flow_world);
        MPI_Allreduce(MPI_IN_PLACE, &FlowOut, 1, MPI_DOUBLE, MPI_SUM, mpi_config->particle_flow_world);
		if(FlowOut < 0.0000000001)
        {
            T ratearea = - FlowIn/areaout;
            FlowOut = 0.0;
            for ( uint64_t face = 0; face < mesh->faces_size; face++ )
            {
                if ( mesh->faces[face].cell1 < mesh->mesh_size )  continue;
                //Boundary only
                const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
                if(boundary_type == OUTLET)
                {
                    //NOTE: this assumes density is constant and uniform
					//NOTE: assumes one outflow region
                    face_mass_fluxes[face] = ratearea*face_areas[face];
                    T FaceFlux = face_mass_fluxes[face]/cell_densities[0]/face_areas[face];
                    
					phi.U[mesh->local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).x;
                    phi.V[mesh->local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).y;
                    phi.W[mesh->local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).z;

                    FlowOut += face_mass_fluxes[face];
                }
            }
        }
        T fact = -FlowIn/(FlowOut + 0.0000001);
        step = 0;
		for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            if ( mesh->faces[face].cell1 < mesh->mesh_size )  continue;
            //Boundary only
            const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
            const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
            if(boundary_type == OUTLET)
            {
                face_mass_fluxes[face] *= FlowFact[step];
                step ++;

                phi.U[mesh->local_mesh_size + nhalos + boundary_cell] *= fact;
                phi.V[mesh->local_mesh_size + nhalos + boundary_cell] *= fact;
                phi.W[mesh->local_mesh_size + nhalos + boundary_cell] *= fact;

                const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
				S_phi.U[block_cell0] -= face_mass_fluxes[face];
            }
        }
	}

	template<typename T> void FlowSolver<T>::precomp_AU()
	{
		/*Compute AU needed for the first calculation of mass flux*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function precomp_AU.\n", mpi_config->rank);	
		#pragma ivdep
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            A_phi.U[i] = 0.0;
        }
		for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            
			if ( mesh->faces[face].cell1 < mesh->mesh_size ) continue;
			//boundary only
			const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
            const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
			if ( boundary_type == INLET )
            {
				const T Visac = effective_viscosity;
				const T VisFace  = Visac * face_rlencos[face];
				const T f = -VisFace + min( face_mass_fluxes[face], 0.0 );
				A_phi.U[block_cell0] = A_phi.U[block_cell0] - f;
			}
			else if( boundary_type == OUTLET )
			{
				const T Visac = effective_viscosity;
				const T VisFace  = Visac * face_rlencos[face];
				const T f = -VisFace + min( face_mass_fluxes[face], 0.0 );
				A_phi.U[block_cell0] = A_phi.U[block_cell0] - f;
			}
		}
		const double rdelta = 1.0 / delta;
		#pragma ivdep
        for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++ )
        {
            double f = cell_densities[i] * cell_volumes[i] * rdelta;
            A_phi.U[i] += f;
        }
	}

	template<typename T> void FlowSolver<T>::set_up_field()
	{
		/*We need inital values for mass_flux and AU for the first iteration*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function set_up_field.\n", mpi_config->rank);
		precomp_AU();
		exchange_A_halos(A_phi.U);
		calculate_mass_flux();
	}

	template<typename T> void FlowSolver<T>::set_up_fgm_table()
	{
		/* Set up some data for the FGM look up table*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function set_up_fgm_table.\n", mpi_config->rank);
		//TODO: take into account to memory requirement of this table.
		//Should we reduce this table to 2-d and compute the PDFs of the two variables
		//to look up with??? I think not but why.
		srand( (unsigned)time( NULL ) );
		for(int i = 0; i < 100; i++)
		{
			for(int j = 0; j < 100; j++)
			{
				for(int k = 0; k < 0; k++)
				{
					for(int l = 0; l < 100; l++)
					{
						FGM_table[i][j][k][l] = (T) rand()/RAND_MAX;
					}
				}
			}
		}
	}
	
	template<typename T> void FlowSolver<T>::FGM_loop_up()
	{
		/*Look up the result in the FGM table for each local cell
          this would be the source term in the progress variable
          calculation*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function FGM_look_up.\n", mpi_config->rank);
		#pragma ivdep
		for(uint64_t i = 0; i < mesh->local_mesh_size; i++)
		{
			//find location in table based on variables
			//simulate finding the closest two points in the database
			int progress_1 = min(100, (int) floor(phi.PRO[i]*100));
			int progress_2 = min(100, (int) ceil(phi.PRO[i]*100));
			int var_progress_1 = min(100, (int) floor(phi.VARP[i]*100));
			int var_progress_2 = min(100, (int) ceil(phi.VARP[i]*100));
			int fuel_1 = min(100, (int) floor(phi.FUL[i]*100));
			int fuel_2 = min(100, (int) ceil(phi.FUL[i]*100));
			int var_fuel_1 = min(100, (int) floor(phi.VARF[i]*100));
			int var_fuel_2 = min(100, (int) ceil(phi.VARF[i]*100));	
			
			//interpolate table values to find given value
			//simulate using the average of the 16
			T sum = 0;
			sum += FGM_table[progress_1][var_progress_1][fuel_1][var_fuel_1];
			sum += FGM_table[progress_1][var_progress_1][fuel_1][var_fuel_2];
			sum += FGM_table[progress_1][var_progress_1][fuel_2][var_fuel_1];
			sum += FGM_table[progress_1][var_progress_1][fuel_2][var_fuel_2];
			sum += FGM_table[progress_1][var_progress_2][fuel_1][var_fuel_1];
			sum += FGM_table[progress_1][var_progress_2][fuel_1][var_fuel_2];
			sum += FGM_table[progress_1][var_progress_2][fuel_2][var_fuel_1];
			sum += FGM_table[progress_1][var_progress_2][fuel_2][var_fuel_2];
			sum += FGM_table[progress_2][var_progress_1][fuel_1][var_fuel_1];
			sum += FGM_table[progress_2][var_progress_1][fuel_1][var_fuel_2];
			sum += FGM_table[progress_2][var_progress_1][fuel_2][var_fuel_1];
			sum += FGM_table[progress_2][var_progress_1][fuel_2][var_fuel_2];
			sum += FGM_table[progress_2][var_progress_2][fuel_1][var_fuel_1];
			sum += FGM_table[progress_2][var_progress_2][fuel_1][var_fuel_2];
			sum += FGM_table[progress_2][var_progress_2][fuel_2][var_fuel_1];
			sum += FGM_table[progress_2][var_progress_2][fuel_2][var_fuel_2];
			//this would give us the value to be used as the source term in the 
			//progress calculation in our code this will be thrown away.
			sum /= 16;
			S_phi.U[i] = sum;
		}
	}	

    template<typename T> void FlowSolver<T>::setup_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component )
    {
		/*Set up A matrix and b vector using PETSc for a sparse linear solve*/
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function setup_sparse_matrix.\n", mpi_config->rank);

        T RURF = 1. / URFactor;

		MatZeroEntries(A);
		VecZeroEntries(b);

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if (mesh->faces[face].cell1 >= mesh->mesh_size)  continue; // Remove when implemented boundary cells. Treat boundary as mesh size

            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
			MatSetValue(A, mesh->faces[face].cell0, mesh->faces[face].cell1, face_fields[face].cell1, INSERT_VALUES);
			MatSetValue(A, mesh->faces[face].cell1, mesh->faces[face].cell0, face_fields[face].cell0, INSERT_VALUES); 
            
            
			A_phi_component[phi_index0] -= face_fields[face].cell1;
            A_phi_component[phi_index1] -= face_fields[face].cell0;

        }

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
            A_phi_component[i] *= RURF;
            S_phi_component[i] = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];

			MatSetValue(A, i+mesh->local_cells_disp, i+mesh->local_cells_disp, A_phi_component[i], INSERT_VALUES);
			VecSetValue(b, i+mesh->local_cells_disp, S_phi_component[i], INSERT_VALUES);

        }
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	
		VecAssemblyBegin(b);
		VecAssemblyEnd(b);
    }

    template<typename T> void FlowSolver<T>::update_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component )
    {
		/*This function is to reduce the number of insertions into
          the sparse A matrix, given that face_fields (RFace) is
          constant for U, V and W.*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function update_sparse_matrix.\n", mpi_config->rank);
        
		T RURF = 1. / URFactor;

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if ( mesh->faces[face].cell1 >= mesh->mesh_size )  continue; // Remove when implemented boundary cells. Treat boundary as mesh size

            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;


            A_phi_component[phi_index0] -= face_fields[face].cell1;
            A_phi_component[phi_index1] -= face_fields[face].cell0;
        }

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
			A_phi_component[i] *= RURF;
            S_phi_component[i] = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];
			MatSetValue(A, i+mesh->local_cells_disp, i+mesh->local_cells_disp, A_phi_component[i], INSERT_VALUES);
            VecSetValue(b, i+mesh->local_cells_disp, S_phi_component[i], INSERT_VALUES);
        }

		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

        VecAssemblyBegin(b);
        VecAssemblyEnd(b);
    }

    template<typename T> void FlowSolver<T>::solve_sparse_matrix ( T *phi_component)
    {
		/*Solve the linear system Au=b using PETSc*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_sparse_matrix.\n", mpi_config->rank);
        VecZeroEntries(u);

		KSPSolve(ksp, b, u);

		PetscInt indx[mesh->local_mesh_size];
		
		for(uint64_t i = 0; i < mesh->local_mesh_size; i++)
		{	
			indx[i] = i+mesh->local_cells_disp;
		}
		VecGetValues(u, mesh->local_mesh_size, indx, phi_component);
    }

    template<typename T> void FlowSolver<T>::calculate_flux_UVW()
    {
		/*Calculate the face based Velocity flux values for UVW*/
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_flux_UVW.\n", mpi_config->rank);

        T pe0 =  9999.;
        T pe1 = -9999.;

        T GammaBlend = 0.0; // NOTE: Change when implemented other differencing schemes.

        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

            if ( mesh->faces[face].cell1 < mesh->mesh_size )  // INTERNAL
            {
                uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
				
                const T lambda0 = face_lambdas[face];
                const T lambda1 = 1.0 - lambda0;
	
                const vec<T> dUdXac  =   phi_grad.U[phi_index0] * lambda0 + phi_grad.U[phi_index1] * lambda1;
                const vec<T> dVdXac  =   phi_grad.V[phi_index0] * lambda0 + phi_grad.V[phi_index1] * lambda1;
                const vec<T> dWdXac  =   phi_grad.W[phi_index0] * lambda0 + phi_grad.W[phi_index1] * lambda1;
				                
				T Visac   = effective_viscosity * lambda0 + effective_viscosity * lambda1;
                T VisFace = Visac * face_rlencos[face];
 
                vec<T> Xpn     = mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0];

                T UFace, VFace, WFace;
                if ( face_mass_fluxes[face] >= 0.0 )
                {
                    UFace  = phi.U[phi_index0];
                    VFace  = phi.V[phi_index0];
                    WFace  = phi.W[phi_index0];
                }
                else
                {
                    UFace  = phi.U[phi_index1];
                    VFace  = phi.V[phi_index1];
                    WFace  = phi.W[phi_index1];
                }

                // explicit higher order convective flux (see eg. eq. 8.16)

                const T fuce = face_mass_fluxes[face] * UFace;
                const T fvce = face_mass_fluxes[face] * VFace;
                const T fwce = face_mass_fluxes[face] * WFace;

                const T sx = face_normals[face].x;
                const T sy = face_normals[face].y;
                const T sz = face_normals[face].z;

                // explicit higher order diffusive flux based on simple uncorrected
                // interpolated cell centred gradients(see eg. eq. 8.19)
                const T fude = Visac * ((dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz);
                const T fvde = Visac * ((dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz);
                const T fwde = Visac * ((dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz);

                // ! implicit lower order (simple upwind)
                // ! convective and diffusive fluxes

                const T fmin = min( face_mass_fluxes[face], 0.0 );
                const T fmax = max( face_mass_fluxes[face], 0.0 );
		
                const T fuci = fmin * phi.U[phi_index0] + fmax * phi.U[phi_index1];
                const T fvci = fmin * phi.V[phi_index0] + fmax * phi.V[phi_index1];
                const T fwci = fmin * phi.W[phi_index0] + fmax * phi.W[phi_index1];

                const T fudi = VisFace * dot_product( dUdXac , Xpn );
                const T fvdi = VisFace * dot_product( dVdXac , Xpn );
                const T fwdi = VisFace * dot_product( dWdXac , Xpn );
                // !
                // ! convective coefficients with deferred correction with
                // ! gamma as the blending factor (0.0 <= gamma <= 1.0)
                // !
                // !      low            high    low  OLD
                // ! F = F    + gamma ( F     - F    )
                // !     ----   -------------------------
                // !      |                  |
                // !  implicit           explicit (dump into source term)
                // !
                // !            diffusion       convection
                // !                v               v
                face_fields[face].cell0 = -VisFace - max( face_mass_fluxes[face] , 0.0 );  // P (e);
                face_fields[face].cell1 = -VisFace + min( face_mass_fluxes[face] , 0.0 );  // N (w);
				const T blend_u = GammaBlend * ( fuce - fuci );
                const T blend_v = GammaBlend * ( fvce - fvci );
                const T blend_w = GammaBlend * ( fwce - fwci );

                // ! assemble the two source terms
                // ! Is it faster to just write to source term vectors?? Can we vectorize this function??
				S_phi.U[phi_index0] = S_phi.U[phi_index0] - blend_u + fude - fudi;
                S_phi.V[phi_index0] = S_phi.V[phi_index0] - blend_v + fvde - fvdi;
                S_phi.W[phi_index0] = S_phi.W[phi_index0] - blend_w + fwde - fwdi;

                S_phi.U[phi_index1] = S_phi.U[phi_index1] + blend_u - fude + fudi;
                S_phi.V[phi_index1] = S_phi.V[phi_index1] + blend_v - fvde + fvdi;
                S_phi.W[phi_index1] = S_phi.W[phi_index1] + blend_w - fwde + fwdi;

                const T small_epsilon = 1.e-20;
                const T peclet = face_mass_fluxes[face] / face_areas[face] * magnitude(Xpn) / (Visac+small_epsilon);
				
				pe0 = min( pe0 , peclet );
                pe1 = max( pe1 , peclet );

            }
            else // BOUNDARY
            {

                // Boundary faces
                const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
                if ( boundary_type == INLET )
                {   
                    // Option to add more inlet region information and functions here.
                    const vec<T> dUdXac = phi_grad.U[block_cell0];
                    const vec<T> dVdXac = phi_grad.V[block_cell0];
                    const vec<T> dWdXac = phi_grad.W[block_cell0];

                    const T UFace = mesh->dummy_gas_vel.x;
                    const T VFace = mesh->dummy_gas_vel.y;
                    const T WFace = mesh->dummy_gas_vel.z;

                    const T Visac = inlet_effective_viscosity;

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];
                    const T VisFace  = Visac * face_rlencos[face];

                    const T sx = face_normals[face].x;
                    const T sy = face_normals[face].y;
                    const T sz = face_normals[face].z;

                    const T fude = Visac * ((dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz);
                    const T fvde = Visac * ((dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz);
                    const T fwde = Visac * ((dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz);

                    const T fudi = VisFace * dot_product( dUdXac , Xpn );
                    const T fvdi = VisFace * dot_product( dVdXac , Xpn );
                    const T fwdi = VisFace * dot_product( dWdXac , Xpn );

                    // ! by definition points a boundary normal outwards
                    // ! therefore an inlet results in a mass flux < 0.0
					const T f = -VisFace + min( face_mass_fluxes[face], 0.0 );
				
					A_phi.U[block_cell0] = A_phi.U[block_cell0] - f;
                    S_phi.U[block_cell0] = S_phi.U[block_cell0] - f * UFace + fude - fudi;
                    phi.U[mesh->local_mesh_size + nhalos + boundary_cell] = UFace;

                    A_phi.V[block_cell0] = A_phi.V[block_cell0] - f;
                    S_phi.V[block_cell0] = S_phi.V[block_cell0] - f * VFace + fvde - fvdi;
                    phi.V[mesh->local_mesh_size + nhalos + boundary_cell] = VFace;

                    A_phi.W[block_cell0] = A_phi.W[block_cell0] - f;
                    S_phi.W[block_cell0] = S_phi.W[block_cell0] - f * WFace + fwde - fwdi;
                    phi.W[mesh->local_mesh_size + nhalos + boundary_cell] = WFace;

                }
                else if( boundary_type == OUTLET )
                {
                    const vec<T> dUdXac = phi_grad.U[block_cell0];
                    const vec<T> dVdXac = phi_grad.V[block_cell0];
                    const vec<T> dWdXac = phi_grad.W[block_cell0];

                    const T Visac = effective_viscosity;

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];

                    const T UFace = phi.U[block_cell0];
                    const T VFace = phi.V[block_cell0];
                    const T WFace = phi.W[block_cell0];

                    const T VisFace  = Visac * face_rlencos[face];

                    const T sx = face_normals[face].x;
                    const T sy = face_normals[face].y;
                    const T sz = face_normals[face].z;

                    const T fude = Visac * ((dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz);
                    const T fvde = Visac * ((dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz);
                    const T fwde = Visac * ((dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz);

                    const T fudi = VisFace * dot_product( dUdXac , Xpn );
                    const T fvdi = VisFace * dot_product( dVdXac , Xpn );
                    const T fwdi = VisFace * dot_product( dWdXac , Xpn );

                    // !
                    // ! by definition points a boundary normal outwards
                    // ! therefore an outlet results in a mass flux >= 0.0
                    // !

                    if( face_mass_fluxes[face] < 0.0 )
                    {
                        printf("MAIN COMP UVW NEGATIVE OUTFLOW %3.18f\n", face_mass_fluxes[face]);
                        face_mass_fluxes[face] = 1e-15;
                    }
                    
					const T f = -VisFace + min( face_mass_fluxes[face], 0.0 );
					
					A_phi.U[block_cell0] = A_phi.U[block_cell0] - f;
                    S_phi.U[block_cell0] = S_phi.U[block_cell0] - f * UFace + fude - fudi;
                    phi.U[mesh->local_mesh_size + nhalos + boundary_cell] = UFace;

                    A_phi.V[block_cell0] = A_phi.V[block_cell0] - f;
                    S_phi.V[block_cell0] = S_phi.V[block_cell0] - f * VFace + fvde - fvdi;
                    phi.V[mesh->local_mesh_size + nhalos + boundary_cell] = VFace;

                    A_phi.W[block_cell0] = A_phi.W[block_cell0] - f;
                    S_phi.W[block_cell0] = S_phi.W[block_cell0] - f * WFace + fwde - fwdi;
                    phi.W[mesh->local_mesh_size + nhalos + boundary_cell] = WFace;
                }
                else if( boundary_type == WALL )
                {
                    const T UFace = 0.; // Customisable (add regions here later)
                    const T VFace = 0.; // Customisable (add regions here later)
                    const T WFace = 0.; // Customisable (add regions here later)

                    const T Visac = effective_viscosity;

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];

                    const T coef = Visac * face_rlencos[face];

                    vec<T> Up;
                    Up.x = phi.U[block_cell0] - UFace;
                    Up.y = phi.V[block_cell0] - VFace;
                    Up.z = phi.W[block_cell0] - WFace;

                    const T dp = dot_product( Up , normalise(face_normals[face]));
                    vec<T> Ut  = Up - dp * normalise(face_normals[face]);

                    const T Uvel = abs(Ut.x) + abs(Ut.y) + abs(Ut.z);
                    
                    vec<T> force;
                    if ( Uvel > 0.0  )
                    {
                        const T distance_to_face = magnitude(Xpn); // TODO: Correct for different meshes
                        force = face_areas[face] * Visac * Ut / distance_to_face;
                    }
                    else
                    {
                        force = {0.0, 0.0, 0.0};
                    }

                    // TotalForce = TotalForce + Force

                    // !               standard
                    // !               implicit
                    // !                  V
					A_phi.U[block_cell0] = A_phi.U[block_cell0] + coef;
                    A_phi.V[block_cell0] = A_phi.V[block_cell0] + coef;
                    A_phi.W[block_cell0] = A_phi.W[block_cell0] + coef;

                    // !
                    // !                    corr.                     expliciet
                    // !                  impliciet
                    // !                     V                         V
                    S_phi.U[block_cell0] = S_phi.U[block_cell0] + coef*phi.U[block_cell0] - force.x;
                    S_phi.V[block_cell0] = S_phi.V[block_cell0] + coef*phi.V[block_cell0] - force.y;					
					S_phi.W[block_cell0] = S_phi.W[block_cell0] + coef*phi.W[block_cell0] - force.z;

                    phi.U[mesh->local_mesh_size + nhalos + boundary_cell] = UFace;
                    phi.V[mesh->local_mesh_size + nhalos + boundary_cell] = VFace;
                    phi.W[mesh->local_mesh_size + nhalos + boundary_cell] = WFace;
                }
            }   
        }
    }

    template<typename T> void FlowSolver<T>::calculate_UVW()
    {
		/*High level function to solve velocity using this procedure:
		  1. Find face based flux values
		  2. Account for buoyancy forces
		  3. Account for pressure forces
		  4. Account transent forces
		  5. Solve each velocity component*/
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_UVW.\n", mpi_config->rank);
		
        vel_total_time -= MPI_Wtime(); 

        // Initialise A_phi and S_phi vectors to 0.
        #pragma ivdep 
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++)
        {
            A_phi.U[i] = 0.0;
            A_phi.V[i] = 0.0;
            A_phi.W[i] = 0.0;

            S_phi.U[i] = 0.0;
            S_phi.V[i] = 0.0;
            S_phi.W[i] = 0.0;
        }

		//calculate fluxes through all inner faces
		vel_flux_time -= MPI_Wtime();
		calculate_flux_UVW();
		vel_flux_time += MPI_Wtime();

        // Gravity force (enthalpy)
		for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++)
		{
			T BodyForce = -0.001*cell_densities[i]*cell_volumes[i]*(phi.TEM[i] - 273);
			T gravity[3] = {0.0, -9.81, 0.0};
			
			S_phi.U[i] += gravity[0]*BodyForce;
			S_phi.V[i] += gravity[1]*BodyForce;
			S_phi.W[i] += gravity[2]*BodyForce;
		}
        
        // Pressure force
		for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++)
		{
			S_phi.U[i] -= phi_grad.P[i].x*cell_volumes[i];
			S_phi.V[i] -= phi_grad.P[i].y*cell_volumes[i];
			S_phi.W[i] -= phi_grad.P[i].z*cell_volumes[i];
		}

        // If Transient and Euler
        double rdelta = 1.0 / delta;
		#pragma ivdep
        for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++ )
        {
            double f = cell_densities[i] * cell_volumes[i] * rdelta;

            S_phi.U[i] += f * phi.U[i];
            S_phi.V[i] += f * phi.V[i];
            S_phi.W[i] += f * phi.W[i];

            A_phi.U[i] += f;
            A_phi.V[i] += f;
            A_phi.W[i] += f;
        }

		//RHS from particle code
		for(uint64_t i = 0; i < mesh->local_mesh_size; i++)
		{
			S_phi.U[i] += mesh->particle_terms[i].momentum.x;
			S_phi.V[i] += mesh->particle_terms[i].momentum.y;
			S_phi.W[i] += mesh->particle_terms[i].momentum.z;
		}

		const double UVW_URFactor = 0.5;
        
		vel_setup_time -= MPI_Wtime();
        setup_sparse_matrix(UVW_URFactor, A_phi.U, phi.U, S_phi.U);   
        vel_setup_time  += MPI_Wtime();

        vel_solve_time  -= MPI_Wtime();		
		solve_sparse_matrix(phi.U);
		vel_solve_time += MPI_Wtime();

        vel_setup_time  -= MPI_Wtime();
        update_sparse_matrix (UVW_URFactor, A_phi.V, phi.V, S_phi.V); 
        vel_setup_time  += MPI_Wtime();
        
		vel_solve_time  -= MPI_Wtime();     
		solve_sparse_matrix (phi.V);
		vel_solve_time += MPI_Wtime();

        vel_setup_time  -= MPI_Wtime();
        update_sparse_matrix (UVW_URFactor, A_phi.W, phi.W, S_phi.W);
        vel_setup_time  += MPI_Wtime();
        
		vel_solve_time  -= MPI_Wtime();  
        solve_sparse_matrix (phi.W);
        vel_solve_time += MPI_Wtime();
		vel_total_time += MPI_Wtime();

		if(((timestep_count + 1) % TIMER_OUTPUT_INTERVAL == 0) 
				&& FLOW_SOLVER_FINE_TIME)
        {
			if(mpi_config->particle_flow_rank == 0)
			{
                MPI_Reduce(MPI_IN_PLACE, &vel_total_time, 1, MPI_DOUBLE, 
							MPI_SUM, 0, mpi_config->particle_flow_world);
				MPI_Reduce(MPI_IN_PLACE, &vel_flux_time, 1, MPI_DOUBLE,
							MPI_SUM, 0, mpi_config->particle_flow_world);
				MPI_Reduce(MPI_IN_PLACE, &vel_setup_time, 1, MPI_DOUBLE,
							MPI_SUM, 0, mpi_config->particle_flow_world);
				MPI_Reduce(MPI_IN_PLACE, &vel_solve_time, 1, MPI_DOUBLE,
							MPI_SUM, 0, mpi_config->particle_flow_world);

                printf("\nVelocity Field Solve Timings:\n");
				printf("Compute Flux time: %.3f (%.2f %%)\n",
                        vel_flux_time / mpi_config->particle_flow_world_size,
                        100 * vel_flux_time / vel_total_time);
				printf("Matrix Setup time: %.3f (%.2f %%)\n",
                        vel_setup_time / mpi_config->particle_flow_world_size,
                        100 * vel_setup_time / vel_total_time);
				printf("Matrix Solve time: %.3f (%.2f %%)\n",
                        vel_solve_time / mpi_config->particle_flow_world_size,
                        100 * vel_solve_time / vel_total_time);
                printf("Total time: %f\n", vel_total_time / mpi_config->particle_flow_world_size);
			}
			else
			{
				MPI_Reduce(&vel_total_time, nullptr, 1, MPI_DOUBLE, MPI_SUM, 
							0, mpi_config->particle_flow_world);
				MPI_Reduce(&vel_flux_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
				MPI_Reduce(&vel_setup_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
				MPI_Reduce(&vel_solve_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
			}
			vel_flux_time = 0.0;
			vel_setup_time = 0.0;
			vel_solve_time = 0.0;
		}
    }

	template<typename T> void FlowSolver<T>::update_mass_flux()
	{
		/*Update the mass flux for the internal faces. Used for the incremental
        update of the velocity and pressure fields*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function update_mass_flux.\n", mpi_config->rank);
		
		//exchange_phi_halos();
		//MPI_Barrier(mpi_config->particle_flow_world);

		for ( uint64_t face = 0; face < mesh->faces_size; face++ )
		{
			const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;			

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;
			if ( mesh->faces[face].cell1 >= mesh->mesh_size ) continue;
			
			uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

				
			const vec<T> Xpac = face_centers[face] - dot_product(face_centers[face] - mesh->cell_centers[shmem_cell0], normalise(face_normals[face]))*normalise(face_normals[face]);
            const vec<T> Xnac = face_centers[face] - dot_product(face_centers[face] - mesh->cell_centers[shmem_cell1], normalise(face_normals[face]))*normalise(face_normals[face]);	
	
			vec<T> Xn = Xnac -  mesh->cell_centers[shmem_cell1];
			vec<T> Xp = Xpac -  mesh->cell_centers[shmem_cell0];

			//NOTE: This is required to avoid small values due to machine precision.
			if(abs(Xn.x) < 0.0000000000000003)
            {
                Xn.x = 0.0;
            }
            if(abs(Xn.y) < 0.0000000000000003)
            {
                Xn.y = 0.0;
            }
            if(abs(Xn.z) < 0.0000000000000003)
            {
                Xn.z = 0.0;
            }
            if(abs(Xp.x) < 0.0000000000000003)
            {
                Xp.x = 0.0;
            }
            if(abs(Xp.y) < 0.0000000000000003)
            {
                Xp.y = 0.0;
            }
            if(abs(Xp.z) < 0.0000000000000003)
            {
                Xp.z = 0.0;
            }

			T fact = face_fields[face].cell0;
			
			const T dpx  = phi_grad.PP[phi_index1].x * Xn.x - phi_grad.PP[phi_index0].x * Xp.x;
            const T dpy  = phi_grad.PP[phi_index1].y * Xn.y - phi_grad.PP[phi_index0].y * Xp.y;
            const T dpz  = phi_grad.PP[phi_index1].z * Xn.z - phi_grad.PP[phi_index0].z * Xp.z;
		
			
			//TODO:I think this is an underrlaxtion value make that clear.
			const T fc = fact * (dpx + dpy + dpz) * 0.8;

			face_mass_fluxes[face] += fc;

			S_phi.U[phi_index0] -= fc;
			S_phi.U[phi_index1] += fc;
		}
	}

    template<typename T> void FlowSolver<T>::calculate_mass_flux()
    {
		/*Calculate face based mass flux values for pressure solve*/
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_mass_flux.\n", mpi_config->rank);

        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

            if ( mesh->faces[face].cell1 < mesh->mesh_size )  // INTERNAL
            {

				uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

				
                const T lambda0 = face_lambdas[face];
                const T lambda1 = 1.0 - lambda0;
                
				const vec<T> dUdXac  =   phi_grad.U[phi_index0] * lambda0 + phi_grad.U[phi_index1] * lambda1;
                const vec<T> dVdXac  =   phi_grad.V[phi_index0] * lambda0 + phi_grad.V[phi_index1] * lambda1;
                const vec<T> dWdXac  =   phi_grad.W[phi_index0] * lambda0 + phi_grad.W[phi_index1] * lambda1;

                vec<T> Xac = mesh->cell_centers[shmem_cell1] * lambda1 + mesh->cell_centers[shmem_cell0] * lambda0;

                const vec<T> delta  = face_centers[face] - Xac;

                const T UFace = phi.U[phi_index1]*lambda1 + phi.U[phi_index0]*lambda0 + dot_product( dUdXac , delta );
                const T VFace = phi.V[phi_index1]*lambda1 + phi.V[phi_index0]*lambda0 + dot_product( dVdXac , delta );
                const T WFace = phi.W[phi_index1]*lambda1 + phi.W[phi_index0]*lambda0 + dot_product( dWdXac , delta );

                const T densityf = cell_densities[phi_index0] * lambda0 + cell_densities[phi_index1] * lambda1;
				
				face_mass_fluxes[face] = densityf * (UFace * face_normals[face].x +
                                                     VFace * face_normals[face].y +
                                                     WFace * face_normals[face].z );
				
				const vec<T> Xpac = face_centers[face] - dot_product(face_centers[face] - mesh->cell_centers[shmem_cell0], normalise(face_normals[face]))*normalise(face_normals[face]);
                const vec<T> Xnac = face_centers[face] - dot_product(face_centers[face] - mesh->cell_centers[shmem_cell1], normalise(face_normals[face]))*normalise(face_normals[face]);


                const vec<T> delp = Xpac - mesh->cell_centers[shmem_cell0];
                const vec<T> deln = Xnac - mesh->cell_centers[shmem_cell1];

                const T cell0_P = phi.P[phi_index0] + dot_product( phi_grad.P[phi_index0] , delp );
                const T cell1_P = phi.P[phi_index1] + dot_product( phi_grad.P[phi_index1] , deln );
                const vec<T> Xpn  = Xnac - Xpac;
                const vec<T> Xpn2 = mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0]; 

                const T ApV0 = (A_phi.U[phi_index0] != 0.0) ? 1.0 / A_phi.U[phi_index0] : 0.0;
                const T ApV1 = (A_phi.U[phi_index1] != 0.0) ? 1.0 / A_phi.U[phi_index1] : 0.0;

                T ApV = cell_densities[phi_index0] * ApV0 * lambda0 + cell_densities[phi_index1] * ApV1 * lambda1;
		
				const T volume_avg = cell_volumes[phi_index0] * lambda0 + cell_volumes[phi_index1] * lambda1;
	
                ApV  = ApV * face_areas[face] * volume_avg/dot_product(Xpn2, normalise(face_normals[face]));
				
				const T dpx  = ( phi_grad.P[phi_index1].x * lambda1 + phi_grad.P[phi_index0].x * lambda0) * Xpn.x; 
                const T dpy  = ( phi_grad.P[phi_index1].y * lambda1 + phi_grad.P[phi_index0].y * lambda0) * Xpn.y;  
                const T dpz  = ( phi_grad.P[phi_index1].z * lambda1 + phi_grad.P[phi_index0].z * lambda0) * Xpn.z; 

                face_fields[face].cell0 = -ApV;
                face_fields[face].cell1 = -ApV;
				
				face_mass_fluxes[face] -= ApV * ((cell1_P - cell0_P) - dpx - dpy - dpz);
			}
            else // BOUNDARY
            {
                // Boundary faces
                const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                const uint64_t boundary_type = mesh->boundary_types[boundary_cell];

                if ( boundary_type == INLET )
                {
                    // Constant inlet values for velocities and densities. Add custom regions laters
                    const vec<T> vel_inward = mesh->dummy_gas_vel;
                    const T Din = 1.2;

                    face_mass_fluxes[face] = Din * dot_product( vel_inward, face_normals[face] );
					S_phi.U[block_cell0] = S_phi.U[block_cell0] - face_mass_fluxes[face];
                }
                else if( boundary_type == OUTLET )
                {
                    const vec<T> vel_outward = { phi.U[block_cell0], 
                                                 phi.V[block_cell0], 
                                                 phi.W[block_cell0] };
					const T Din = 1.2;

                    face_mass_fluxes[face] = Din * dot_product(vel_outward, face_normals[face]);
					// !
                    // ! For an outlet face_mass_fluxes must be 0.0 or positive
                    // !
                    if( face_mass_fluxes[face] < 0.0 )
                    {
                        printf("MAIN COMP PRES NEGATIVE OUTFLOW %3.18f\n", face_mass_fluxes[face]);
                        face_mass_fluxes[face] = 1e-15;

						phi.TE[mesh->local_mesh_size + nhalos + boundary_cell] =
                            phi.TE[block_cell0];
						phi.ED[mesh->local_mesh_size + nhalos + boundary_cell] =
                            phi.ED[block_cell0];
						phi.TEM[mesh->local_mesh_size + nhalos + boundary_cell] =
                            phi.TEM[block_cell0];
                    }
                }
                else if( boundary_type == WALL )
                {
                    face_mass_fluxes[face] = 0.0;   
                }
            }
        }
		//Conserve mass if the flow has not reached the outflow yet.
		T FlowOut = 0.0;
		T FlowIn = 0.0;
		T areaout= 0.0;
		int count_out = 0;		
		for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
			if ( mesh->faces[face].cell1 < mesh->mesh_size )  continue;
			//Boundary only
			const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
            const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
			if(boundary_type == INLET)
			{
				FlowIn += face_mass_fluxes[face];
			}
			else if(boundary_type == OUTLET)
			{
				FlowOut += face_mass_fluxes[face];
				count_out++;
				areaout += face_areas[face];
			}
		}
		MPI_Allreduce(MPI_IN_PLACE, &FlowIn, 1, MPI_DOUBLE, MPI_SUM, mpi_config->particle_flow_world);
		MPI_Allreduce(MPI_IN_PLACE, &FlowOut, 1, MPI_DOUBLE, MPI_SUM, mpi_config->particle_flow_world);
		T FlowFact[count_out];
		int step = 0;
		for(int i = 0; i < count_out; i++){
			if(FlowOut == 0.0)
			{
				//This protects against NaN
				FlowFact[i] = 0.0;
			}
			else
			{
				FlowFact[i] = -FlowIn/FlowOut;
			}
		}
		if(FlowOut < 0.0000000001)
		{
			T ratearea = - FlowIn/areaout;
			FlowOut = 0.0;
			for ( uint64_t face = 0; face < mesh->faces_size; face++ )
            {
                if ( mesh->faces[face].cell1 < mesh->mesh_size )  continue;
                //Boundary only
                const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
				if(boundary_type == OUTLET)
				{
					//NOTE: assumes constent and uniform density
					//NOTE: assumes one outflow region
					face_mass_fluxes[face] = ratearea*face_areas[face];
					T FaceFlux = face_mass_fluxes[face]/cell_densities[0]/face_areas[face];
					
					phi.U[mesh->local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).x;
					phi.V[mesh->local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).y;
					phi.W[mesh->local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).z;

					FlowOut += face_mass_fluxes[face];
				}
			}
		}
		T fact = -FlowIn/(FlowOut + 0.0000001);
		step = 0;
		for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            if ( mesh->faces[face].cell1 < mesh->mesh_size )  continue;
            //Boundary only
            const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
            const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
            if(boundary_type == OUTLET)
            {
				face_mass_fluxes[face] *= FlowFact[step];
				step ++;
				 	
				phi.U[mesh->local_mesh_size + nhalos + boundary_cell] *= fact;
				phi.V[mesh->local_mesh_size + nhalos + boundary_cell] *= fact;
				phi.W[mesh->local_mesh_size + nhalos + boundary_cell] *= fact; 

			
				const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
				S_phi.U[block_cell0] -= face_mass_fluxes[face];
			}
		}
    }

    template<typename T> void FlowSolver<T>::setup_pressure_matrix()
    {
		/*Set up a sparse A matrix using PETSc for the pressure solve*/
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function setup_pressure_matrix.\n", mpi_config->rank);

		MatZeroEntries(A);

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if (mesh->faces[face].cell1 >= mesh->mesh_size)  continue; // Remove when implemented boundary cells. Treat boundary as mesh size
            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
			
			MatSetValue(A, mesh->faces[face].cell0, mesh->faces[face].cell1, face_fields[face].cell1, INSERT_VALUES);
            MatSetValue(A, mesh->faces[face].cell1, mesh->faces[face].cell0, face_fields[face].cell0, INSERT_VALUES);		

            S_phi.U[phi_index0] -= face_mass_fluxes[face];
            S_phi.U[phi_index1] += face_mass_fluxes[face];
           
			A_phi.V[phi_index0] -= face_fields[face].cell1;
            A_phi.V[phi_index1] -= face_fields[face].cell0;

		}

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
			MatSetValue(A, i+mesh->local_cells_disp, i+mesh->local_cells_disp, A_phi.V[i], INSERT_VALUES);
		}

		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    }

    template<typename T> void FlowSolver<T>::solve_pressure_matrix()
    {
		/*Set up b vectory and solve linear system Au=b using PETSc*/
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_pressure_matrix.\n", mpi_config->rank);

		VecZeroEntries(b);
		VecZeroEntries(u);
		int flag = 1;
		//NOTE: We do this since the solution falls apart with RHS vector close to zero.
		//TODO: Compare zero with RHS using confidence if that is acceptable use that.
        for(uint64_t i = 0; i < mesh->local_mesh_size; i++)
		{
			if(abs(S_phi.U[i]) > 0.0000000001)
			{
				flag = 0;
			}
			VecSetValue(b, i+mesh->local_cells_disp, S_phi.U[i], INSERT_VALUES);
		}

		if(flag == 1)
		{
			for(uint64_t i = 0; i < mesh->local_mesh_size; i++)
			{
				VecSetValue(b, i+mesh->local_cells_disp, 0.0, INSERT_VALUES);
			}
		}
		
		VecAssemblyBegin(b);
        VecAssemblyEnd(b);

		KSPSolve(ksp, b, u);

		PetscInt indx[mesh->local_mesh_size];
        for(uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
            indx[i] = i+mesh->local_cells_disp;
        }
        VecGetValues(u, mesh->local_mesh_size, indx, phi.PP);
    }

	template<typename T> void FlowSolver<T>::Update_P_at_boundaries(T *phi_component)
	{
		/*Update the value of a Pressure phi component at the boundaries, with
        the value of phi for the internal side of the face. Used in the progressive update
        of Pressure and Velocity fields in compute pressure.*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function Update_P_at_boundaries.\n", mpi_config->rank);
		#pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
		{
			const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
			if (mesh->faces[face].cell1 < mesh->mesh_size)  continue;
			//we only want boundary faces.
			const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
			phi_component[mesh->local_mesh_size + nhalos + boundary_cell] = phi_component[block_cell0];	
		}
	}

	template<typename T> void FlowSolver<T>::update_P(T *phi_component, vec<T> *phi_grad_component)
	{
		/*Final update of the value of P at each boundary*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function update_P.\n", mpi_config->rank);
		#pragma ivdep
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
			const uint64_t shmem_cell = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            if (mesh->faces[face].cell1 < mesh->mesh_size)  continue;
            //we only want boundary faces.
			const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
            const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
			if(boundary_type == OUTLET)
			{		
				phi_component[mesh->local_mesh_size + nhalos + boundary_cell] = phi_component[block_cell0];
			}
			else
			{
				vec<T> ds = face_centers[face] - mesh->cell_centers[shmem_cell];
				phi_component[mesh->local_mesh_size + nhalos + boundary_cell] = phi_component[block_cell0] + dot_product(phi_grad_component[block_cell0], ds);
			}
		}
	}

    template<typename T> void FlowSolver<T>::calculate_pressure()
    {
		/*Solve and update pressure using the SIMPLE alogrithm
            1. Solve the pressure correction equation.
            2. Update the pressure field
            3. Update the boundary pressure corrections
            4. Correct the face mass fluxes
            5. Correct the cell velocities
        */
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_pressure.\n", mpi_config->rank);
		pres_total_time -= MPI_Wtime();
		int Loop_num = 0;
		bool Loop_continue = true;
		T Pressure_correction_max = 0;
		T Pressure_correction_ref = 0;
        
		#pragma ivdep 
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
			A_phi.V[i] = 0.0;
			S_phi.U[i] = 0.0;
        }

		#pragma ivdep
		for ( uint64_t face = 0; face < mesh->faces_size; face++)
		{
			face_mass_fluxes[face] = 0.0;
			face_fields[face].cell0 = 0.0;
			face_fields[face].cell1 = 0.0;		
		}

		pres_halo_time -= MPI_Wtime();
		exchange_A_halos(A_phi.U);
		pres_halo_time += MPI_Wtime();
      

		pres_flux_time -= MPI_Wtime();
		calculate_mass_flux();  //Compute the uncorrected mass fluxes at every faces
		pres_flux_time += MPI_Wtime();
		
		pres_setup_time -= MPI_Wtime();
		setup_pressure_matrix(); //Set up Sp and Ap for the initial pressure solve.
		pres_setup_time += MPI_Wtime();
		
		while(Loop_continue)  //conduct a number of improvements to the pressure and velocity fields.
		{
			Loop_num++;
		
			pres_solve_time -= MPI_Wtime();
			solve_pressure_matrix(); //Compute pressure correction
			pres_solve_time += MPI_Wtime();

			Pressure_correction_max = phi.PP[0];
			for ( uint64_t i = 1; i < mesh->local_mesh_size; i++ )
			{
				if(abs(phi.PP[i]) > Pressure_correction_max) 
				{
					Pressure_correction_max = abs(phi.PP[i]);
				}
			}
	
			pres_halo_time -= MPI_Wtime();
			MPI_Allreduce(MPI_IN_PLACE, &Pressure_correction_max, 1, 
				MPI_DOUBLE, MPI_MAX, mpi_config->particle_flow_world);
			pres_halo_time += MPI_Wtime();			

			if(Loop_num == 1)
			{
				Pressure_correction_ref = Pressure_correction_max;
			}

			pres_halo_time -= MPI_Wtime();
			exchange_single_phi_halo(phi.PP); //exchange so phi.PP is correct at halos.
			pres_halo_time += MPI_Wtime();

			Update_P_at_boundaries(phi.PP); //Update boundary pressure
   
			get_phi_gradient(phi.PP, phi_grad.PP, true); //Compute gradient of correction.
			
			pres_halo_time -= MPI_Wtime();
			exchange_single_grad_halo(phi_grad.PP); //exchange so phi_grad.PP is correct at halos 
			pres_halo_time += MPI_Wtime();

			for ( uint64_t face = 0; face < mesh->faces_size; face++ ){
				const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
				const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;
				
				if( mesh->faces[face].cell1 >= mesh->mesh_size ) continue;
				//internal cells
				uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
				uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
				face_mass_fluxes[face] += (face_fields[face].cell0*(phi.PP[phi_index1] - phi.PP[phi_index0]));
				
			}

			/*halos values of phi.PP correct so updating halo values of phi.P here avoids
			  a halo exchange later*/
			for ( uint64_t cell = 0; cell < mesh->local_mesh_size + nhalos; cell++ )
			{
				//Partial update of the velocity and pressure field.
				T Ar = (A_phi.U[cell] != 0.0) ? 1.0 / A_phi.U[cell] : 0.0;
				T fact = cell_volumes[cell] * Ar;

				phi.P[cell] += 0.2*phi.PP[cell];

				phi.U[cell] -= phi_grad.PP[cell].x * fact;
				phi.V[cell] -= phi_grad.PP[cell].y * fact; 
				phi.W[cell] -= phi_grad.PP[cell].z * fact;
			}
			
			//Reset Su for the next partial solve.
			#pragma ivdep 
			for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
			{
				S_phi.U[i] = 0.0; 
			}

			pres_flux_time -= MPI_Wtime();
			update_mass_flux(); //Compute the correction for the mass fluxes
			pres_flux_time += MPI_Wtime();

			//Reset PP for next partial solve
			for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos + mesh->boundary_cells_size; i++ )
			{
				phi.PP[i] = 0.0;
			}
			if(Loop_num >= 4 or Pressure_correction_max <= 0.25*Pressure_correction_ref) Loop_continue = false;
		}

		Update_P_at_boundaries(phi.P); //update boundaries for full Pressure field.
		
		get_phi_gradient(phi.P, phi_grad.P, true); //Update gradient for Pressure field.

		pres_halo_time -= MPI_Wtime();
		exchange_single_grad_halo(phi_grad.P); //Exchange so phi_grad.P is correct at halos
		pres_halo_time += MPI_Wtime();
		
		update_P(phi.P, phi_grad.P); //Final update of pressure field.
		pres_total_time += MPI_Wtime();
	
		if(((timestep_count + 1) % TIMER_OUTPUT_INTERVAL == 0) 
			&& FLOW_SOLVER_FINE_TIME)
        {
            if(mpi_config->particle_flow_rank == 0)
            {
                MPI_Reduce(MPI_IN_PLACE, &pres_total_time, 1, MPI_DOUBLE,
                            MPI_SUM, 0, mpi_config->particle_flow_world);
                MPI_Reduce(MPI_IN_PLACE, &pres_flux_time, 1, MPI_DOUBLE,
                            MPI_SUM, 0, mpi_config->particle_flow_world);
                MPI_Reduce(MPI_IN_PLACE, &pres_setup_time, 1, MPI_DOUBLE,
                            MPI_SUM, 0, mpi_config->particle_flow_world);
                MPI_Reduce(MPI_IN_PLACE, &pres_solve_time, 1, MPI_DOUBLE,
                            MPI_SUM, 0, mpi_config->particle_flow_world);
				MPI_Reduce(MPI_IN_PLACE, &pres_halo_time, 1, MPI_DOUBLE,
							MPI_SUM, 0, mpi_config->particle_flow_world);

                printf("\nPressure Field Solve Timings:\n");
                printf("Compute Flux time: %.3f (%.2f %%)\n",
                        pres_flux_time / mpi_config->particle_flow_world_size,
                        100 * pres_flux_time / pres_total_time);
                printf("Matrix Setup time: %.3f (%.2f %%)\n",
                        pres_setup_time / mpi_config->particle_flow_world_size,
                        100 * pres_setup_time / pres_total_time);
                printf("Matrix Solve time: %.3f (%.2f %%)\n",
                        pres_solve_time / mpi_config->particle_flow_world_size,
                        100 * pres_solve_time / pres_total_time);
				printf("Communication time: %.3f (%.2f %%)\n",
						pres_halo_time / mpi_config->particle_flow_world_size,
						100 * pres_halo_time / pres_total_time);
                printf("Total time: %f\n", pres_total_time / mpi_config->particle_flow_world_size);
            }
            else
            {
                MPI_Reduce(&pres_total_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
                MPI_Reduce(&pres_flux_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
                MPI_Reduce(&pres_setup_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
                MPI_Reduce(&pres_solve_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
				MPI_Reduce(&pres_halo_time, nullptr, 1, MPI_DOUBLE, MPI_SUM,
							0, mpi_config->particle_flow_world);
			}
            pres_flux_time = 0.0;
            pres_setup_time = 0.0;
            pres_solve_time = 0.0;
			pres_halo_time = 0.0;
        }
    }

	template<typename T> void FlowSolver<T>::FluxScalar(int type, T *phi_component, vec<T> *phi_grad_component)
	{
		/*Compute the face based flux values for a general Scalar.*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function FluxScalar.\n", mpi_config->rank);

        T pe0 =  9999.;
        T pe1 = -9999.;
        T GammaBlend = 0.0; // NOTE: Change when implemented other differencing schemes.

        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

            if ( mesh->faces[face].cell1 < mesh->mesh_size )  // INTERNAL
            {
                uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
				uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
                
				const T lambda0 = face_lambdas[face];
                const T lambda1 = 1.0 - lambda0;

				T Visac    = effective_viscosity * lambda0 + effective_viscosity * lambda1;
				
				Visac -= effective_viscosity; //This will always be 0 right?
				
				if(type == TEMP)
				{
					Visac = (effective_viscosity + Visac / 0.9) / 0.6905;
				}
				else if(type  == TERBTE)
				{
					Visac = effective_viscosity + Visac;
				}
				else if(type == TERBED)
				{
					Visac = effective_viscosity + Visac / 1.219;
				}
				else
				{
					Visac = (effective_viscosity + Visac / 0.9) / 0.9;
				}

				vec<T> dPhiXac = phi_grad_component[phi_index0] * lambda0 + phi_grad_component[phi_index1] * lambda1;

                vec<T> Xpn     = mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0];
				const T VisFace = Visac * face_rlencos[face];

                T PhiFace;
                if ( face_mass_fluxes[face] >= 0.0 )
                {
                    PhiFace  = phi_component[phi_index0];
                }
                else
                {
                    PhiFace  = phi_component[phi_index1];
                }

                // explicit higher order convective flux (see eg. eq. 8.16)
                const T fce = face_mass_fluxes[face] * PhiFace;
				const T fde1 = Visac * dot_product ( dPhiXac , face_normals[face] );
				
				//implicit lower order (simple upwind)
				//convective and diffusive fluxes
                const T fci = min( face_mass_fluxes[face], 0.0 ) * phi_component[phi_index0] + max( face_mass_fluxes[face], 0.0 ) * phi_component[phi_index1];

                const T fdi = VisFace * dot_product( dPhiXac , Xpn );

                // !
                // ! convective coefficients with deferred correction with
                // ! gamma as the blending factor (0.0 <= gamma <= 1.0)
                // !
                // !      low            high    low  OLD
                // ! F = F    + gamma ( F     - F    )
                // !     ----   -------------------------
                // !      |                  |
                // !  implicit           explicit (dump into source term)
                // !
				face_fields[face].cell0 = -VisFace - max( face_mass_fluxes[face] , 0.0 );
                face_fields[face].cell1 = -VisFace + min( face_mass_fluxes[face] , 0.0 );

				const T blend = GammaBlend * ( fce - fci );

				// ! assemble the two source terms
                S_phi.U[phi_index0] = S_phi.U[phi_index0] - blend + fde1 - fdi;
                S_phi.U[phi_index1] = S_phi.U[phi_index1] + blend - fde1 + fdi;

                const T small_epsilon = 1.e-20;
                const T peclet = face_mass_fluxes[face] / face_areas[face] * magnitude(Xpn) / (Visac+small_epsilon);

                pe0 = min( pe0 , peclet );
                pe1 = max( pe1 , peclet );

            }
            else // BOUNDARY
            {
                // Boundary faces
                const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                const uint64_t boundary_type = mesh->boundary_types[boundary_cell];

                if ( boundary_type == INLET )
                {
                    // Option to add more inlet region information and functions here.
                    const vec<T> dPhidXac = phi_grad_component[block_cell0];

                    T PhiFace;
					if(type == TEMP)
					{
						PhiFace = mesh->dummy_gas_tem;
					}
					else if(type  == TERBTE)
					{
						T velmag2 = pow(mesh->dummy_gas_vel.x,2) + pow(mesh->dummy_gas_vel.y,2) + pow(mesh->dummy_gas_vel.z,2);
						PhiFace = 3.0/2.0*((0.1*0.1)*velmag2);;
					}
					else if(type == TERBED)
					{
						T velmag2 = pow(mesh->dummy_gas_vel.x,2) + pow(mesh->dummy_gas_vel.y,2) + pow(mesh->dummy_gas_vel.z,2);	
						PhiFace = pow(0.09,0.75) * pow((3.0/2.0*((0.1*0.1)*velmag2)),1.5);
					}
					else if(type == FUEL)
					{
						PhiFace = mesh->dummy_gas_fuel;
					}
					else if(type == PROG)
					{
						PhiFace = 0.0;
					}
					else if(type == VARFU)
					{
						PhiFace = mesh->dummy_gas_fuel;
					}
					else if(type == VARPR)
					{
						PhiFace = 0.0;
					}
					else
					{
						printf("Error: unkown type in flux scalar\n");
						exit(0);
					}		

                    T Visac = inlet_effective_viscosity;
					
					Visac -= effective_viscosity; //This will always be 0 right?

					if(type == TEMP)
					{
						Visac = (effective_viscosity + Visac / 0.9) / 0.6905;
					}
					else if(type  == TERBTE)
					{
						Visac = effective_viscosity + Visac;
					}
					else if(type == TERBED)
					{
						Visac = effective_viscosity + Visac / 1.219;
					}
					else
					{
						Visac = (effective_viscosity + Visac / 0.9) / 0.9;
					}

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];
					const T VisFace  = Visac * face_rlencos[face];

                    const T fde = Visac * dot_product( dPhidXac , face_normals[face]);

					//implicit part
					const T fdi = VisFace * dot_product( dPhidXac, Xpn);
                    
					const T f = -VisFace + min( face_mass_fluxes[face], 0.0 );
                    
					A_phi.V[block_cell0] = A_phi.V[block_cell0] - f;
                    S_phi.U[block_cell0] = S_phi.U[block_cell0] - f * PhiFace + fde - fdi;
                    
					phi_component[mesh->local_mesh_size + nhalos + boundary_cell] = PhiFace;
                }
                else if( boundary_type == OUTLET )
                {
					const vec<T> dPhidXac = phi_grad_component[block_cell0];

                    T Visac = effective_viscosity;

                    Visac -= effective_viscosity; //This will always be 0 right?

                    if(type == TEMP)
                    {
                        Visac = (effective_viscosity + Visac / 0.9) / 0.6905;
                    }
                    else if(type  == TERBTE)
                    {
                        Visac = effective_viscosity + Visac;
                    }
                    else if(type == TERBED)
                    {
                        Visac = effective_viscosity + Visac / 1.219;
                    }
                    else
                    {
                        Visac = (effective_viscosity + Visac / 0.9) / 0.9;
                    }

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];

					const T PhiFace = phi_component[block_cell0] + dot_product( dPhidXac , Xpn );
                    const T VisFace  = Visac * face_rlencos[face];
				
                    const T fde = Visac * dot_product( dPhidXac , face_normals[face] );

                    const T fdi = VisFace * dot_product( dPhidXac , Xpn );

                    S_phi.U[block_cell0] = S_phi.U[block_cell0] + fde - fdi;
                    
					phi_component[mesh->local_mesh_size + nhalos + boundary_cell] = PhiFace;
                }
                else if( boundary_type == WALL )
                {
					if( type != TERBTE or type != TERBED )
					{ 
						phi_component[mesh->local_mesh_size + nhalos + boundary_cell] = phi_component[block_cell0];
					}
				}
			}
		}
	}

	template<typename T> void FlowSolver<T>::solveTurbulenceModels(int type)
	{
		/*compute the effects of the terbulent forces including the effect
        at the wall*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solveTurbulenceModels.\n", mpi_config->rank);
		if(type == TERBTE)
		{
			for ( uint64_t i = 0; i < mesh->local_mesh_size; i++ )
			{
				const vec<T> dUdXp  =   phi_grad.U[i];
                const vec<T> dVdXp  =   phi_grad.V[i];
                const vec<T> dWdXp  =   phi_grad.W[i];

				const T s1 = (dUdXp.x+dUdXp.x)*dUdXp.x + (dUdXp.y+dVdXp.x)*dUdXp.y + (dUdXp.z+dWdXp.x)*dUdXp.z;
				const T s2 = (dVdXp.x+dUdXp.y)*dVdXp.x + (dVdXp.y+dVdXp.y)*dVdXp.y + (dVdXp.z+dWdXp.y)*dVdXp.z;
				const T s3 = (dWdXp.x+dUdXp.z)*dWdXp.x + (dWdXp.y+dVdXp.z)*dWdXp.y + (dWdXp.z+dWdXp.z)*dWdXp.z;

				//NOTE: The first part of the below becomes Vis[i] if 
				//We compute viscosity
				T VisT = effective_viscosity - effective_viscosity; 
				
				T Pk = VisT * (s1 + s2 + s3);

				phi.TP[i] = Pk;

				T Dis = cell_densities[i] * phi.ED[i];

				S_phi.U[i] = S_phi.U[i] + phi.TP[i] * cell_volumes[i];
				A_phi.V[i] = A_phi.V[i] + Dis / (phi.TE[i] + 0.000000000000000001) * cell_volumes[i]; 	
			}
			T Cmu = 0.09;
			T Cmu75 = pow(Cmu, 0.75);
			for ( uint64_t face = 0; face < mesh->faces_size; face++ )
			{
				const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
				const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
				if ( mesh->faces[face].cell1 < mesh->mesh_size ) continue;
				//only need the boundary cells
				const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
				const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
				if ( boundary_type == WALL )
				{
					//at walls we need a different source term
					S_phi.U[block_cell0] = S_phi.U[block_cell0] - phi.TP[block_cell0] * cell_volumes[block_cell0];
					
					const T UFace = 0.; // Customisable (add regions here later)
                    const T VFace = 0.; // Customisable (add regions here later)
                    const T WFace = 0.; // Customisable (add regions here later)

                    const T Visc = effective_viscosity;

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];

                    vec<T> Up;
                    Up.x = phi.U[block_cell0] - UFace;
                    Up.y = phi.V[block_cell0] - VFace;
                    Up.z = phi.W[block_cell0] - WFace;

                    const T dp = dot_product( Up , normalise(face_normals[face]));
                    vec<T> Ut  = Up - dp * normalise(face_normals[face]);

                    const T Uvel = sqrt(dot_product( Ut, Ut));
					const T distance_to_face = magnitude(Xpn);

					//production of in wall region
					const T rkapdn = 1.0/( 0.419 * distance_to_face);
					
					//if yplus > ylog we only have less than implemented
					const T Tau_w = Visc * Uvel / distance_to_face;
					const T Utau = sqrt( Tau_w / cell_densities[block_cell0]);

					phi.TP[block_cell0] = Tau_w * Utau * rkapdn;
					phi.TE[mesh->local_mesh_size + nhalos + boundary_cell] = phi.TE[block_cell0];
					
					S_phi.U[block_cell0] = S_phi.U[block_cell0] + phi.TP[block_cell0] * cell_volumes[block_cell0];

					//dissipation term
					T DisP = Cmu75*sqrt(phi.TE[block_cell0])*rkapdn;
					
					A_phi.V[block_cell0] = A_phi.V[block_cell0] + cell_densities[block_cell0] * DisP * cell_volumes[block_cell0];
				}
			}
		}
		else if(type == TERBED)
		{
			for ( uint64_t i = 0; i < mesh->local_mesh_size; i++ )
			{
				T fact = phi.ED[i]/(phi.TE[i]+0.000000000000000001) * cell_volumes[i];
				S_phi.U[i] = S_phi.U[i] + 1.44 * fact * phi.TP[i];
				A_phi.V[i] = A_phi.V[i] + 1.92 * fact * cell_densities[i];
			}
			T Cmu = 0.09;
			T Cmu75 = pow(Cmu, 0.75);

			for ( uint64_t face = 0; face < mesh->faces_size; face++ )
            {
                const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
                const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                if ( mesh->faces[face].cell1 < mesh->mesh_size ) continue;
                //only need the boundary cells
                
				const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
                const uint64_t boundary_type = mesh->boundary_types[boundary_cell];
                if ( boundary_type == WALL )
                {
					const T turb = phi.TE[block_cell0];
					const T distance = magnitude(face_centers[face] - mesh->cell_centers[shmem_cell0]);
					const T Dis = Cmu75 * pow(turb,1.5) / ( distance * 0.419 );
					
					for ( uint64_t j = 0; j < 6; j++ )
					{
						uint64_t neigh_face = mesh->cell_faces[(block_cell0 * mesh->faces_per_cell) + j];
						if((mesh->faces[neigh_face].cell0 < mesh->mesh_size) and (mesh->faces[neigh_face].cell1 < mesh->mesh_size))
						{
							if((mesh->faces[neigh_face].cell1 - mesh->local_cells_disp) >= mesh->local_mesh_size)
							{
								face_fields[neigh_face].cell0 = 0.0;
								face_fields[neigh_face].cell1 = 0.0;
							}
							//internal node
							if((mesh->faces[neigh_face].cell0 - mesh->local_cells_disp) == block_cell0)
							{
								face_fields[neigh_face].cell1 = 0.0;
							}
							else if((mesh->faces[neigh_face].cell1 - mesh->local_cells_disp) == block_cell0)
							{
								face_fields[neigh_face].cell0 = 0.0;
							}
						}
					}
					phi.ED[block_cell0] = Dis;
					S_phi.U[block_cell0] = Dis;
					A_phi.V[block_cell0] = 1;
					phi.ED[mesh->local_mesh_size + nhalos + boundary_cell] = phi.ED[block_cell0];
				}
			}
		}
	}

	template<typename T> void FlowSolver<T>::Scalar_solve(int type, T *phi_component, vec<T> *phi_grad_component)
	{
		/*Solve for a general scalar used for most transport equations
          Follows the general procedure:
          1. Collect face fluxes
          2. Compute extra steps for terbulence
          3. set-up and solve matrix*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function Scalar_solve.\n", mpi_config->rank);
		
		sca_total_time[type] -= MPI_Wtime();
		//reuse Au and Su to reduce storage requirements
		#pragma ivdep
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            A_phi.V[i] = 0.0;
            S_phi.U[i] = 0.0;
        }

		//collect face fluxes
		sca_flux_time[type] -= MPI_Wtime();
		FluxScalar(type, phi_component, phi_grad_component);
		sca_flux_time[type] += MPI_Wtime();

		//Unsteady term 
		double rdelta = 1.0 / delta;
        #pragma ivdep
        for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++ )
        {
            double f = cell_densities[i] * cell_volumes[i] * rdelta;
            S_phi.U[i] += f * phi_component[i];
            A_phi.V[i] += f;
        }

		//RHS from particles for TEMP
		if(type == TEMP)
		{
			for(uint64_t i = 0; i < mesh->local_mesh_size; i++)
			{
				S_phi.U[i] += mesh->particle_terms[i].energy;
			}
		}

		if(type == TERBTE or type == TERBED)
		{
			sca_terb_time[type] -= MPI_Wtime();
			solveTurbulenceModels(type);
			sca_terb_time[type] += MPI_Wtime();
		}

		T underrelax = 0.0;
		if(type == TERBTE or type == TERBED)
		{
			underrelax = 0.5;
		}
		else if(type == TEMP)
		{
			underrelax = 0.95;
		}
		else
		{
			underrelax = 0.95;
		}
	
		sca_setup_time[type] -= MPI_Wtime();
		setup_sparse_matrix(underrelax, A_phi.V, phi_component, S_phi.U);
		sca_setup_time[type] += MPI_Wtime();	

		sca_solve_time[type] -= MPI_Wtime();
        solve_sparse_matrix (phi_component);
		sca_solve_time[type] += MPI_Wtime();

		if(type == TERBTE or type == TERBED)
		{
			//Make sure nonnegative
			for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++ )
			{
				int count = 0;
				T phisum = 0.0;
				if(phi_component[i] < 0.0)
				{
					for( uint64_t j = 0; j < 6; j++)
					{
						uint64_t neighbour = mesh->cell_neighbours[(i - mesh->shmem_cell_disp) * mesh->faces_per_cell + j];
						if(neighbour < mesh->mesh_size) 
						{
							//if internal
							const uint64_t block_neighbour = neighbour - mesh->local_cells_disp;
							//only average the neighbours on our process to avoid halo exchange
							if(block_neighbour < mesh->local_mesh_size)
							{
								count++;
								phisum += phi_component[block_neighbour];
							}	
						}
					}
					phisum = (phisum/count);
					phi_component[i] = max(phisum, 0.000000000001);
				}
			}
		}
		sca_total_time[type] += MPI_Wtime();
		
		if(((timestep_count + 1) % TIMER_OUTPUT_INTERVAL == 0) 
				&& FLOW_SOLVER_FINE_TIME)
        {
            if(mpi_config->particle_flow_rank == 0)
            {
                MPI_Reduce(MPI_IN_PLACE, &sca_total_time[type], 1, MPI_DOUBLE,
                            MPI_SUM, 0, mpi_config->particle_flow_world);
                MPI_Reduce(MPI_IN_PLACE, &sca_flux_time[type], 1, MPI_DOUBLE,
                            MPI_SUM, 0, mpi_config->particle_flow_world);
                MPI_Reduce(MPI_IN_PLACE, &sca_setup_time[type], 1, MPI_DOUBLE,
                            MPI_SUM, 0, mpi_config->particle_flow_world);
                MPI_Reduce(MPI_IN_PLACE, &sca_solve_time[type], 1, MPI_DOUBLE,
                            MPI_SUM, 0, mpi_config->particle_flow_world);
				if(type < 2)
				{
					MPI_Reduce(MPI_IN_PLACE, &sca_terb_time[type], 1, MPI_DOUBLE,
							MPI_SUM, 0, mpi_config->particle_flow_world);
				}
				if(type == TERBTE)
				{
					printf("\nTE Turbulence Field Solve Timings:\n");
					printf("Compute Turbulence time: %.3f (%.2f %%)\n",
                        sca_terb_time[type] / mpi_config->particle_flow_world_size,
                        100 * sca_terb_time[type] / sca_total_time[type]);
				}
				else if(type == TERBED)
				{
					printf("\nED Turbulence Field Solve Timings:\n");
					printf("Compute Turbulence time: %.3f (%.2f %%)\n",
                        sca_terb_time[type] / mpi_config->particle_flow_world_size,
                        100 * sca_terb_time[type] / sca_total_time[type]);
				}
				else if(type == TEMP)
				{
					printf("\nTempurature Field Solve Timings:\n");
				} 
				else if(type == FUEL)
				{
					printf("\nMixture Fraction Field Solve Timings:\n");
				}
				else if(type == PROG)
				{
					printf("\nProgress Variable Field Solve Timings:\n");
				}
				else if(type == VARFU)
				{
					printf("\nVariance of Mixture Fraction Field Solve Timings:\n");
				}
				else if(type == VARPR)
				{
					printf("\nVariance of Progress Variable Field Solve Timings:\n");
				}
                printf("Compute Flux time: %.3f (%.2f %%)\n",
                        sca_flux_time[type] / mpi_config->particle_flow_world_size,
                        100 * sca_flux_time[type] / sca_total_time[type]);
                printf("Matrix Setup time: %.3f (%.2f %%)\n",
                        sca_setup_time[type] / mpi_config->particle_flow_world_size,
                        100 * sca_setup_time[type] / sca_total_time[type]);
                printf("Matrix Solve time: %.3f (%.2f %%)\n",
                        sca_solve_time[type] / mpi_config->particle_flow_world_size,
                        100 * sca_solve_time[type] / sca_total_time[type]);
                printf("Total time: %f\n", sca_total_time[type] / mpi_config->particle_flow_world_size);
            }
            else
            {
                MPI_Reduce(&sca_total_time[type], nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
                MPI_Reduce(&sca_flux_time[type], nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
                MPI_Reduce(&sca_setup_time[type], nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
                MPI_Reduce(&sca_solve_time[type], nullptr, 1, MPI_DOUBLE, MPI_SUM,
                            0, mpi_config->particle_flow_world);
				if(type < 2)
				{
					MPI_Reduce(&sca_terb_time[type], nullptr, 1, MPI_DOUBLE, MPI_SUM,
								0, mpi_config->particle_flow_world);
				}
			}
            sca_flux_time[type] = 0.0;
            sca_setup_time[type] = 0.0;
            sca_solve_time[type] = 0.0;
			if(type < 2)
			{
				sca_terb_time[type] = 0.0;
			}
		}
	}

        template<class T>
    void FlowSolver<T>::print_logger_stats(uint64_t timesteps, double runtime)
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
		compute_time -= MPI_Wtime();
		exchange_phi_halos();

		get_phi_gradients();

		if(FLOW_SOLVER_LIMIT_GRAD)
			limit_phi_gradients();

		exchange_grad_halos();
	
		if(timestep_count == 0)
		{
			set_up_field();
			set_up_fgm_table();
		}
	
		compute_time += MPI_Wtime();
        
		if ((timestep_count % comms_timestep) == 0)  
            update_flow_field();

		compute_time -= MPI_Wtime();
		
		calculate_UVW();

		exchange_phi_halos(); //exchange new UVW values.

        calculate_pressure();
		
		//Turbulence solve
		Scalar_solve(TERBTE, phi.TE, phi_grad.TE);
		Scalar_solve(TERBED, phi.ED, phi_grad.ED);
	
		//temperature solve
		Scalar_solve(TEMP, phi.TEM, phi_grad.TEM);

		//fuel mixture fraction solve
		Scalar_solve(FUEL, phi.FUL, phi_grad.FUL);

		//rection progression solve
		Scalar_solve(PROG, phi.PRO, phi_grad.PRO);
	
		//Solve Variance of mixture fraction as transport equ
		Scalar_solve(VARFU, phi.VARF, phi_grad.VARF);

		//Solve Variance of progression as trasnport equ
		Scalar_solve(VARPR, phi.VARP, phi_grad.VARP);

		fgm_lookup_time -= MPI_Wtime();
		//Look up results from the FGM look-up table
		FGM_loop_up();
		fgm_lookup_time += MPI_Wtime();
		compute_time += MPI_Wtime();

		if(((timestep_count + 1) % 5) == 0)
		{
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
        
		if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Stop flow timestep.\n", mpi_config->rank);
        timestep_count++;
    }
}
