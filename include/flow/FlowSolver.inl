#include <stdio.h>
#include <limits.h>

#include "flow/FlowSolver.hpp"

#define TERBTE 1
#define TERBED 2
#define TEMP 3
#define FUEL 4
#define PROG  5
using namespace std;

namespace minicombust::flow 
{
    template<typename T> inline bool FlowSolver<T>::is_halo ( uint64_t cell )
    {
        return ( cell - mesh->local_cells_disp >= mesh->local_mesh_size );
    }


    template<typename T> void FlowSolver<T>::exchange_cell_info_halos ()
    {
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

        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::exchange_phi_halos ()
    {
		//TODO: fix to add PP and grad.pp
		//do we need TP
        int num_requests = 20;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Isend( phi.U,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
            MPI_Isend( phi.V,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 1, mpi_config->particle_flow_world, &send_requests[num_requests*r + 1] );
            MPI_Isend( phi.W,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 2, mpi_config->particle_flow_world, &send_requests[num_requests*r + 2] );
            MPI_Isend( phi.P,        1, halo_mpi_double_datatypes[r],     halo_ranks[r], 3, mpi_config->particle_flow_world, &send_requests[num_requests*r + 3] );
			MPI_Isend( phi.PP,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 4, mpi_config->particle_flow_world, &send_requests[num_requests*r + 4] );
			MPI_Isend( phi.TE,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 5, mpi_config->particle_flow_world, &send_requests[num_requests*r + 5] );	
			MPI_Isend( phi.ED,       1, halo_mpi_double_datatypes[r],     halo_ranks[r], 6, mpi_config->particle_flow_world, &send_requests[num_requests*r + 6] );
			MPI_Isend( phi.TEM,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 7, mpi_config->particle_flow_world, &send_requests[num_requests*r + 7] );
			MPI_Isend( phi.FUL,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 8, mpi_config->particle_flow_world, &send_requests[num_requests*r + 8] ); 
			MPI_Isend( phi.PRO,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 9, mpi_config->particle_flow_world, &send_requests[num_requests*r + 9] );

			MPI_Isend( phi_grad.U,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 10, mpi_config->particle_flow_world, &send_requests[num_requests*r + 10] );
            MPI_Isend( phi_grad.V,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 11, mpi_config->particle_flow_world, &send_requests[num_requests*r + 11] );
            MPI_Isend( phi_grad.W,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 12, mpi_config->particle_flow_world, &send_requests[num_requests*r + 12] );
            MPI_Isend( phi_grad.P,   1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 13, mpi_config->particle_flow_world, &send_requests[num_requests*r + 13] );
			MPI_Isend( phi_grad.PP,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 14, mpi_config->particle_flow_world, &send_requests[num_requests*r + 14] );
			MPI_Isend( phi_grad.TE,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 15, mpi_config->particle_flow_world, &send_requests[num_requests*r + 15] );
			MPI_Isend( phi_grad.ED,  1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 16, mpi_config->particle_flow_world, &send_requests[num_requests*r + 16] );
			MPI_Isend( phi_grad.TEM, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 17, mpi_config->particle_flow_world, &send_requests[num_requests*r + 17] );
			MPI_Isend( phi_grad.FUL, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 18, mpi_config->particle_flow_world, &send_requests[num_requests*r + 18] );	
			MPI_Isend( phi_grad.PRO, 1, halo_mpi_vec_double_datatypes[r], halo_ranks[r], 19, mpi_config->particle_flow_world, &send_requests[num_requests*r + 19] );
	}

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &phi.U[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
            MPI_Irecv( &phi.V[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 1, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 1] );
            MPI_Irecv( &phi.W[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 2, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 2] );
            MPI_Irecv( &phi.P[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 3, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 3] );
			MPI_Irecv( &phi.PP[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 4, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 4] );
			MPI_Irecv( &phi.TE[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 5, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 5] );
			MPI_Irecv( &phi.ED[mesh->local_mesh_size + halo_disps[r]],       halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 6, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 6] );
			MPI_Irecv( &phi.TEM[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 7, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 7] );
			MPI_Irecv( &phi.FUL[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 8, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 8] );
			MPI_Irecv( &phi.PRO[mesh->local_mesh_size + halo_disps[r]],      halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 9, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 9] );

            MPI_Irecv( &phi_grad.U[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 10, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 10] );
            MPI_Irecv( &phi_grad.V[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 11, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 11] );
            MPI_Irecv( &phi_grad.W[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 12, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 12] );
            MPI_Irecv( &phi_grad.P[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 13, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 13] );
			MPI_Irecv( &phi_grad.PP[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 14, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 14] );
			MPI_Irecv( &phi_grad.TE[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 15, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 15] );
			MPI_Irecv( &phi_grad.ED[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 16, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 16] );
			MPI_Irecv( &phi_grad.TEM[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 17, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 17] );
			MPI_Irecv( &phi_grad.FUL[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 18, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 18] );
			MPI_Irecv( &phi_grad.PRO[mesh->local_mesh_size + halo_disps[r]], 3*halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 19, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 19] );
		}
		//1 line NEW code
		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::exchange_A_halos (T *A_phi_component)
    {
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

    template<typename T> void FlowSolver<T>::exchange_S_halos (T *S_phi_component)
    {
        int num_requests = 1;

        MPI_Request send_requests[halo_ranks.size() * num_requests];
        MPI_Request recv_requests[halo_ranks.size() * num_requests];
        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Isend( S_phi_component,      1, halo_mpi_double_datatypes[r],     halo_ranks[r], 0, mpi_config->particle_flow_world, &send_requests[num_requests*r + 0] );
        }

        for ( uint64_t r = 0; r < halo_ranks.size(); r++ )
        {
            MPI_Irecv( &S_phi_component[mesh->local_mesh_size + halo_disps[r]],        halo_sizes[r], MPI_DOUBLE, halo_ranks[r], 0, mpi_config->particle_flow_world, &recv_requests[num_requests*r + 0] );
        }

		MPI_Waitall(num_requests * halo_ranks.size(), send_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_requests * halo_ranks.size(), recv_requests, MPI_STATUSES_IGNORE);
    }

    template<typename T> void FlowSolver<T>::get_neighbour_cells ( const uint64_t recv_id )
    {
        double node_neighbours   = 8;
        const uint64_t cell_size = mesh->cell_size;

        resize_nodes_arrays(node_to_position_map.size() + elements[recv_id] * cell_size + 1 ); // TODO: Move outside loop

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

        // TODO: Can comment this with properly implemented halo exchange. Need cell neighbours for halo and nodes!
        //uint64_t const nsize = node_to_position_map.size();

        /*if (FLOW_SOLVER_DEBUG)
        {
            #pragma ivdep
            for ( uint64_t i = 0;  i < nsize; i++ ) 
            {
                // if (FLOW_SOLVER_DEBUG) check_flow_field_exit ( "INTERP NODAL FINAL ERROR: ", &interp_node_flow_fields[i], &mesh->dummy_flow_field, i );
            }
        }*/
    }

    template<typename T> void FlowSolver<T>::update_flow_field()
    {
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

        // MPI_Barrier(mpi_config->world);
        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Completed Barrrier.\n", mpi_config->rank);

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
                if ( FLOW_SOLVER_DEBUG )  printf("\tFlow block %d: Recieving %d indexes from %d (slot %lu). Max element size %lu. neighbour index rank size %ld array_pointer %p \n", mpi_config->particle_flow_rank, elements[rank_slot], ranks.back(), rank_slot, cell_index_array_size[rank_slot] / sizeof(uint64_t), neighbour_indexes.size(), neighbour_indexes[rank_slot]);

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
                    if ( FLOW_SOLVER_DEBUG )  printf("\tFlow block %d: Processing %d indexes from %d. Local set size %lu (%lu of %lu sets)\n", mpi_config->particle_flow_rank, elements[p], ranks[p], local_particle_node_sets[p].size(), p, local_particle_node_sets.size());
                    
                    get_neighbour_cells (p);
                    processed_neighbours[p] = true;

                }

                all_processed &= processed_neighbours[p];
            }

            time1 += MPI_Wtime(); //1
            time2 -= MPI_Wtime(); //1

            MPI_Test ( &bcast_request, &recvs_complete, MPI_STATUS_IGNORE );
            MPI_Iprobe (MPI_ANY_SOURCE, 0, mpi_config->world, &message_waiting, &statuses[ranks.size()]);

            if ( FLOW_SOLVER_DEBUG && recvs_complete )  printf("\tFlow block %d: Recieved broadcast signal. message_waiting %d recvs_complete %d all_processed %d\n", mpi_config->particle_flow_rank, message_waiting, recvs_complete, all_processed);
            
            all_processed = all_processed & !message_waiting & recvs_complete;
            time2 += MPI_Wtime(); //1
        }

        logger.reduced_recieved_cells += new_cells_set.size();

        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Recieved index sizes.\n", mpi_config->rank);

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

        // MPI_Barrier(mpi_config->world);
        MPI_Barrier(mpi_config->particle_flow_world);
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //6

        
        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //7

        time_stats[time_count++] += MPI_Wtime();
        time_stats[time_count]   -= MPI_Wtime(); //8

        performance_logger.my_papi_stop(performance_logger.update_flow_field_event_counts, &performance_logger.update_flow_field_time);
        
        time_stats[time_count++] += MPI_Wtime();

        if (timestep_count == 1499)
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
                    printf("Time stats %d: %.3f %.2f\n", i, time_stats[i]  / mpi_config->particle_flow_world_size, 100 * time_stats[i] / total_time);
                printf("Total time %f\n", total_time / mpi_config->particle_flow_world_size);

            }
            else{
                for (int i = 0; i < time_count; i++)
                    MPI_Reduce(&time_stats[i], nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);
            }
        }
    }

	template<typename T> void FlowSolver<T>::get_phi_gradient ( T *phi_component, vec<T> *phi_grad_component, bool pressure )
	{
		/*Use the Least Squares method to find the gradient of a phi component*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function get_phi_gradient.\n", mpi_config->rank);

        Eigen::Matrix3d A;
        Eigen::PartialPivLU<Eigen::Matrix3d> A_decomposition;
        Eigen::Vector3d bPhi;

        for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
        {
            const uint64_t cell = block_cell + mesh->local_cells_disp;
            A  = Eigen::Matrix3d::Zero();
            bPhi = Eigen::Vector3d::Zero();
			//printf("locate (%.15f,%.15f,%.15f)\n",mesh->cell_centers[cell].x,mesh->cell_centers[cell].y,mesh->cell_centers[cell].z);
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


                A(0,0) = A(0,0) + dX.x * dX.x;
                A(1,0) = A(1,0) + dX.x * dX.y;
                A(2,0) = A(2,0) + dX.x * dX.z;

                A(0,1) = A(1,0);
                A(1,1) = A(1,1) + dX.y * dX.y;
                A(2,1) = A(2,1) + dX.y * dX.z;

                A(0,2) = A(2,0);
                A(1,2) = A(2,1);
                A(2,2) = A(2,2) + dX.z * dX.z;

                bPhi(0) = bPhi(0) + dX.x * dPhi;
                bPhi(1) = bPhi(1) + dX.y * dPhi;
                bPhi(2) = bPhi(2) + dX.z * dPhi;
            }

            Eigen::Map<Eigen::Vector3d> xPhi(&phi_grad_component[block_cell].x);

            A_decomposition = A.partialPivLu();
            xPhi = A_decomposition.solve(bPhi);
			//printf("PPgrad is (%.15f,%.15f,%.15f)\n",phi_grad.PP[block_cell].x,phi_grad.PP[block_cell].y,phi_grad.PP[block_cell].z);
        }
	}

	template<typename T> void FlowSolver<T>::limit_phi_gradients ()
    {
		T Umax, Umin, Vmax, Vmin, Wmax, Wmin, Pmax, Pmin;
		T Udeltamax, Udeltamin, Vdeltamax, Vdeltamin, Wdeltamax, Wdeltamin, Pdeltamax, Pdeltamin;

		for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
		{
			const uint64_t cell = block_cell + mesh->local_cells_disp;
			const uint64_t shmem_cell = cell - mesh->shmem_cell_disp;
			
			Umax = phi.U[block_cell];
			Umin = phi.U[block_cell];
			Vmax = phi.V[block_cell];
			Vmin = phi.V[block_cell];
			Wmax = phi.W[block_cell];
			Wmin = phi.W[block_cell];
			Pmax = phi.P[block_cell];
			Pmin = phi.P[block_cell];
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
						Umax = max(Umax,phi.U[phi_index1]);
						Umin = min(Umin,phi.U[phi_index1]);
						Vmax = max(Vmax,phi.V[phi_index1]);
	                    Vmin = min(Vmin,phi.V[phi_index1]);
						Wmax = max(Wmax,phi.W[phi_index1]);
		                Wmin = min(Wmin,phi.W[phi_index1]);
						Pmax = max(Pmax,phi.P[phi_index1]);
				        Pmin = min(Pmin,phi.P[phi_index1]);
					}
					else
					{
						Umax = max(Umax,phi.U[phi_index0]);
					    Umin = min(Umin,phi.U[phi_index0]);
						Vmax = max(Vmax,phi.V[phi_index0]);
                        Vmin = min(Vmin,phi.V[phi_index0]);
	                    Wmax = max(Wmax,phi.W[phi_index0]);
		                Wmin = min(Wmin,phi.W[phi_index0]);
			            Pmax = max(Pmax,phi.P[phi_index0]);
				        Pmin = min(Pmin,phi.P[phi_index0]);
					}
				}
				else //Boundary cell
				{
					const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;
					Umax = max(Umax,phi.U[mesh->local_mesh_size + nhalos + boundary_cell]);
                    Umin = min(Umin,phi.U[mesh->local_mesh_size + nhalos + boundary_cell]);
                    Vmax = max(Vmax,phi.V[mesh->local_mesh_size + nhalos + boundary_cell]);
                    Vmin = min(Vmin,phi.V[mesh->local_mesh_size + nhalos + boundary_cell]);
                    Wmax = max(Wmax,phi.W[mesh->local_mesh_size + nhalos + boundary_cell]);
                    Wmin = min(Wmin,phi.W[mesh->local_mesh_size + nhalos + boundary_cell]);
                    Pmax = max(Pmax,phi.P[mesh->local_mesh_size + nhalos + boundary_cell]);
                    Pmin = min(Pmin,phi.P[mesh->local_mesh_size + nhalos + boundary_cell]);
				}
			}
			Udeltamax = Umax - phi.U[block_cell];
			Udeltamin = Umin - phi.U[block_cell];
			Vdeltamax = Vmax - phi.V[block_cell];
            Vdeltamin = Vmin - phi.V[block_cell];
			Wdeltamax = Wmax - phi.W[block_cell];
            Wdeltamin = Wmin - phi.W[block_cell];
			Pdeltamax = Pmin - phi.P[block_cell];
            Pdeltamin = Pmax - phi.P[block_cell];
			
			T Ualpha= 1.0, Valpha= 1.0, Walpha = 1.0, Palpha = 1.0;
			T Ur = 0.0, Vr = 0.0, Wr = 0.0, Pr = 0.0;
			vec<T> ds = {0.0, 0.0, 0.0};
			for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
			{
				const uint64_t face  = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];
				ds = face_centers[face] - mesh->cell_centers[shmem_cell];
				T Udelta_face = dot_product(phi_grad.U[block_cell],ds);
				T Vdelta_face = dot_product(phi_grad.V[block_cell],ds);
				T Wdelta_face = dot_product(phi_grad.W[block_cell],ds);
				T Pdelta_face = dot_product(phi_grad.P[block_cell],ds);
				
				if(abs(Udelta_face) < 0.000006)
				{
					Ur = 1000.0;
				}
				else if(Udelta_face > 0.0)
				{
					Ur = Udeltamax/Udelta_face;
				}
				else
				{
					Ur = Udeltamin/Udelta_face;
				}
				if(abs(Vdelta_face) < 0.000006)
                {
                    Vr = 1000.0;
                }
                else if(Vdelta_face > 0.0) 
                {
                    Vr = Vdeltamax/Vdelta_face;
                }
                else
                {
                    Vr = Vdeltamin/Vdelta_face;
                } 
				if(abs(Wdelta_face) < 0.000006)
                {
                    Wr = 1000.0;
                }
                else if(Wdelta_face > 0.0) 
                {
                    Wr = Wdeltamax/Wdelta_face;
                }
                else
                {
                    Wr = Wdeltamin/Wdelta_face;
                } 
				if(abs(Pdelta_face) < 0.000006)
                {
                    Pr = 1000.0;
                }
                else if(Pdelta_face > 0.0) 
                {
                    Pr = Pdeltamax/Pdelta_face;
                }
                else
                {
                    Pr = Pdeltamin/Pdelta_face;
                }
				Ualpha = min(Ualpha, (pow(Ur,2) + 2.0 * Ur)/(pow(Ur,2) + Ur + 2.0));
				Valpha = min(Valpha, (pow(Vr,2) + 2.0 * Vr)/(pow(Vr,2) + Vr + 2.0));
				Walpha = min(Walpha, (pow(Wr,2) + 2.0 * Wr)/(pow(Wr,2) + Wr + 2.0));
				Palpha = min(Palpha, (pow(Pr,2) + 2.0 * Pr)/(pow(Pr,2) + Pr + 2.0));
			}
			phi_grad.U[block_cell] = Ualpha * phi_grad.U[block_cell];
			phi_grad.V[block_cell] = Valpha * phi_grad.V[block_cell];
			phi_grad.W[block_cell] = Walpha * phi_grad.W[block_cell];
			phi_grad.P[block_cell] = Palpha * phi_grad.P[block_cell];	
		}
	}

    template<typename T> void FlowSolver<T>::get_phi_gradients ()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function get_phi_gradients.\n", mpi_config->rank);
        // NOTE: Currently Least squares is the only method supported

        Eigen::Matrix3d A;             // Independent of phi, reusable for each variable
        Eigen::PartialPivLU<Eigen::Matrix3d> A_decomposition;


        Eigen::Vector3d bU, bV, bW, bP, bTE, bED, bT, bFU, bPR;
        for ( uint64_t block_cell = 0; block_cell < mesh->local_mesh_size; block_cell++ )
        {
			//printf("locate (%.15f,%.15f,%.15f)\n",mesh->cell_centers[block_cell].x,mesh->cell_centers[block_cell].y,mesh->cell_centers[block_cell].z);
            const uint64_t cell = block_cell + mesh->local_cells_disp;
            A  = Eigen::Matrix3d::Zero();
            bU = Eigen::Vector3d::Zero();
            bV = Eigen::Vector3d::Zero();
            bW = Eigen::Vector3d::Zero();
            bP = Eigen::Vector3d::Zero();
			bTE = Eigen::Vector3d::Zero();
			bED = Eigen::Vector3d::Zero();
			bT = Eigen::Vector3d::Zero();
			bFU = Eigen::Vector3d::Zero();
			bPR = Eigen::Vector3d::Zero();
            for ( uint64_t f = 0; f < mesh->faces_per_cell; f++ )
            {
                const uint64_t face  = mesh->cell_faces[block_cell * mesh->faces_per_cell + f];
                
				const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
                const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

                const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
                const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;
				
                T dU, dV, dW, dP, dTE, dED, dT, dFU, dPR;
                vec<T> dX;
                if ( mesh->faces[face].cell1 < mesh->mesh_size )  // Inner cell
                {
                    const uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                    const uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
                    const T mask = ( mesh->faces[face].cell0 == cell ) ? 1. : -1.;

                    dU = mask * ( phi.U[phi_index1] - phi.U[phi_index0] );
                    dV = mask * ( phi.V[phi_index1] - phi.V[phi_index0] );
                    dW = mask * ( phi.W[phi_index1] - phi.W[phi_index0] );
                    dP = mask * ( phi.P[phi_index1] - phi.P[phi_index0] );
					dTE = mask * ( phi.TE[phi_index1] - phi.TE[phi_index0] );
					dED = mask * ( phi.ED[phi_index1] - phi.ED[phi_index0] );
					dT = mask * ( phi.TEM[phi_index1] - phi.TEM[phi_index0] );
					dFU = mask * ( phi.FUL[phi_index1] - phi.FUL[phi_index0] );
					dPR = mask * ( phi.PRO[phi_index1] - phi.PRO[phi_index0] );
                    
					dX = mask * ( mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0] );
                    // Note: ADD code for porous cells here
                } 
                else // Boundary face
                {
					const uint64_t boundary_cell = mesh->faces[face].cell1 - mesh->mesh_size;

                    dU = phi.U[mesh->local_mesh_size + nhalos + boundary_cell] - phi.U[block_cell0];
                    dV = phi.V[mesh->local_mesh_size + nhalos + boundary_cell] - phi.V[block_cell0];
                    dW = phi.W[mesh->local_mesh_size + nhalos + boundary_cell] - phi.W[block_cell0];
                    dP = 0.0;//phi.P[mesh->local_mesh_size + nhalos + boundary_cell] - phi.P[block_cell0]; //dolfyn also enforces dp = 0.0 over boundary //TODO: does this fix
					dTE = phi.TE[mesh->local_mesh_size + nhalos + boundary_cell] - phi.TE[block_cell0];
					dED = phi.ED[mesh->local_mesh_size + nhalos + boundary_cell] - phi.ED[block_cell0];
					dT = phi.TEM[mesh->local_mesh_size + nhalos + boundary_cell] - phi.TEM[block_cell0];
					dFU = phi.FUL[mesh->local_mesh_size + nhalos + boundary_cell] - phi.FUL[block_cell0];
					dPR = phi.PRO[mesh->local_mesh_size + nhalos + boundary_cell] - phi.PRO[block_cell0];

                    dX = face_centers[face] - mesh->cell_centers[shmem_cell0];
                }


                A(0,0) = A(0,0) + dX.x * dX.x;
                A(1,0) = A(1,0) + dX.x * dX.y;
                A(2,0) = A(2,0) + dX.x * dX.z;

                A(0,1) = A(1,0);
                A(1,1) = A(1,1) + dX.y * dX.y;
                A(2,1) = A(2,1) + dX.y * dX.z;

                A(0,2) = A(2,0);
                A(1,2) = A(2,1);
                A(2,2) = A(2,2) + dX.z * dX.z;

                bU(0) = bU(0) + dX.x * dU;
                bU(1) = bU(1) + dX.y * dU;
                bU(2) = bU(2) + dX.z * dU;

                bV(0) = bV(0) + dX.x * dV;
                bV(1) = bV(1) + dX.y * dV;
                bV(2) = bV(2) + dX.z * dV;

                bW(0) = bW(0) + dX.x * dW;
                bW(1) = bW(1) + dX.y * dW;
                bW(2) = bW(2) + dX.z * dW;

                bP(0) = bP(0) + dX.x * dP;
                bP(1) = bP(1) + dX.y * dP;
                bP(2) = bP(2) + dX.z * dP;

				bTE(0) = bTE(0) + dX.x * dTE;
                bTE(1) = bTE(1) + dX.y * dTE;
                bTE(2) = bTE(2) + dX.z * dTE;

				bED(0) = bED(0) + dX.x * dED;
                bED(1) = bED(1) + dX.y * dED;
                bED(2) = bED(2) + dX.z * dED;

				bT(0) = bT(0) + dX.x * dT;
                bT(1) = bT(1) + dX.y * dT;
                bT(2) = bT(2) + dX.z * dT;

				bFU(0) = bFU(0) + dX.x * dFU;
                bFU(1) = bFU(1) + dX.y * dFU;
                bFU(2) = bFU(2) + dX.z * dFU;

				bPR(0) = bPR(0) + dX.x * dPR;
                bPR(1) = bPR(1) + dX.y * dPR;
                bPR(2) = bPR(2) + dX.z * dPR;
            }

            Eigen::Map<Eigen::Vector3d> xU(&phi_grad.U[block_cell].x);
            Eigen::Map<Eigen::Vector3d> xV(&phi_grad.V[block_cell].x);
            Eigen::Map<Eigen::Vector3d> xW(&phi_grad.W[block_cell].x);
            Eigen::Map<Eigen::Vector3d> xP(&phi_grad.P[block_cell].x);
			Eigen::Map<Eigen::Vector3d> xTE(&phi_grad.TE[block_cell].x);
			Eigen::Map<Eigen::Vector3d> xED(&phi_grad.ED[block_cell].x);
			Eigen::Map<Eigen::Vector3d> xT(&phi_grad.TEM[block_cell].x);
			Eigen::Map<Eigen::Vector3d> xFU(&phi_grad.FUL[block_cell].x);
			Eigen::Map<Eigen::Vector3d> xPR(&phi_grad.PRO[block_cell].x);
	

            A_decomposition = A.partialPivLu();
            xU = A_decomposition.solve(bU);
            xV = A_decomposition.solve(bV);
            xW = A_decomposition.solve(bW);
            xP = A_decomposition.solve(bP);
			xTE = A_decomposition.solve(bTE);
			xED = A_decomposition.solve(bED);
			xT = A_decomposition.solve(bT);
			xFU = A_decomposition.solve(bFU);
			xPR = A_decomposition.solve(bPR);
            //printf("Ugrad is (%.15f,%.15f,%.15f)\n",phi_grad.U[block_cell].x,phi_grad.U[block_cell].y,phi_grad.U[block_cell].z);
			//printf("Vgrad is (%.15f,%.15f,%.15f)\n",phi_grad.V[block_cell].x,phi_grad.V[block_cell].y,phi_grad.V[block_cell].z);
			//printf("Wgrad is (%.15f,%.15f,%.15f)\n",phi_grad.W[block_cell].x,phi_grad.W[block_cell].y,phi_grad.W[block_cell].z);
			//printf("Pgrad is (%.15f,%.15f,%.15f)\n",phi_grad.P[block_cell].x,phi_grad.P[block_cell].y,phi_grad.P[block_cell].z);
        }
    }

	template<typename T> void FlowSolver<T>::precomp_mass_flux()
	{
		/*we need the value of mass flux at the inlets and outlets to caculate AU*/
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
                    printf("NEGATIVE OUTFLOW %f\n", face_mass_fluxes[face]);
                    face_mass_fluxes[face] = 1e-15;

                    // ! to be sure reset add. variables too
                    // if( SolveTurbEnergy ) TE(Ncel+ib) = TE(ip)
                    // if( SolveTurbDiss   ) ED(Ncel+ib) = ED(ip)
                    // if( SolveVisc       ) VisEff(Ncel+ib) = VisEff(ip)
                    // if( SolveEnthalpy   ) T(Ncel+ib) = T(ip)
                    // if( SolveScalars    ) SC(Ncel+ib,1:Nscal) = SC(ip,1:Nscal)
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
                //This will cause a NaN
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
                    //TODO:do we ever do something with denisity
                    face_mass_fluxes[face] = ratearea*face_areas[face];//NOTE: assumes one outflow region
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
                S_phi.P[block_cell0] -= face_mass_fluxes[face];
            }
        }
	}

	template<typename T> void FlowSolver<T>::precomp_AU()
	{
		/*Compute AU needed for the first calculation of mass flux*/
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
        for ( uint64_t i = 0 ; i < mesh->local_mesh_size + nhalos; i++ )
        {
            double f = cell_densities[i] * cell_volumes[i] * rdelta;
            A_phi.U[i] += f;
        }
	}

	template<typename T> void FlowSolver<T>::set_up_field()
	{
		/*We need inital values for mass_flux and AU for the first iteration*/
		precomp_mass_flux();
		precomp_AU();
		calculate_mass_flux();
	}


    template<typename T> void FlowSolver<T>::setup_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component )
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function setup_sparse_matrix.\n", mpi_config->rank);

        uint64_t face_count = 0;
        T RURF = 1. / URFactor;

        static double init_time     = 0.0;
        static double res_phi_time  = 0.0;
        static double halo_time     = 0.0;
        static double diagonal_time = 0.0;
        static double compress_time = 0.0;

        init_time -= MPI_Wtime();


        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
            residual[i]         = 0.0;
        }

        init_time    += MPI_Wtime();
        res_phi_time -= MPI_Wtime();
        

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if (mesh->faces[face].cell1 >= mesh->mesh_size)  continue; // Remove when implemented boundary cells. Treat boundary as mesh size

            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

            A_spmatrix.coeffRef(phi_index0, phi_index1) = face_fields[face].cell1;
            A_spmatrix.coeffRef(phi_index1, phi_index0) = face_fields[face].cell0;

            residual[phi_index0] = residual[phi_index0] - face_fields[face].cell1 * phi_component[phi_index1];
            residual[phi_index1] = residual[phi_index1] - face_fields[face].cell0 * phi_component[phi_index0];

            face_count += 2;
            
			A_phi_component[phi_index0] -= face_fields[face].cell1;
            A_phi_component[phi_index1] -= face_fields[face].cell0;

        }

        res_phi_time += MPI_Wtime();
        halo_time    -= MPI_Wtime();

        Eigen::Map<Eigen::VectorXd> A_phi_vector(A_phi_component, mesh->local_mesh_size + nhalos);
        Eigen::Map<Eigen::VectorXd> S_phi_vector(S_phi_component, mesh->local_mesh_size + nhalos);

        exchange_A_halos ( A_phi_component );

        MPI_Barrier(mpi_config->particle_flow_world);


        halo_time    += MPI_Wtime();
        diagonal_time -= MPI_Wtime();

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep 

        for (uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++)
        {
            A_phi_component[i] *= RURF;
            S_phi_component[i] = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];

            A_spmatrix.coeffRef(i, i) = A_phi_component[i];

            residual[i] = residual[i] + S_phi_component[i] - A_phi_component[i] * phi_component[i];
            face_count++;
        }

        diagonal_time += MPI_Wtime();
        compress_time -= MPI_Wtime();

        //exchange_S_halos ( S_phi_component );  // TODO: We don't really need to do this halo exchange.

        //cout << endl << "Phi   : " << endl << phi_vector   << endl;
        // for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        // {
        //     const uint64_t cell = i + mesh->local_cells_disp;
        //     T app = A_phi_component[i];

        //     for (uint64_t f = 0; f < mesh->faces_per_cell; f++)
        //     {
        //         T face_value = 0.0;
        //         uint64_t face  = mesh->cell_faces[i * mesh->faces_per_cell + f];

        //         const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
        //         const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

        //         if (mesh->faces[face].cell1 == MESH_BOUNDARY)  continue;  // Remove when implemented boundary cells. Treat boundary as mesh size
                
        //         uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
        //         uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

        //         // if ( NOT BOUNDARY ) // Calculations are different for inlets, outlets, walls etc.
        //         // {
        //             if ( mesh->faces[face].cell0 == cell )
        //             {
        //                 face_value = face_fields[face].cell1;

        //                 A_spmatrix.coeffRef(i, phi_index1) = face_value;

        //                 residual[i] = residual[i] - face_value * phi_component[phi_index1];
        //                 face_count++;
        //             }
        //             else if ( mesh->faces[face].cell1 == cell )
        //             {
        //                 face_value = face_fields[face].cell0;

        //                 A_spmatrix.coeffRef(i, phi_index0) = face_value;

        //                 residual[i] = residual[i] - face_value * phi_component[phi_index0];
        //                 face_count++;
        //             }
        //             app = app - face_value;
        //         // }
        //     }

        //     A_phi_component[i]  = app * RURF;
        //     S_phi_component[i]  = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];
            
        //     A_spmatrix.coeffRef(i, i) = A_phi_component[i];
        //     face_count++;

        //     residual[i] = residual[i] + S_phi_component[i] - A_phi_component[i] * phi_component[i];
        // }

        A_spmatrix.makeCompressed();

        compress_time += MPI_Wtime();

        if ( mpi_config->particle_flow_rank == 0 && timestep_count == 1499)
        {
            printf("SETUP Init  time:         %7.2fs\n", init_time  );
            printf("SETUP res_phi_time  time: %7.2fs\n", res_phi_time  );
            printf("SETUP Halo time:          %7.2fs\n", halo_time );
            printf("SETUP Diagonal time:      %7.2fs\n", diagonal_time );
            printf("SETUP compress time:      %7.2fs\n", compress_time );
        }
    }

    template<typename T> void FlowSolver<T>::update_sparse_matrix ( T URFactor, T *A_phi_component, T *phi_component, T *S_phi_component )
    {
        // The idea for this function is to reduce the number of insertions into the sparse A matrix, 
        // given that face_fields (RFace) is constant for U, V and W. Possible to write app to seperate array and then only do one halo exchange here.

        uint64_t face_count = 0;
        T RURF = 1. / URFactor;

        static double init_time     = 0.0;
        static double res_phi_time  = 0.0;
        static double halo_time     = 0.0;
        static double diagonal_time = 0.0;

        init_time -= MPI_Wtime();

        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size; i++)
        {
            residual[i]         = 0.0;
            A_phi_component[i] *= RURF;
        }

        init_time    += MPI_Wtime();
        res_phi_time -= MPI_Wtime();

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            if ( mesh->faces[face].cell1 >= mesh->mesh_size )  continue; // Remove when implemented boundary cells. Treat boundary as mesh size

            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

            residual[phi_index0] = residual[phi_index0] - face_fields[face].cell1 * phi_component[phi_index1];
            residual[phi_index1] = residual[phi_index1] - face_fields[face].cell0 * phi_component[phi_index0];

            face_count += 2;

            A_phi_component[phi_index0] -= RURF * face_fields[face].cell1;
            A_phi_component[phi_index1] -= RURF * face_fields[face].cell0;
        }

        res_phi_time += MPI_Wtime();
        halo_time    -= MPI_Wtime();

        exchange_A_halos (A_phi_component);

        halo_time    += MPI_Wtime();
        diagonal_time -= MPI_Wtime();

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep 
        for (uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++)
        {
            S_phi_component[i] = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];
            A_spmatrix.coeffRef(i, i) = A_phi_component[i];

            residual[i] = residual[i] + S_phi_component[i] - A_phi_component[i] * phi_component[i];
            face_count++;
        }

        diagonal_time += MPI_Wtime();

        if (mpi_config->particle_flow_rank == 0 && timestep_count == 1499)
        {
            printf("UPDATE Init  time:         %7.2fs\n", init_time  );
            printf("UPDATE res_phi_time  time: %7.2fs\n", res_phi_time  );
            printf("UPDATE Halo time:          %7.2fs\n", halo_time );
            printf("UPDATE Diagonal time:      %7.2fs\n", diagonal_time );
        }
    }


    template<typename T> void FlowSolver<T>::solve_sparse_matrix ( T *phi_component, T *S_phi_component )
    {
        static double init_time     = 0.0;
        static double compute_time  = 0.0;
        static double solve_time     = 0.0;

        init_time -= MPI_Wtime();

        Eigen::Map<Eigen::VectorXd> S_phi_vector(S_phi_component, mesh->local_mesh_size + nhalos);
        Eigen::Map<Eigen::VectorXd>   phi_vector(phi_component,   mesh->local_mesh_size + nhalos);

        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_sparse_matrix A = (%lu %lu) x = (%lu %lu) b = (%lu %lu).\n", mpi_config->rank, 
                                                                                                                                       A_spmatrix.rows(),   A_spmatrix.cols(), 
                                                                                                                                       phi_vector.rows(),   phi_vector.cols(),
                                                                                                                                       S_phi_vector.rows(), S_phi_vector.cols());
        init_time += MPI_Wtime();
        compute_time -= MPI_Wtime();

        eigen_solver.setTolerance(0.001);

        check_array_nan("S_phi_vector", S_phi_component, mesh->local_mesh_size + nhalos, mpi_config, timestep_count);

		cout << mpi_config->particle_flow_rank << endl << "A Matrix     : " << endl << Eigen::MatrixXd(A_spmatrix) << endl;
		cout << endl << "RHS : " << endl << S_phi_vector << endl;	

        eigen_solver.compute(A_spmatrix);

        compute_time += MPI_Wtime();
        solve_time -= MPI_Wtime();

        phi_vector = eigen_solver.solveWithGuess(S_phi_vector, phi_vector);

        phi_vector = eigen_solver.solve(S_phi_vector);

		cout << endl << "Result : " << endl << phi_vector << endl;

        check_array_nan("Phi_vector", phi_component, mesh->local_mesh_size + nhalos + mesh->boundary_cells_size, mpi_config, timestep_count);

        solve_time += MPI_Wtime();

        if (mpi_config->particle_flow_rank == 0 && timestep_count == 1499)
        {
            printf("SOLVE Init  time:          %7.2fs\n", init_time  );
            printf("SOLVE compute  time:       %7.2fs\n", compute_time  );
            printf("SOLVE solve time:          %7.2fs\n", solve_time );
        }
    }

    template<typename T> void FlowSolver<T>::calculate_flux_UVW()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_flux_UVW.\n", mpi_config->rank);

        T pe0 =  9999.;
        T pe1 = -9999.;
        // TotalForce = 0.0

        T GammaBlend = 0.0; // NOTE: Change when implemented other differencing schemes.

        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

            if ( mesh->faces[face].cell1 < mesh->mesh_size )  // INTERNAL
            {
				//printf("identify (%.15f,%.15f,%.15f) and (%.15f,%.15f,%.15f)\n",mesh->cell_centers[block_cell0].x,mesh->cell_centers[block_cell0].y,mesh->cell_centers[block_cell0].z,mesh->cell_centers[block_cell1].x,mesh->cell_centers[block_cell1].y,mesh->cell_centers[block_cell1].z); 
                uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
                const T lambda0 = face_lambdas[face];    // dist(cell_center0, face_center) / dist(cell_center0, cell_center1)
                const T lambda1 = 1.0 - lambda0;         // dist(face_center,  cell_center1) / dist(cell_center0, cell_center1)
				//printf("grad 1 (%.15f,%.15f,%.15f) grad 2 (%.15f,%.15f,%.15f)\n",phi_grad.U[phi_index0].x,phi_grad.U[phi_index0].y,phi_grad.U[phi_index0].z,phi_grad.U[phi_index1].x,phi_grad.U[phi_index1].y,phi_grad.U[phi_index1].z);
                // T Uac     = phi.U[block_cell0] * lambda0 + phi.U[ip] * lambda1;
                // T Vac     = phi.V[block_cell0] * lambda0 + phi.V[ip] * lambda1;
                // T Wac     = phi.W[block_cell0] * lambda0 + phi.W[ip] * lambda1;

                const vec<T> dUdXac  =   phi_grad.U[phi_index0] * lambda0 + phi_grad.U[phi_index1] * lambda1;
                const vec<T> dVdXac  =   phi_grad.V[phi_index0] * lambda0 + phi_grad.V[phi_index1] * lambda1;
                const vec<T> dWdXac  =   phi_grad.W[phi_index0] * lambda0 + phi_grad.W[phi_index1] * lambda1;
				                
				T Visac   = effective_viscosity * lambda0 + effective_viscosity * lambda1;
                T VisFace = Visac * face_rlencos[face];
 
                vec<T> Xpn     = mesh->cell_centers[shmem_cell1] - mesh->cell_centers[shmem_cell0];

                // NOTE: ADD other differencing schemes. For now we just use Upwind Differencing Scheme (UDS)

                // call SelectDiffSchemeVector();

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

				//printf("test visc %.15f and norm (%.15f,%.15f,%.15f)\n",Visac,face_normals[face].x,face_normals[face].y,face_normals[face].z);
                // ! implicit lower order (simple upwind)
                // ! convective and diffusive fluxes

                const T fmin = min( face_mass_fluxes[face], 0.0 );
                const T fmax = max( face_mass_fluxes[face], 0.0 );
		
				//printf("fmax is %f, f min is %f\n",fmax,fmin);

				
                const T fuci = fmin * phi.U[phi_index0] + fmax * phi.U[phi_index1];
                const T fvci = fmin * phi.V[phi_index0] + fmax * phi.V[phi_index1];
                const T fwci = fmin * phi.W[phi_index0] + fmax * phi.W[phi_index1];

                const T fudi = VisFace * dot_product( dUdXac , Xpn );
				//printf("VisFace %.15f dudx (%.15f,%.15f,%.15f) Xpn (%.15f,%.15f,%.15f)\n",VisFace,dUdXac.x,dUdXac.y,dUdXac.z,Xpn.x,Xpn.y,Xpn.z);
                const T fvdi = VisFace * dot_product( dVdXac , Xpn );
				//printf("VisFace %.15f dVdx (%.15f,%.15f,%.15f) Xpn (%.15f,%.15f,%.15f)\n",VisFace,dVdXac.x,dVdXac.y,dVdXac.z,Xpn.x,Xpn.y,Xpn.z);
                const T fwdi = VisFace * dot_product( dWdXac , Xpn );
				//printf("VisFace %.15f dWdx (%.15f,%.15f,%.15f) Xpn (%.15f,%.15f,%.15f)\n",VisFace,dWdXac.x,dWdXac.y,dWdXac.z,Xpn.x,Xpn.y,Xpn.z);
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
				//printf("Rface is %.15f and %.15f\n",VisFace, face_mass_fluxes[face]); 
                face_fields[face].cell0 = -VisFace - max( face_mass_fluxes[face] , 0.0 );  // P (e);
                face_fields[face].cell1 = -VisFace + min( face_mass_fluxes[face] , 0.0 );  // N (w);

                


                const T blend_u = GammaBlend * ( fuce - fuci );
                const T blend_v = GammaBlend * ( fvce - fvci );
                const T blend_w = GammaBlend * ( fwce - fwci );

                // ! assemble the two source terms
                // ! Is it faster to just write to source term vectors?? Can we vectorize this function??
                //printf("identify (%.15f,%.15f,%.15f) and (%.15f,%.15f,%.15f)\n",mesh->cell_centers[block_cell0].x,mesh->cell_centers[block_cell0].y,mesh->cell_centers[block_cell0].z,mesh->cell_centers[block_cell1].x,mesh->cell_centers[block_cell1].y,mesh->cell_centers[block_cell1].z);
				//printf("internal U values are blend %.15f fude %.15f, fudi %.15f\n",blend_u,fude,fudi); 
				S_phi.U[phi_index0] = S_phi.U[phi_index0] - blend_u + fude - fudi;
				//printf("internal V values are blend %.22f fvde %.22f, fvdi %.22f\n",blend_v,fvde,fvdi);
                S_phi.V[phi_index0] = S_phi.V[phi_index0] - blend_v + fvde - fvdi;
				//printf("internal W values are blend %.22f fwde %.22f, fwdi %.22f\n",blend_w,fwde,fwdi);
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
				
					//printf("inlet f is %4.18f Uface %4.18f\n",f,UFace);
	
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
                        printf("Error: neg. massflux in outlet\n");
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
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_UVW.\n", mpi_config->rank);


        static double init_time  = 0.0;
        static double flux_time  = 0.0;
        static double setup_time = 0.0;
        static double solve_time = 0.0;
		
        init_time -= MPI_Wtime(); 

        // Initialise A_phi.U, S_phi.U and Aval vectors to 0.
        #pragma ivdep 
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            A_phi.U[i] = 0.0;
            A_phi.V[i] = 0.0;
            A_phi.W[i] = 0.0;

            S_phi.U[i] = 0.0;
            S_phi.V[i] = 0.0;
            S_phi.W[i] = 0.0;
        }

        init_time += MPI_Wtime();
        flux_time -= MPI_Wtime();

        // calculate fluxes through all inner faces
        calculate_flux_UVW ();
		/*for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++ )
		{
			printf("locate (%4.18f,%4.18f,%4.18f)\n", mesh->cell_centers[i].x,mesh->cell_centers[i].y,mesh->cell_centers[i].z);
			printf("Su: %.15f \n",S_phi.U[i]);
            printf("Sv: %.22f \n",S_phi.V[i]);
            printf("Sw: %.22f \n",S_phi.W[i]);
            printf("Au: %.15f \n",A_phi.U[i]);
            printf("Av: %.15f \n",A_phi.V[i]);
            printf("Aw: %.15f \n\n",A_phi.W[i]);
        }
        printf("\n");*/

        // Gravity force (enthalpy)
		for ( uint64_t i = 0 ; i < mesh->local_mesh_size + nhalos; i++ )
		{
			T BodyForce = -0.001*cell_densities[i]*cell_volumes[i]*(phi.TEM[i] - 273);
			T gravity[3] = {0.0, -9.81, 0.0};

			S_phi.U[i] += gravity[0]*BodyForce;
			S_phi.V[i] += gravity[1]*BodyForce;
			S_phi.W[i] += gravity[2]*BodyForce;
		}
        
		
        // Pressure force
		for ( uint64_t i = 0 ; i < mesh->local_mesh_size + nhalos; i++ )
		{
			S_phi.U[i] -= phi_grad.P[i].x*cell_volumes[i];
			S_phi.V[i] -= phi_grad.P[i].y*cell_volumes[i];
			S_phi.W[i] -= phi_grad.P[i].z*cell_volumes[i];
		}

        // If Transient and Euler
        double rdelta = 1.0 / delta;
		#pragma ivdep
        for ( uint64_t i = 0 ; i < mesh->local_mesh_size + nhalos; i++ )
        {
            double f = cell_densities[i] * cell_volumes[i] * rdelta;

            S_phi.U[i] += f * phi.U[i];
            S_phi.V[i] += f * phi.V[i];
            S_phi.W[i] += f * phi.W[i];

            A_phi.U[i] += f;
            A_phi.V[i] += f;
            A_phi.W[i] += f;
        }
       
		/*for ( uint64_t i = 0 ; i < mesh->local_mesh_size; i++ )
        {
            printf("locate (%4.18f,%4.18f,%4.18f)\n", mesh->cell_centers[i].x,mesh->cell_centers[i].y,mesh->cell_centers[i].z);
            printf("Su: %.15f \n",S_phi.U[i]);
            printf("Sv: %.22f \n",S_phi.V[i]);
            printf("Sw: %.22f \n",S_phi.W[i]);
            printf("Au: %.15f \n",A_phi.U[i]);
            printf("Av: %.15f \n",A_phi.V[i]);
            printf("Aw: %.15f \n\n",A_phi.W[i]);
        }
        printf("\n");*/
 
		const double UVW_URFactor = 0.5;
		
        flux_time  += MPI_Wtime();
        setup_time -= MPI_Wtime();
        setup_sparse_matrix (UVW_URFactor, A_phi.U, phi.U, S_phi.U);   
        setup_time  += MPI_Wtime();
        solve_time  -= MPI_Wtime();
		MPI_Barrier(mpi_config->particle_flow_world);
		for(int i = 0; i < mpi_config->particle_flow_world_size; i++)
        {
			if(i == mpi_config->particle_flow_rank)
            {
				solve_sparse_matrix (phi.U, S_phi.U);
			}
			MPI_Barrier(mpi_config->particle_flow_world);
        }
		MPI_Barrier(mpi_config->particle_flow_world);

        //MPI_Barrier(mpi_config->particle_flow_world); 
        //solve_sparse_matrix (phi.U, S_phi.U);
        //MPI_Barrier(mpi_config->particle_flow_world); 
		/*
        setup_time  -= MPI_Wtime();
        solve_time  += MPI_Wtime();  
        update_sparse_matrix (UVW_URFactor, A_phi.V, phi.V, S_phi.V); 
        setup_time  += MPI_Wtime();
        solve_time  -= MPI_Wtime();     

        MPI_Barrier(mpi_config->particle_flow_world); 
        solve_sparse_matrix (phi.V, S_phi.V);
        MPI_Barrier(mpi_config->particle_flow_world);  


        setup_time  -= MPI_Wtime();
        solve_time  += MPI_Wtime();  
        update_sparse_matrix (UVW_URFactor, A_phi.W, phi.W, S_phi.W);
        setup_time  += MPI_Wtime();
        solve_time  -= MPI_Wtime();  

        MPI_Barrier(mpi_config->particle_flow_world); 
        solve_sparse_matrix (phi.W, S_phi.W);
        MPI_Barrier(mpi_config->particle_flow_world);
		*/
        
        solve_time += MPI_Wtime();


        if (mpi_config->particle_flow_rank == 0 && timestep_count == 1499)
        {
            printf("TOTAL Init  time: %7.2fs\n", init_time  );
            printf("TOTAL Flux  time: %7.2fs\n", flux_time  );
            printf("TOTAL Setup time: %7.2fs\n", setup_time );
            printf("TOTAL Solve time: %7.2fs\n", solve_time );
        }
    }

	template<typename T> void FlowSolver<T>::update_mass_flux()
	{
	/*Update the mass flux for the internal faces. Used for the incremental
	update of the velocity and pressure fields*/
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function update_mass_flux.\n", mpi_config->rank);
		exchange_phi_halos();
		MPI_Barrier(mpi_config->particle_flow_world);

		for ( uint64_t face = 0; face < mesh->faces_size; face++ )
		{
			const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;			

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;
			if ( mesh->faces[face].cell1 >= mesh->mesh_size ) continue;
			//printf("locate is (%.15f,%.15f,%.15f) and (%.15f,%.15f,%.15f)\n",mesh->cell_centers[shmem_cell0].x,mesh->cell_centers[shmem_cell0].y,mesh->cell_centers[shmem_cell0].z,mesh->cell_centers[shmem_cell1].x,mesh->cell_centers[shmem_cell1].y,mesh->cell_centers[shmem_cell1].z);
			
			uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

				
			const vec<T> Xpac = face_centers[face] - dot_product(face_centers[face] - mesh->cell_centers[shmem_cell0], normalise(face_normals[face]))*normalise(face_normals[face]);
            const vec<T> Xnac = face_centers[face] - dot_product(face_centers[face] - mesh->cell_centers[shmem_cell1], normalise(face_normals[face]))*normalise(face_normals[face]);	
			
			const vec<T> Xn = Xnac -  mesh->cell_centers[shmem_cell1];
			const vec<T> Xp = Xpac -  mesh->cell_centers[shmem_cell0];

			T fact = face_fields[face].cell0;
			
			const T dpx  = phi_grad.PP[phi_index1].x * Xn.x - phi_grad.PP[phi_index0].x * Xp.x;
            const T dpy  = phi_grad.PP[phi_index1].y * Xn.y - phi_grad.PP[phi_index0].y * Xp.y;
            const T dpz  = phi_grad.PP[phi_index1].z * Xn.z - phi_grad.PP[phi_index0].z * Xp.z;
			
			const T fc = fact * (dpx + dpy + dpz) * 0.8;

			face_mass_fluxes[face] += fc;

			//printf("fc is %.22f\n",fc);

			S_phi.P[block_cell0] -= fc;
			S_phi.P[block_cell1] += fc;		
		}
	}

    template<typename T> void FlowSolver<T>::calculate_mass_flux()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_mass_flux.\n", mpi_config->rank);

        exchange_phi_halos();
        MPI_Barrier(mpi_config->particle_flow_world);

        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

            const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

            if ( mesh->faces[face].cell1 < mesh->mesh_size )  // INTERNAL
            {
				//printf("locate is (%.15f,%.15f,%.15f) and (%.15f,%.15f,%.15f)\n",mesh->cell_centers[shmem_cell0].x,mesh->cell_centers[shmem_cell0].y,mesh->cell_centers[shmem_cell0].z,mesh->cell_centers[shmem_cell1].x,mesh->cell_centers[shmem_cell1].y,mesh->cell_centers[shmem_cell1].z);
                uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
                uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;


                const T lambda0 = 0.5;//face_lambdas[face];
                const T lambda1 = 0.5;//1.0 - lambda0;

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
				//printf("den %.15f * ar %.15f * facn %.15f + den %.15f * ar %.15f * facp %.15f\n",cell_densities[phi_index0], ApV0, lambda0, cell_densities[phi_index1], ApV1, lambda1); 
                const T volume_avg = cell_volumes[phi_index0] * lambda0 + cell_volumes[phi_index1] * lambda1;
                ApV  = ApV * face_areas[face] * volume_avg/dot_product(Xpn2, normalise(face_normals[face]));
                //printf("apv %.15f, area %.15f, factV %.15f, dot %.15f\n",ApV,face_areas[face],volume_avg,dot_product(Xpn2, normalise(face_normals[face])));
				const T dpx  = ( phi_grad.P[phi_index1].x * lambda1 + phi_grad.P[phi_index0].x * lambda0) * Xpn.x; 
                const T dpy  = ( phi_grad.P[phi_index1].y * lambda1 + phi_grad.P[phi_index0].y * lambda0) * Xpn.y;  
                const T dpz  = ( phi_grad.P[phi_index1].z * lambda1 + phi_grad.P[phi_index0].z * lambda0) * Xpn.z; 

				//printf("face fields are %.15f\n",-ApV);
                face_fields[face].cell0 = -ApV;
                face_fields[face].cell1 = -ApV;
				
				face_mass_fluxes[face] -= ApV * ((cell1_P - cell0_P) - dpx - dpy - dpz);
				//printf("mass flux internal final is %.15f\n",face_mass_fluxes[face]);
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
                    const T Din = 1.20000005;

					//printf("dens is 1.20000005 vel is (%.5f,%.5f,%.5f) normal (%.12f,%.12f,%.12f)\n",vel_inward.x,vel_inward.y,vel_inward.z,face_normals[face].x,face_normals[face].y,face_normals[face].z);
                    face_mass_fluxes[face] = Din * dot_product( vel_inward, face_normals[face] );
					//printf("inlet mass flux %.22f\n",face_mass_fluxes[face]);
					//printf("inlet sphi before %.22f\n",S_phi.P[block_cell0]);
					S_phi.P[block_cell0] = S_phi.P[block_cell0] - face_mass_fluxes[face];
					//printf("inlet sphi after %.22f\n",S_phi.P[block_cell0]);
                }
                else if( boundary_type == OUTLET )
                {
                    const vec<T> vel_outward = { phi.U[block_cell0], 
                                                 phi.V[block_cell0], 
                                                 phi.W[block_cell0] };
					const T Din = 1.2;

                    face_mass_fluxes[face] = Din * dot_product(vel_outward, face_normals[face]);
					//printf("outlet mass flux %.22f\n",face_mass_fluxes[face]);
					// !
                    // ! For an outlet face_mass_fluxes must be 0.0 or positive
                    // !
                    if( face_mass_fluxes[face] < 0.0 )
                    {
                        cout << "vel " << print_vec(vel_outward) << " normal " << print_vec(face_normals[face]) << " mass " << face_mass_fluxes[face] << endl;
                        printf("NEGATIVE OUTFLOW %f\n", face_mass_fluxes[face]);
                        face_mass_fluxes[face] = 1e-15;

                        // ! to be sure reset add. variables too
                        // if( SolveTurbEnergy ) TE(Ncel+ib) = TE(ip)
                        // if( SolveTurbDiss   ) ED(Ncel+ib) = ED(ip)
                        // if( SolveVisc       ) VisEff(Ncel+ib) = VisEff(ip)
                        // if( SolveEnthalpy   ) T(Ncel+ib) = T(ip)
                        // if( SolveScalars    ) SC(Ncel+ib,1:Nscal) = SC(ip,1:Nscal)
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
		T FlowFact[count_out];
		int step = 0;
		for(int i = 0; i < count_out; i++){
			if(FlowOut == 0.0)
			{
				//This will cause a	NaN
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
					//TODO:do we ever do something with denisity
					face_mass_fluxes[face] = ratearea*face_areas[face];//NOTE: assumes one outflow region
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
				//printf("final mass flux %.22f\n",face_mass_fluxes[face]);
				step ++;
				//printf("fact is %.15f\n",fact);
				 	
				phi.U[mesh->local_mesh_size + nhalos + boundary_cell] *= fact;
				phi.V[mesh->local_mesh_size + nhalos + boundary_cell] *= fact;
				phi.W[mesh->local_mesh_size + nhalos + boundary_cell] *= fact; 

				const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
                //printf("outlet sphi before %.22f\n",S_phi.P[block_cell0]);
				S_phi.P[block_cell0] -= face_mass_fluxes[face];
				//printf("outlet sphi after %.22f\n",S_phi.P[block_cell0]);
			}
		}			
    }

    template<typename T> void FlowSolver<T>::setup_pressure_matrix()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function setup_pressure_matrix.\n", mpi_config->rank);

        #pragma ivdep 
        for ( uint64_t face = 0; face < mesh->faces_size; face++ )
        {
            const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
            const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;

			//const uint64_t shmem_cell0 = mesh->faces[face].cell0 - mesh->shmem_cell_disp;
            //const uint64_t shmem_cell1 = mesh->faces[face].cell1 - mesh->shmem_cell_disp;

			//printf("locate is (%.15f,%.15f,%.15f) and (%.15f,%.15f,%.15f)\n",mesh->cell_centers[shmem_cell0].x,mesh->cell_centers[shmem_cell0].y,mesh->cell_centers[shmem_cell0].z,mesh->cell_centers[shmem_cell1].x,mesh->cell_centers[shmem_cell1].y,mesh->cell_centers[shmem_cell1].z);
            if (mesh->faces[face].cell1 >= mesh->mesh_size)  continue; // Remove when implemented boundary cells. Treat boundary as mesh size

            uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
            uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;

            A_spmatrix.coeffRef(phi_index0, phi_index1) = face_fields[face].cell1;
            A_spmatrix.coeffRef(phi_index1, phi_index0) = face_fields[face].cell0;
			
			//printf("taking %.22f from cell %lu and adding %.22f to cell %lu\n",face_mass_fluxes[face],phi_index0+1,face_mass_fluxes[face],phi_index1+1);
			
            S_phi.P[phi_index0] -= face_mass_fluxes[face];
            S_phi.P[phi_index1] += face_mass_fluxes[face];
            
			A_phi.P[phi_index0] -= face_fields[face].cell1;
            A_phi.P[phi_index1] -= face_fields[face].cell0;

		}
        exchange_A_halos ( A_phi.P );
        exchange_S_halos ( S_phi.P );

        MPI_Barrier(mpi_config->particle_flow_world);

        // Add A matrix diagonal after exchanging halos
        #pragma ivdep
        for (uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++)
        {
            A_spmatrix.coeffRef(i, i) = A_phi.P[i];
        }

        A_spmatrix.makeCompressed();
    }

    template<typename T> void FlowSolver<T>::solve_pressure_matrix()
    {
        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_pressure_matrix.\n", mpi_config->rank);

        Eigen::Map<Eigen::VectorXd> S_phi_vector(S_phi.P, mesh->local_mesh_size + nhalos);
        Eigen::Map<Eigen::VectorXd>   phi_vector(phi.PP,   mesh->local_mesh_size + nhalos);

        if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function solve_pressure_matrix A = (%lu %lu) x = (%lu %lu) b = (%lu %lu).\n", mpi_config->rank, 
                                                                                                                                         A_spmatrix.rows(),   A_spmatrix.cols(), 
                                                                                                                                         phi_vector.rows(),   phi_vector.cols(),
                                                                                                                                         S_phi_vector.rows(), S_phi_vector.cols());

        MPI_Barrier(mpi_config->particle_flow_world);

        eigen_solver.setTolerance(0.005);
        eigen_solver.compute(A_spmatrix);
        phi_vector = eigen_solver.solve(S_phi_vector);

        //auto check = Eigen::MatrixXd(A_spmatrix) * phi_vector;
	
		//cout << "check results" << check << endl;
        //cout << "A Matrix     : " << endl << Eigen::MatrixXd(A_spmatrix) << endl;

        //cout << endl << "S_Phi : " << endl << S_phi_vector << endl;
		//cout << endl << "Phi   : " << endl << phi_vector   << endl;
    }

	template<typename T> void FlowSolver<T>::Update_P_at_boundaries(T *phi_component)
	{
	/*Update the value of a Pressure phi component at the boundaries, with
	the value of phi for the internal side of the face. Used in the progressive update 
	of Pressure and Velocity fields in compute pressure.*/ 
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
		//TODO: communication check
		
		T Loop_num = 0;
		bool Loop_continue = true;
		T Pressure_correction_max = 0;
		T Pressure_correction_ref = 0;
		if (FLOW_SOLVER_DEBUG)  printf("\tRank %d: Running function calculate_pressure.\n", mpi_config->rank);
        
		#pragma ivdep 
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            A_phi.P[i] = 0.0;
            S_phi.P[i] = 0.0;
        }

		#pragma ivdep
		for ( uint64_t face = 0; face < mesh->faces_size; face++)
		{
			face_mass_fluxes[face] = 0.0;
			face_fields[face].cell0 = 0.0;
			face_fields[face].cell1 = 0.0;		
		}

        calculate_mass_flux();  //Compute the uncorrected mass fluxes at every faces

		for ( uint64_t face = 0; face < mesh->faces_size; face++)
        {
			//printf("locate (%.8f,%.8f,%.8f) and (%.8f,%.8f,%.8f)\n", mesh->cell_centers[mesh->faces[face].cell0].x, mesh->cell_centers[mesh->faces[face].cell0].y, mesh->cell_centers[mesh->faces[face].cell0].z, mesh->cell_centers[mesh->faces[face].cell1].x,mesh->cell_centers[mesh->faces[face].cell1].y,mesh->cell_centers[mesh->faces[face].cell1].z);
            //printf("after mass flux is %.22f\n",face_mass_fluxes[face]);
            //printf("after face fields is %.22f and %.22f\n",face_fields[face].cell0,face_fields[face].cell1);
        }

		/*printf("S_phi ");
        #pragma ivdep
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            printf("%.15f ",S_phi.P[i]);
        }
        printf("\n");*/
	
		setup_pressure_matrix(); //Set up Sp and Ap for the initial pressure solve.


		/*printf("A_phi ");
		#pragma ivdep
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            printf("%.15f ",A_phi.P[i]);
		}            
		printf("\nS_phi ");
		#pragma ivdep
		for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
		{
			printf("%.15f ",S_phi.P[i]);
        }
		printf("\n");*/
	
		while(Loop_continue)  //conduct a number of improvements to the pressure and velocity fields.
		{
			Loop_num++;
			
			solve_pressure_matrix(); //Compute pressure correction
			/*if(Loop_num == 1){
				phi.PP[0] = 0.12009795721005093;
				phi.PP[1] = 0.046391111053743779;
				phi.PP[2] = -0.0408160677443343348;
				phi.PP[3] = -0.12835557100986023;
				phi.PP[4] = 0.12004155837395714;
				phi.PP[5] = 0.0463189297217485724;
			    phi.PP[6] = -0.041058095875166796;
				phi.PP[7] = -0.12885626226613001;
		        phi.PP[8] = 0.12004155837395714;
				phi.PP[9] = 0.0463189297217485724;
		        phi.PP[10] = -0.041058095875166796;
			    phi.PP[11] = -0.12885626226613001;
				phi.PP[12] = 0.12009795166577833;
				phi.PP[13] = 0.0463911117166303025;
		        phi.PP[14] = -0.0408160698725814064;
			    phi.PP[15] = -0.12835557185112204;
			}else{
				phi.PP[0] = 0.0;
                phi.PP[1] = 0.0;
                phi.PP[2] = 0.0;
                phi.PP[3] = 0.0;
                phi.PP[4] = 0.0;
                phi.PP[5] = 0.0;
                phi.PP[6] = 0.0;
                phi.PP[7] = 0.0;
                phi.PP[8] = 0.0;
                phi.PP[9] = 0.0;
                phi.PP[10] = 0.0;
                phi.PP[11] = 0.0;
                phi.PP[12] = 0.0;
                phi.PP[13] = 0.0;
                phi.PP[14] = 0.0;
                phi.PP[15] = 0.0;
			}*/
			Pressure_correction_max = phi.PP[0];
			for ( uint64_t i = 1; i < mesh->local_mesh_size + nhalos + mesh->boundary_cells_size; i++ )
			{
				if(abs(phi.PP[i]) > Pressure_correction_max) 
				{
					Pressure_correction_max = abs(phi.PP[i]);
				}
			}
			if(Loop_num == 1)
			{
				Pressure_correction_ref = Pressure_correction_max;
			}

			Update_P_at_boundaries(phi.PP); //Update boundary pressure
    
            get_phi_gradient(phi.PP, phi_grad.PP, true); //Compute gradient of correction.

			exchange_phi_halos(); //exchange so phi_grad.PP is correct at halos 

			for ( uint64_t face = 0; face < mesh->faces_size; face++ ){
				const uint64_t block_cell0 = mesh->faces[face].cell0 - mesh->local_cells_disp;
				const uint64_t block_cell1 = mesh->faces[face].cell1 - mesh->local_cells_disp;
				if( mesh->faces[face].cell1 >= mesh->mesh_size ) continue;
				//internal cells
				//printf("locate is (%.15f,%.15f,%.15f) and (%.15f,%.15f,%.15f)\n",mesh->cell_centers[block_cell0].x,mesh->cell_centers[block_cell0].y,mesh->cell_centers[block_cell0].z,mesh->cell_centers[block_cell1].x,mesh->cell_centers[block_cell1].y,mesh->cell_centers[block_cell1].z);
				uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
				uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
				//printf("mass before is %.22f\n",face_mass_fluxes[face]);
				//printf("rafce is %.22f, PP(IN) is %.22f PP(ip) is %.22f\n",face_fields[face].cell0,phi.PP[phi_index1],phi.PP[phi_index0]);
				//printf("total is %.22f\n",(face_fields[face].cell0*(phi.PP[phi_index1] - phi.PP[phi_index0])));
				face_mass_fluxes[face] += (face_fields[face].cell0*(phi.PP[phi_index1] - phi.PP[phi_index0]));
				//printf("mass after is %.22f\n",face_mass_fluxes[face]);
			}
	
			T Pref = phi.PP[0];

			for ( uint64_t cell = 0; cell < mesh->local_mesh_size + nhalos; cell++ )
			{
				//Partial update of the velocity and pressure field.
				T Ar = (A_phi.U[cell] != 0.0) ? 1.0 / A_phi.U[cell] : 0.0;
				T fact = cell_volumes[cell] * Ar;
				
				phi.P[cell] += 0.2*(phi.PP[cell] - Pref);

				//printf("pre u is %.22f\n",phi.U[cell]);		
				//printf("1 is %.22f\n",phi_grad.PP[cell].x * fact);
				//printf("2 is %.22f\n",phi_grad.PP[cell].y * fact);
				//printf("3 is %.22f\n",phi_grad.PP[cell].z * fact);	
				phi.U[cell] -= phi_grad.PP[cell].x * fact;
				phi.V[cell] -= phi_grad.PP[cell].y * fact; 
				phi.W[cell] -= phi_grad.PP[cell].z * fact;
			}

			for(uint64_t i = 0; i < mesh->local_mesh_size + nhalos + mesh->boundary_cells_size; i++){
                //printf("cell %lu has pressure %.15f and velocity (%.15f,%.15f,%.15f)\n", i, phi.P[i], phi.U[i], phi.V[i], phi.W[i]);
            }
			
			//Reset Su for the next partial solve.
			#pragma ivdep 
			for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
			{
				S_phi.P[i] = 0.0; 
			}
			phi.P[0] = 0.0; //in dolfyn we set P(0) to 0.0 but why????????

			update_mass_flux(); //Compute the correction for the mass fluxes

			/*printf("\nS_phi ");
			#pragma ivdep
			for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
			{
				printf("%.15f ",S_phi.P[i]);
			}
			printf("\n");*/

			//Reset PP for next partial solve
			for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos + mesh->boundary_cells_size; i++ )
			{
				phi.PP[i] = 0.0;
			}
			if(Loop_num >= 4 or Pressure_correction_max <= 0.25*Pressure_correction_ref) Loop_continue = false;
		}

		Update_P_at_boundaries(phi.P); //update boundaries for full Pressure field.

		get_phi_gradient(phi.P, phi_grad.P, true); //Update gradient for Pressure field.

		exchange_phi_halos(); //Exchange so phi_grad.P is correct at halos

		update_P(phi.P, phi_grad.P); //Final update of pressure field.

		/*printf("Pressure final is ");
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos + mesh->boundary_cells_size; i++ )
        {
            printf("%.15f ", phi.P[i]);
        }
        printf("\n");*/	
    }

	template<typename T> void FlowSolver<T>::FluxScalar(int type, T *phi_component, vec<T> *phi_grad_component)
	{
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
                //printf("identify (%.15f,%.15f,%.15f) and (%.15f,%.15f,%.15f)\n",mesh->cell_centers[block_cell0].x,mesh->cell_centers[block_cell0].y,mesh->cell_centers[block_cell0].z,mesh->cell_centers[block_cell1].x,mesh->cell_centers[block_cell1].y,mesh->cell_centers[block_cell1].z);
                uint64_t phi_index0 = ( block_cell0 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell0] : block_cell0;
				uint64_t phi_index1 = ( block_cell1 >= mesh->local_mesh_size ) ? boundary_map[mesh->faces[face].cell1] : block_cell1;
                
				const T lambda0 = face_lambdas[face];
                const T lambda1 = 1.0 - lambda0;

				T Visac    = effective_viscosity * lambda0 + effective_viscosity * lambda1;
				
				Visac -= effective_viscosity; //This will always be 0 right?
				
				if(type == TEMP)
				{
					//printf("Viac is = (%4.18f + %4.18f / 0.9) / 0.6905\n", effective_viscosity, Visac);
					Visac = (effective_viscosity + Visac / 0.9) / 0.6905;
				}
				else if(type  == TERBTE)
				{
					//printf("Visac is = %4.18f + %4.18f\n",effective_viscosity,Visac);
					Visac = effective_viscosity + Visac;
				}
				else if(type == TERBED)
				{
					//printf("Visac is %4.18f + %4.18f / 1.219\n",effective_viscosity,Visac);
					Visac = effective_viscosity + Visac / 1.219;
				}
				else
				{
					Visac = (effective_viscosity + Visac / 0.9) / 0.9;
				}
				//printf("visac is %4.18f\n",Visac);

				vec<T> dPhiXac = phi_grad_component[shmem_cell1] * lambda0 + phi_grad_component[shmem_cell0] * lambda1;

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

                const T fci = min( face_mass_fluxes[face], 0.0 ) * phi.U[phi_index0] + max( face_mass_fluxes[face], 0.0 ) * phi.U[phi_index1];

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

                //printf("face feilds -+ %4.18f +- %4.18f\n", VisFace, face_mass_fluxes[face]);
				const T blend = GammaBlend * ( fce - fci );

				// ! assemble the two source terms

                S_phi.U[phi_index0] = S_phi.U[phi_index0] - blend + fde1 - fdi;

                S_phi.U[phi_index1] = S_phi.U[phi_index1] + blend - fde1 + fdi;

				//printf("Su is +- %4.18f -+ %4.18f +- %4.18f \n",blend,fde1,fdi); 
                const T small_epsilon = 1.e-20;
                const T peclet = face_mass_fluxes[face] / face_areas[face] * magnitude(Xpn) / (Visac+small_epsilon);

                pe0 = min( pe0 , peclet );
                pe1 = max( pe1 , peclet );

            }
            else // BOUNDARY
            {

				//TODO: need to continue from here 

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
					else
					{
						printf("Error: unkown type in flux scalar\n");
						exit(0);
					}		

                    T Visac = inlet_effective_viscosity;
					
					//printf("before visac is %4.18f\n",Visac);
					Visac -= effective_viscosity; //This will always be 0 right?
					//printf("after visac is %4.18f\n",Visac);

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

					//printf("visac final is %4.18f\n",Visac);

                    const vec<T> Xpn = face_centers[face] - mesh->cell_centers[shmem_cell0];
					const T VisFace  = Visac * face_rlencos[face];

                    const T fde = Visac * dot_product( dPhidXac , face_normals[face]);

					//implicit part
					const T fdi = VisFace * dot_product( dPhidXac, Xpn);
					//printf("inlet\n");
                    const T f = -VisFace + min( face_mass_fluxes[face], 0.0 );
					//printf("Aphi adding %4.18f\n",f);
					//printf("f made up of %4.18f and %4.18f\n", VisFace, face_mass_fluxes[face]);
                    A_phi.U[block_cell0] = A_phi.U[block_cell0] - f;
					//printf("made up of f above %4.18f %4.18f %4.18f\n",PhiFace,fde,fdi);
					//printf("Sphi adding %4.18f\n",f * PhiFace + fde - fdi);
                    S_phi.U[block_cell0] = S_phi.U[block_cell0] - f * PhiFace + fde - fdi;
                    phi_component[mesh->local_mesh_size + nhalos + boundary_cell] = PhiFace;;
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

					//printf("outlet\n");
                    S_phi.U[block_cell0] = S_phi.U[block_cell0] + fde - fdi;
					//printf("made of %4.18f and %4.18f\n",fde,fdi);
					//printf("adding Sphi %4.18f\n",fde - fdi);
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
		if(type == TERBTE)
		{
			for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
			{
				const vec<T> dUdXp  =   phi_grad.U[i];
                const vec<T> dVdXp  =   phi_grad.V[i];
                const vec<T> dWdXp  =   phi_grad.W[i];

				const T s1 = (dUdXp.x+dUdXp.x)*dUdXp.x + (dUdXp.y+dVdXp.x)*dUdXp.y + (dUdXp.z+dWdXp.x)*dUdXp.z;
				const T s2 = (dVdXp.x+dUdXp.y)*dVdXp.x + (dVdXp.y+dVdXp.y)*dVdXp.y + (dVdXp.z+dWdXp.y)*dVdXp.z;
				const T s3 = (dWdXp.x+dUdXp.z)*dWdXp.x + (dWdXp.y+dVdXp.z)*dWdXp.y + (dWdXp.z+dWdXp.z)*dWdXp.z;

				T VisT = effective_viscosity - effective_viscosity; //TODO: if we compute viscocity the first part of this is replaced with vis[i]
				
				T Pk = VisT * (s1 + s2 + s3);

				phi.TP[i] = Pk;

				T Dis = cell_densities[i] * phi.ED[i];

				//printf("Su adding %4.18f\n",phi.TP[i] * cell_volumes[i]);
				S_phi.U[i] = S_phi.U[i] + phi.TP[i] * cell_volumes[i];
				//printf("Au adding %4.18f\n",Dis / (phi.TE[i] + 0.000000000000000001) * cell_volumes[i]);
				A_phi.U[i] = A_phi.U[i] + Dis / (phi.TE[i] + 0.000000000000000001) * cell_volumes[i]; 	
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
					//printf("wall\n");
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
					//printf("adding is = %4.18f * %4.18f\n",phi.TP[block_cell0],cell_volumes[block_cell0]);
					//printf("Su adding %4.18f\n",phi.TP[block_cell0] * cell_volumes[block_cell0]);
					S_phi.U[block_cell0] = S_phi.U[block_cell0] + phi.TP[block_cell0] * cell_volumes[block_cell0];

					//dissipation term
					T DisP = Cmu75*sqrt(phi.TE[block_cell0])*rkapdn;
					//printf("test2 %4.18f\n",sqrt(0.0001));
					//printf("test %4.18f\n",sqrt(phi.TE[block_cell0]));
					//printf("dis term %4.18f = %4.18f * sqrt(%4.18f)*%4.18f\n",DisP,Cmu75,phi.TE[block_cell0],rkapdn);
					//printf("Au adding %4.18f\n",cell_densities[block_cell0] * DisP * cell_volumes[block_cell0]);
					A_phi.U[block_cell0] = A_phi.U[block_cell0] + cell_densities[block_cell0] * DisP * cell_volumes[block_cell0];
				}
			}
		}
		else if(type == TERBED)
		{
			for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
			{
				T fact = phi.ED[i]/(phi.TE[i]+0.000000000000000001) * cell_volumes[i];
				S_phi.U[i] = S_phi.U[i] + 1.44 * fact * phi.TP[i];
				A_phi.U[i] = A_phi.U[i] + 1.92 * fact * cell_densities[i];
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
					//printf("made up of %4.18f and %4.18f and %4.18f and %4.18f\n", Cmu75, pow(turb,1.5), distance, 0.419);
					const T Dis = Cmu75 * pow(turb,1.5) / ( distance * 0.419 );
					
					for ( uint64_t j = 0; j < 6; j++ )
					{
						uint64_t neigh_face = mesh->cell_faces[(block_cell0 * mesh->faces_per_cell) + j];
						uint64_t neighbour = mesh->cell_neighbours[(block_cell0 - mesh->shmem_cell_disp) * mesh->faces_per_cell + j];
						if(neighbour < mesh->mesh_size)
						{
							//internal node
							if(mesh->faces[neigh_face].cell0 == block_cell0)
							{ 
								face_fields[neigh_face].cell1 = 0.0;
							}
							else if(mesh->faces[neigh_face].cell1 == block_cell0)
							{
								face_fields[neigh_face].cell0 = 0.0;
							}
						}
					}
					//printf("Dis is %4.18f = %4.18f * %4.18f / (%4.18f * 0.419)\n",Dis,Cmu75,pow(turb,1.5),distance);
					phi.ED[block_cell0] = Dis;
					S_phi.U[block_cell0] = Dis;
					A_phi.U[block_cell0] = 1;
					phi.ED[mesh->local_mesh_size + nhalos + boundary_cell] = phi.ED[block_cell0];	
				}
			}
		}
	}

	template<typename T> void FlowSolver<T>::Scalar_solve(int type, T *phi_component, vec<T> *phi_grad_component, T *old_phi)
	{
		/*Solve for a general scalar used for most transport equations
		  Follows the general procedure:
		  1. Calculate gradient
		  2. Collect face fluxes
		  3. set-up and solve matrix*/
		//get_phi_gradient(phi_component, phi_grad_component, false);
		//^ computed in get phi gradients() now
		/*for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
		{
			printf("grad is (%4.18f,%4.18f,%4.18f)\n",phi_grad_component[i].x,phi_grad_component[i].y,phi_grad_component[i].z);
		}*/
			
		//reuse Au and Su to reduce storage requirements
		#pragma ivdep
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            A_phi.U[i] = 0.0;
            S_phi.U[i] = 0.0;
        }

		//for ( uint64_t face = 0; face < mesh->faces_size; face++ )
		//{
		//	printf("mass flux before is %4.18f\n",face_mass_fluxes[face]);
		//}

		//collect face fluxes
		FluxScalar(type, phi_component, phi_grad_component);

		/*printf("Au is:");
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            printf(" %4.18f ", A_phi.U[i]);
		}
		printf("\n");
		printf("Su is:");
		for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
		{
            printf(" %4.18f ", S_phi.U[i]);
        }	
		printf("\n");*/


		//Unsteady term 
		double rdelta = 1.0 / delta;
        #pragma ivdep
        for ( uint64_t i = 0 ; i < mesh->local_mesh_size + nhalos; i++ )
        {
            double f = cell_densities[i] * cell_volumes[i] * rdelta;
            S_phi.U[i] += f * phi_component[i];
            A_phi.U[i] += f;
        }

		/*printf("after this bit\n");
		printf("Au is:");
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            printf(" %4.18f ", A_phi.U[i]);
        }
        printf("\n");
        printf("Su is:");
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            printf(" %4.18f ", S_phi.U[i]);
        }
        printf("\n");*/
		
		if(type == TERBTE or type == TERBED)
		{
			solveTurbulenceModels(type);
		}

		/*printf("after next bit\n");
        printf("Au is:");
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            printf(" %4.18f ", A_phi.U[i]);
        }
        printf("\n");
        printf("Su is:");
        for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        {
            printf(" %4.18f ", S_phi.U[i]);
        }
        printf("\n");*/	

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
	
		setup_sparse_matrix (underrelax, A_phi.U, phi_component, S_phi.U);

        MPI_Barrier(mpi_config->particle_flow_world);
        solve_sparse_matrix (phi_component, S_phi.U);

		//printf("phi is:");
        //for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        //{
        //    printf(" %4.18f ", phi_component[i]);
        //}
        //printf("\n");

		if(type == TERBTE or type == TERBED)
		{
			//Make sure nonnegative
			for ( uint64_t i = 0 ; i < mesh->local_mesh_size + nhalos; i++ )
			{
				int count = 0;
				T phisum = 0.0;
				if(phi_component[i] < 0.0)
				{
					//printf("here with phi %4.18f\n",phi_component[i]);
					for( uint64_t j = 0; j < 6; j++)
					{
						uint64_t neighbour = mesh->cell_neighbours[(i - mesh->shmem_cell_disp) * mesh->faces_per_cell + j];
						if(neighbour < mesh->mesh_size) 
						{
							//if internal
							const uint64_t block_neighbour = neighbour - mesh->local_cells_disp;
							const uint64_t phi_neighbour = ( block_neighbour >= mesh->local_mesh_size ) ? boundary_map[neighbour] : block_neighbour;
							count++;
							phisum += phi_component[phi_neighbour];	
						}
					}
					phisum = (phisum/count);
					phi_component[i] = max(phisum, 0.000000000001);
				}
			}
		}
		//printf("phi is:");
        //for ( uint64_t i = 0; i < mesh->local_mesh_size + nhalos; i++ )
        //{
            //printf(" %4.18f ", phi_component[i]);
        //}
        //printf("\n");	
	}

        template<class T>
    void FlowSolver<T>::print_logger_stats(uint64_t timesteps, double runtime)
    {
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
        if (FLOW_SOLVER_DEBUG)    printf("Start flow timestep\n");
        if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Start flow timestep.\n", mpi_config->rank);

        int comms_timestep = 1;
        static double halo_time = 0.0, grad_time = 0.0;
        grad_time -= MPI_Wtime();
        // Note: After pointer swap, last iterations phi is now in phi.
		if(mpi_config->particle_flow_rank == 0){
			printf("timestep %lu\n",timestep_count);
		}
	
		get_phi_gradients();
		grad_time += MPI_Wtime();	
		
		//limit_phi_gradients();
		
		if(timestep_count == 0)
		{
			set_up_field();
		}

        halo_time -= MPI_Wtime();
        exchange_phi_halos();
        halo_time += MPI_Wtime();

        if ((timestep_count % comms_timestep) == 0)  
            //update_flow_field();

        if ((timestep_count % 100) == 0)
        {
            double arr_usage  = ((double)get_array_memory_usage()) / 1.e9;
            double stl_usage  = ((double)get_stl_memory_usage())   / 1.e9 ;
            double mesh_usage = ((double)mesh->get_memory_usage()) / 1.e9 ;
            double arr_usage_total, stl_usage_total, mesh_usage_total;

            MPI_Reduce(&arr_usage,  &arr_usage_total,  1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&stl_usage,  &stl_usage_total,  1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);
            MPI_Reduce(&mesh_usage, &mesh_usage_total, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_config->particle_flow_world);

            if ( mpi_config->particle_flow_rank == 0 )
            {
                printf("                Flow     mem (TOTAL %8.3f GB) (AVG %8.3f GB) \n", (arr_usage_total + stl_usage_total + mesh_usage_total), (arr_usage_total + stl_usage_total + mesh_usage_total) / mpi_config->particle_flow_world_size);

            }
        }
		/*if(mpi_config->particle_flow_rank == 0)
		{
			printf("Result is:\n");
		}
		MPI_Barrier(mpi_config->particle_flow_world);
        for(int i = 0; i < mpi_config->particle_flow_world_size; i++)
        {
            if(i == mpi_config->particle_flow_rank)
            {
                for(uint64_t block_cell = 0; block_cell < mesh->local_mesh_size + nhalos; block_cell++ )
                {
                    const uint64_t cell = block_cell + mesh->local_cells_disp;
                    printf("locate (%4.18f,%4.18f,%4.18f)\n", mesh->cell_centers[block_cell].x, mesh->cell_centers[block_cell].y, mesh->cell_centers[block_cell].z);
                    printf("grads are pressure (%4.18f,%4.18f,%4.18f) \nu-vel (%4.18f,%4.18f,%4.18f) \nV-vel (%4.18f,%4.18f,%4.18f) \nw-vel (%4.18f,%4.18f,%4.18f)\n\n", phi_grad.P[block_cell].x, phi_grad.P[block_cell].y, phi_grad.P[block_cell].z, phi_grad.U[block_cell].x, phi_grad.U[block_cell].y, phi_grad.U[block_cell].z, phi_grad.V[block_cell].x, phi_grad.V[block_cell].y, phi_grad.V[block_cell].z, phi_grad.W[block_cell].x, phi_grad.W[block_cell].y, phi_grad.W[block_cell].z);
                }
            }
            MPI_Barrier(mpi_config->particle_flow_world);
        }*/

		//for(int i = 0; i < mpi_config->particle_flow_world_size; i++)
        //{
        //    if(i == mpi_config->particle_flow_rank)
        //    {	
		calculate_UVW();
		//	}
		//	MPI_Barrier(mpi_config->particle_flow_world);
		//}

		//exchange_phi_halos(); //exchange new UVW values.

//        calculate_pressure();
		
		//Turbulence solve
//		Scalar_solve(TERBTE, phi.TE, phi_grad.TE, phi.TE);
//		Scalar_solve(TERBED, phi.ED, phi_grad.ED, phi.ED);
		
		//temperature solve
//		Scalar_solve(TEMP, phi.TEM, phi_grad.TEM, phi.TEM);

		//fuel mixture fraction solve
//		Scalar_solve(FUEL, phi.FUL, phi_grad.FUL, phi.FUL);

		//rection progression solve
//		Scalar_solve(PROG, phi.PRO, phi_grad.PRO, phi.PRO);

		printf("Result is:\n");
		for(int i = 0; i < mpi_config->particle_flow_world_size; i++)
		{
			if(i == mpi_config->particle_flow_rank)
			{
				for(uint64_t block_cell = 0; block_cell < mesh->local_mesh_size + nhalos; block_cell++ )
				{
					const uint64_t cell = block_cell + mesh->local_cells_disp;
					printf("locate (%4.18f,%4.18f,%4.18f)\n", mesh->cell_centers[cell].x,mesh->cell_centers[cell].y,mesh->cell_centers[cell].z);
					printf("Variables are pressure %4.18f \nvel (%4.18f,%4.18f,%4.18f) \nTerb (%4.18f,%4.18f) \ntemerature %4.18f fuel mizx %4.18f \nand progression %.6f\n\n", phi.P[block_cell], phi.U[block_cell], phi.V[block_cell], phi.W[block_cell], phi.TE[block_cell], phi.ED[block_cell], phi.TEM[block_cell], phi.FUL[block_cell], phi.PRO[block_cell]);
				}
			}
			MPI_Barrier(mpi_config->particle_flow_world);
		}

        if ((timestep_count == 1499) && mpi_config->particle_flow_rank == 0)
        {
            printf("Halo time: %7.2fs\n", halo_time );
            printf("Grad time: %7.2fs\n", grad_time );
        }

        if ((timestep_count % 20) == 0)
        {
            VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh, mpi_config);
            vtk_writer->write_flow_velocities("out/minicombust", timestep_count, &phi);
        }
		MPI_Barrier(mpi_config->world);
        
		if ( FLOW_SOLVER_DEBUG )  printf("\tFlow Rank %d: Stop flow timestep.\n", mpi_config->rank);
        timestep_count++;
    }
}
