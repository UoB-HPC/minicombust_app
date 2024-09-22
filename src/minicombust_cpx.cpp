#include <stdio.h>
#include <ctime>
#include <inttypes.h>

#include "examples/mesh_examples.hpp"
#include "examples/particle_examples.hpp"
#include "geometry/Mesh.hpp"
#include "utils/utils.hpp"

#include <chrono>
#include <ctime>    


#include "particles/ParticleSolver.inl"
#ifdef have_gpu
	#include "flow/gpu/FlowSolver.inl"
    #include <cuda.h>
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

    // #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    // inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    // {
    // if (code != cudaSuccess) 
    // {
    //     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    //     if (abort) exit(code);
    // }
    // }

#else
	#include "flow/FlowSolver.inl"
#endif
#include "cpx/cpx_utils.inl"

using namespace minicombust::flow;
using namespace minicombust::particles;
using namespace minicombust::visit;


int main_minicombust(int argc, char ** argv, MPI_Fint custom, int instance_number, struct unit units[], struct locators relative_positions[])
{
    char filename[2];
	char default_name[31] = "COMBUST_output_instance_";
	sprintf(filename, "%d", instance_number);
	strcat(default_name, filename);
	FILE *fp = fopen(default_name, "w");

    //Input variables
    int flow_ranks;
    uint64_t particles_per_timestep;
    uint64_t modifier;
    int64_t output_iteration;
    uint64_t ntimesteps = coupler_cycles * combust_conversion_factor;

    read_inputs(&flow_ranks, &particles_per_timestep, &modifier, &output_iteration);

    MPI_Comm custom_comm = MPI_Comm_f2c(custom);

    MPI_Config mpi_config;

    mpi_config.world = custom_comm;

    // Create overall world
    MPI_Comm_rank(mpi_config.world,  &mpi_config.rank);
    MPI_Comm_size(mpi_config.world,  &mpi_config.world_size);

    int world_rank;

    MPI_Comm_rank(MPI_COMM_WORLD,  &world_rank);
    
    int particle_ranks = mpi_config.world_size - flow_ranks;

    mpi_config.solver_type = (mpi_config.rank >= flow_ranks); // 1 for particle, 0 for flow
    MPI_Comm_split(mpi_config.world, mpi_config.solver_type, mpi_config.rank, &mpi_config.particle_flow_world);
    MPI_Comm_rank(mpi_config.particle_flow_world,  &mpi_config.particle_flow_rank);
    MPI_Comm_size(mpi_config.particle_flow_world,  &mpi_config.particle_flow_world_size);
    
    // Create Flow/Particle Datatypes
    MPI_Type_contiguous(sizeof(flow_aos<double>)/sizeof(double),     MPI_DOUBLE, &mpi_config.MPI_FLOW_STRUCTURE);
    MPI_Type_contiguous(sizeof(vec<double>)/sizeof(double),          MPI_DOUBLE, &mpi_config.MPI_VEC_STRUCTURE);
    MPI_Type_contiguous(sizeof(particle_aos<double>)/sizeof(double), MPI_DOUBLE, &mpi_config.MPI_PARTICLE_STRUCTURE);
    MPI_Type_commit(&mpi_config.MPI_FLOW_STRUCTURE);
    MPI_Type_commit(&mpi_config.MPI_PARTICLE_STRUCTURE);

    MPI_Op_create(&sum_particle_aos<double>, 1, &mpi_config.MPI_PARTICLE_OPERATION);

    // Run Configuration
    const double   delta                        = 1.0e-5;
   
    // Mesh Configuration
    vec<double>    box_dim                 = {0.05*modifier*2, 0.05*modifier, 0.05*modifier};
    vec<uint64_t>  elements_per_dim        = {modifier*2,   modifier*1,  modifier*1};

    if (mpi_config.rank == 0)  
    {
        fprintf(fp, "Starting miniCOMBUST..\n");
        fprintf(fp, "MPI Configuration:\n\tFlow Ranks: %d\n\tParticle Ranks: %d\n", flow_ranks, particle_ranks);
    }

    // Performance
    double mesh_time = 0., setup_time = 0., program_time = 0., output_time = 0.;

    // Perform setup and benchmark cases
    MPI_Barrier(mpi_config.world); setup_time  -= MPI_Wtime(); mesh_time  -= MPI_Wtime(); 
    Mesh<double> *mesh                          = load_mesh(&mpi_config, box_dim, elements_per_dim, flow_ranks, fp);
    MPI_Barrier(mpi_config.world); mesh_time   += MPI_Wtime();

	if (mpi_config.rank == 0)  fprintf(fp, "Mesh built in %6.2fs!\n\n", mesh_time);

    mpi_config.one_flow_rank             = (int *)     malloc(flow_ranks * sizeof(int));
    mpi_config.every_one_flow_rank       = (int *)     malloc(flow_ranks * sizeof(int));
    mpi_config.one_flow_world_size       = (int *)     malloc(flow_ranks * sizeof(int));
    mpi_config.every_one_flow_world_size = (int *)     malloc(flow_ranks * sizeof(int));
    mpi_config.one_flow_world            = (MPI_Comm *)malloc(flow_ranks * sizeof(MPI_Comm));
    mpi_config.every_one_flow_world      = (MPI_Comm *)malloc(flow_ranks * sizeof(MPI_Comm));
    mpi_config.alias_rank                = (int *)     malloc(flow_ranks * sizeof(int));

    //coupling data
    double *p_variables_data;
    double *p_variables_recv;
    if(mpi_config.rank == 0)
    {
        send_num_data(units, relative_positions, mesh->mesh_size, &p_variables_data, &p_variables_recv);
    }

    MPI_Barrier(mpi_config.world);

    //Setup solvers
    ParticleSolver<double> *particle_solver = nullptr;
    FlowSolver<double>     *flow_solver     = nullptr;
    ParticleDistribution<double> *particle_dist;
    if (mpi_config.solver_type == PARTICLE)
    {
        uint64_t       local_particles_per_timestep   = particles_per_timestep / mpi_config.particle_flow_world_size;
        int            remainder_particles            = particles_per_timestep % mpi_config.particle_flow_world_size;

        const uint64_t reserve_particles_size         = 2 * (local_particles_per_timestep + 1) * ntimesteps;

        particle_dist = load_injector_particle_distribution(particles_per_timestep, local_particles_per_timestep, remainder_particles, &mpi_config, box_dim, mesh);
        //ParticleDistribution<double> *particle_dist = load_particle_distribution(particles_per_timestep, local_particles_per_timestep, remainder_particles, &mpi_config, mesh);
        particle_solver = new ParticleSolver<double>(&mpi_config, ntimesteps, delta, particle_dist, mesh, reserve_particles_size, fp);
        if(mpi_config.particle_flow_rank == 0)
        {
            MPI_Send(&mpi_config.particle_flow_world_size, 1, MPI_INT, 0, 10, mpi_config.world);
            MPI_Send(&particle_solver->total_node_index_array_size, 1, MPI_UINT64_T, 0, 11, mpi_config.world);
            MPI_Send(&particle_solver->total_node_flow_array_size, 1, MPI_UINT64_T, 0, 12, mpi_config.world);
            MPI_Send(&particle_solver->total_cell_particle_index_array_size, 1, MPI_UINT64_T, 0, 13, mpi_config.world);
            MPI_Send(&particle_solver->total_cell_particle_array_size, 1, MPI_UINT64_T, 0, 14, mpi_config.world);
            MPI_Send(&particle_solver->total_neighbours_sets_size, 1, MPI_UINT64_T, 0, 15, mpi_config.world);
            MPI_Send(&particle_solver->total_cell_particle_field_map_size, 1, MPI_UINT64_T, 0, 16, mpi_config.world);
            MPI_Send(&particle_solver->total_particles_size, 1, MPI_UINT64_T, 0, 17, mpi_config.world);
            MPI_Send(&particle_solver->total_node_to_field_address_map_size, 1, MPI_UINT64_T, 0, 18, mpi_config.world);
            MPI_Send(&particle_solver->total_memory_usage, 1, MPI_UINT64_T, 0, 19, mpi_config.world);
        }
    }
    else
    {
		#ifdef have_gpu
			AMGX_SAFE_CALL(AMGX_initialize());
		#else
			PETSC_COMM_WORLD = mpi_config.particle_flow_world;
			PetscInitialize(&argc, &argv, nullptr, nullptr);
		#endif
        flow_solver     = new FlowSolver<double>(&mpi_config, mesh, delta, fp);

        if(mpi_config.particle_flow_rank == 0)
        {
            int part_world_size = 0;
            // Array sizes
            uint64_t total_node_index_array_size = 0;
            uint64_t total_node_flow_array_size = 0;
            uint64_t total_cell_particle_index_array_size = 0;
            uint64_t total_cell_particle_array_size = 0;

            // STL sizes
            uint64_t total_neighbours_sets_size = 0;
            uint64_t total_cell_particle_field_map_size = 0;

            uint64_t total_particles_size = 0;
            uint64_t total_node_to_field_address_map_size = 0;

            uint64_t total_memory_usage = 0;

            MPI_Recv(&part_world_size, 1, MPI_INT, mpi_config.particle_flow_world_size, 10, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_node_index_array_size, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 11, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_node_flow_array_size, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 12, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_cell_particle_index_array_size, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 13, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_cell_particle_array_size, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 14, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_neighbours_sets_size, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 15, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_cell_particle_field_map_size, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 16, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_particles_size, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 17, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_node_to_field_address_map_size, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 18, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&total_memory_usage, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 19, mpi_config.world, MPI_STATUS_IGNORE);

            fprintf(fp, "Particle solver storage requirements (%d processes) : \n", part_world_size);
            fprintf(fp, "\ttotal_node_index_array_size                           (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_index_array_size           / 1000000.0, (float) total_node_index_array_size          / (1000000.0 * part_world_size));
            fprintf(fp, "\ttotal_node_flow_array_size                            (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_node_flow_array_size            / 1000000.0, (float) total_node_flow_array_size           / (1000000.0 * part_world_size));
            fprintf(fp, "\ttotal_cell_particle_index_array_size                  (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_index_array_size  / 1000000.0, (float) total_cell_particle_index_array_size / (1000000.0 * part_world_size));
            fprintf(fp, "\ttotal_cell_particle_array_size                        (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_array_size        / 1000000.0, (float) total_cell_particle_array_size       / (1000000.0 * part_world_size));
            fprintf(fp, "\ttotal_neighbours_sets_size            (STL set)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_neighbours_sets_size            / 1000000.0, (float) total_neighbours_sets_size           / (1000000.0 * part_world_size));
            fprintf(fp, "\ttotal_cell_particle_field_map_size    (STL map)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_cell_particle_field_map_size    / 1000000.0, (float) total_cell_particle_field_map_size   / (1000000.0 * part_world_size));
            fprintf(fp, "\ttotal_particles_size                  (STL vector)    (TOTAL %8.2f MB) (AVG %8.2f MB) \n"    , (float) total_particles_size                  / 1000000.0, (float) total_particles_size                 / (1000000.0 * part_world_size));
            fprintf(fp, "\ttotal_node_to_field_address_map_size  (STL map)       (TOTAL %8.2f MB) (AVG %8.2f MB) \n\n"  , (float) total_node_to_field_address_map_size  / 1000000.0, (float) total_node_to_field_address_map_size / (1000000.0 * part_world_size));

            fprintf(fp, "\tParticle solver size                                  (TOTAL %12.2f MB) (AVG %.2f MB) \n\n"  , (float)total_memory_usage                      /1000000.0,  (float)total_memory_usage / (1000000.0 * part_world_size));
        }
    }

	if (mpi_config.rank == 0)   cout << endl;
    setup_time += MPI_Wtime(); MPI_Barrier(mpi_config.world); 

    if (mpi_config.rank == 0)  fprintf(fp, "Mesh built in %6.2fs!\n\n", mesh_time);


    // Output mesh 
    MPI_Barrier(mpi_config.world); output_time -= MPI_Wtime(); 
    if ( output_iteration != -1 )
    {
        VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh, &mpi_config);
        vtk_writer->write_mesh("out/minicombust");
    }
    output_time += MPI_Wtime(); MPI_Barrier(mpi_config.world); 

    //Remove the effects of setup on the timings
    MPI_Barrier(MPI_COMM_WORLD);

    // Main loop
    if (mpi_config.rank == 0)  fprintf(fp, "Starting simulation..\n");
    MPI_Barrier(mpi_config.world);
    program_time -= MPI_Wtime();

    // auto start = std::chrono::system_clock::now();
 
    // std::time_t start_time = std::chrono::system_clock::to_time_t(start);
 
    // if (mpi_config.rank == 0) std::cout << "STARTING CLOCK 1410 " << std::ctime(&start_time) << std::endl;

	for(uint64_t t = 0; t < ntimesteps; t++)
    {
        if (mpi_config.solver_type == PARTICLE)
        {
            if ((output_iteration != -1) and (((int64_t)((t+1) % output_iteration) == 0) or (t == 0)))
            {
                output_time -= MPI_Wtime();
                particle_solver->output_data(t+1);
                output_time += MPI_Wtime();
            }

            particle_solver->timestep();
        }
		else
		{
    		if ((output_iteration != -1) and (((int64_t)((t+1) % output_iteration) == 0) or (t == 0)))
            {
                output_time -= MPI_Wtime();
				flow_solver->output_data(t+1);
                output_time += MPI_Wtime();
            }

            flow_solver->timestep();

            if(mpi_config.particle_flow_rank == 0)
            {
                if(((t+1) % combust_conversion_factor == 0) || ((hide_search == true) && ((t+1) % combust_conversion_factor) == (combust_conversion_factor - 1)))               {
                    send_recv_data(units, relative_positions, mesh->mesh_size, t, ntimesteps, p_variables_data, p_variables_recv, fp);
                }
            }
            MPI_Barrier(mpi_config.particle_flow_world);
		}
    }

    program_time += MPI_Wtime();
    MPI_Barrier(mpi_config.world);
    // auto end = std::chrono::system_clock::now();

    // std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    // if (mpi_config.rank == 0) std::cout << "ENDING CLOCK 1410 " << std::ctime(&end_time) << std::endl;

    if (mpi_config.rank == 0) fprintf(fp, "Done!\n\n");

    //Print logger stats and write performance counters
    if (LOGGER) 
    {
        if (mpi_config.solver_type == PARTICLE)
        {
            particle_solver->print_logger_stats(ntimesteps, program_time);
            if(mpi_config.particle_flow_rank == 0)
            {
                MPI_Send(&mpi_config.particle_flow_world_size, 1, MPI_INT, 0, 10, mpi_config.world);
                MPI_Send(&particle_solver->logger.num_particles, 1, MPI_UINT64_T, 0, 11, mpi_config.world);
                MPI_Send(&particle_dist->even_particles_per_timestep, 1, MPI_UINT64_T, 0, 12, mpi_config.world);
                MPI_Send(&particle_solver->logger.emitted_particles, 1, MPI_UINT64_T, 0, 13, mpi_config.world);
                MPI_Send(&particle_solver->logger.avg_particles, 1, MPI_DOUBLE, 0, 14, mpi_config.world);
                MPI_Send(&particle_solver->logger.cell_checks, 1, MPI_UINT64_T, 0, 15, mpi_config.world);
                MPI_Send(&particle_solver->logger.position_adjustments, 1, MPI_UINT64_T, 0, 16, mpi_config.world);
                MPI_Send(&particle_solver->logger.lost_particles, 1, MPI_UINT64_T, 0, 17, mpi_config.world);
                MPI_Send(&particle_solver->logger.boundary_intersections, 1, MPI_UINT64_T, 0, 18, mpi_config.world);
                MPI_Send(&particle_solver->logger.decayed_particles, 1, MPI_UINT64_T, 0, 19, mpi_config.world);
                MPI_Send(&particle_solver->logger.burnt_particles, 1, MPI_UINT64_T, 0, 20, mpi_config.world);
                MPI_Send(&particle_solver->logger.breakups, 1, MPI_UINT64_T, 0, 21, mpi_config.world);
                MPI_Send(&particle_solver->logger.breakup_age, 1, MPI_DOUBLE, 0, 22, mpi_config.world);
                MPI_Send(&particle_solver->logger.sent_cells_per_block, 1, MPI_DOUBLE, 0, 23, mpi_config.world);
                MPI_Send(&particle_solver->logger.sent_cells, 1, MPI_DOUBLE, 0, 24, mpi_config.world);
                MPI_Send(&particle_solver->logger.nodes_recieved, 1, MPI_DOUBLE, 0, 25, mpi_config.world);
                MPI_Send(&particle_solver->logger.useful_nodes_proportion, 1, MPI_DOUBLE, 0, 26, mpi_config.world);
            }
        }
        else
        {
            flow_solver->print_logger_stats(ntimesteps, program_time);
            if(mpi_config.particle_flow_rank == 0)
            {
                int part_world_size = 0;
                Particle_Logger part_log;
                uint64_t even_particles_per_timestep = 0;

                MPI_Recv(&part_world_size, 1, MPI_INT, mpi_config.particle_flow_world_size, 10, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.num_particles, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 11, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&even_particles_per_timestep, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 12, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.emitted_particles, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 13, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.avg_particles, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 14, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.cell_checks, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 15, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.position_adjustments, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 16, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.lost_particles, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 17, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.boundary_intersections, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 18, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.decayed_particles, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 19, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.burnt_particles, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 20, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.breakups, 1, MPI_UINT64_T, mpi_config.particle_flow_world_size, 21, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.breakup_age, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 22, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.sent_cells_per_block, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 23, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.sent_cells, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 24, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.nodes_recieved, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 25, mpi_config.world, MPI_STATUS_IGNORE);
                MPI_Recv(&part_log.useful_nodes_proportion, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 26, mpi_config.world, MPI_STATUS_IGNORE);

                fprintf(fp, "Particle Solver Stats:                         \n");
                fprintf(fp, "\tParticles:                                   %e\n", ((double)part_log.num_particles));
                fprintf(fp, "\tParticles (per iter):                        %lu\n", even_particles_per_timestep*part_world_size);
                fprintf(fp, "\tEmitted Particles:                           %lu\n", part_log.emitted_particles);
                fprintf(fp, "\tAvg Particles (per iter):                    %e\n", part_log.avg_particles);
                fprintf(fp, "\n");
                fprintf(fp, "\tCell checks:                                 %e\n", ((double)part_log.cell_checks));
                fprintf(fp, "\tCell checks (per iter):                      %e\n", ((double)part_log.cell_checks) / ntimesteps);
                fprintf(fp, "\tCell checks (per particle, per iter):        %.2f\n", ((double)part_log.cell_checks) / (((double)part_log.num_particles)*ntimesteps));
                fprintf(fp, "\n");
                fprintf(fp, "\tEdge adjustments:                            %.2f\n", ((double)part_log.position_adjustments));
                fprintf(fp, "\tEdge adjustments (per iter):                 %.2f\n", ((double)part_log.position_adjustments) / ntimesteps);
                fprintf(fp, "\tEdge adjustments (per particle, per iter):   %.2f\n", ((double)part_log.position_adjustments) / (((double)part_log.num_particles)*ntimesteps));
                fprintf(fp, "\tLost Particles:                              %.2f\n", ((double)part_log.lost_particles      ));
                fprintf(fp, "\n");
                fprintf(fp, "\tBoundary Intersections:                      %.2f\n", ((double)part_log.boundary_intersections));
                fprintf(fp, "\tDecayed Particles:                           %e\n", ((double)part_log.decayed_particles));
                fprintf(fp, "\tDecayed Particles:                           %.2f%%\n", round(10000.*(((double)part_log.decayed_particles) / ((double)part_log.num_particles)))/100.);
                fprintf(fp, "\tBurnt Particles:                             %e\n", ((double)part_log.burnt_particles));
                fprintf(fp, "\tBreakups:                                    %.2f\n", ((double)part_log.breakups));
                fprintf(fp, "\tBreakup Age:                                 %.2f\n", ((double)part_log.breakup_age));
                fprintf(fp, "\n");
                fprintf(fp, "\tAvg Sent Cells       (avg per rank, block):  %.0f\n", round(part_log.sent_cells_per_block / ntimesteps));
                fprintf(fp, "\tTotal Sent Cells     (avg per rank):         %.0f\n", round(part_log.sent_cells / ntimesteps));
                fprintf(fp, "\tTotal Recieved Nodes (avg per rank):         %.0f\n", round(part_log.nodes_recieved / ntimesteps));
                fprintf(fp, "\tUseful Nodes         (avg per rank):         %.0f\n", round(part_log.useful_nodes_proportion / ntimesteps));
                fprintf(fp, "\tUseful Nodes (%%)     (avg per rank):         %.2f%%\n", round(10000.*((part_log.useful_nodes_proportion) / (part_log.nodes_recieved))) / 100.);
            }
        }
    }

    // Get program times
    double setup_time_avg = 0., program_time_avg = 0., output_time_avg = 0.;
    double setup_time_max = 0., program_time_max = 0., output_time_max = 0.;
    double setup_time_min = 0., program_time_min = 0., output_time_min = 0.;
    double comp_time_avg = 0., comp_time_max = 0., comp_time_min = 0.;
    MPI_Reduce(&setup_time,    &setup_time_avg,   1, MPI_DOUBLE, MPI_SUM, 0, mpi_config.world);
    MPI_Reduce(&setup_time,    &setup_time_max,   1, MPI_DOUBLE, MPI_MAX, 0, mpi_config.world);
    MPI_Reduce(&setup_time,    &setup_time_min,   1, MPI_DOUBLE, MPI_MIN, 0, mpi_config.world);
    MPI_Reduce(&program_time,  &program_time_avg, 1, MPI_DOUBLE, MPI_SUM, 0, mpi_config.world);
    MPI_Reduce(&program_time,  &program_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_config.world);
    MPI_Reduce(&program_time,  &program_time_min, 1, MPI_DOUBLE, MPI_MIN, 0, mpi_config.world);
    MPI_Reduce(&output_time,   &output_time_avg,  1, MPI_DOUBLE, MPI_SUM, 0, mpi_config.world);
    MPI_Reduce(&output_time,   &output_time_max,  1, MPI_DOUBLE, MPI_MAX, 0, mpi_config.world);
    MPI_Reduce(&output_time,   &output_time_min,  1, MPI_DOUBLE, MPI_MIN, 0, mpi_config.world);
	if(mpi_config.solver_type == PARTICLE)
	{
		MPI_Reduce(&(particle_solver->compute_time), &comp_time_avg, 1,
                MPI_DOUBLE, MPI_SUM, 0, mpi_config.particle_flow_world);
        MPI_Reduce(&(particle_solver->compute_time), &comp_time_max, 1,
                MPI_DOUBLE, MPI_MAX, 0, mpi_config.particle_flow_world);
        MPI_Reduce(&(particle_solver->compute_time), &comp_time_min, 1,
                MPI_DOUBLE, MPI_MIN, 0, mpi_config.particle_flow_world);
	}
	else
	{
		MPI_Reduce(&(flow_solver->compute_time), &comp_time_avg, 1, 
				MPI_DOUBLE, MPI_SUM, 0, mpi_config.particle_flow_world);
		MPI_Reduce(&(flow_solver->compute_time), &comp_time_max, 1,
				MPI_DOUBLE, MPI_MAX, 0, mpi_config.particle_flow_world);
		MPI_Reduce(&(flow_solver->compute_time), &comp_time_min, 1,
				MPI_DOUBLE, MPI_MIN, 0, mpi_config.particle_flow_world);
	}

	int precision = 5+log10(max({setup_time_max, program_time_max, 
								output_time_max, comp_time_max}));
    precision = 2;
    cout.precision(2);
    cout.setf(ios::fixed);

    if (mpi_config.rank == 0)
    {
        setup_time_avg    /= (double)mpi_config.world_size;
        program_time_avg  /= (double)mpi_config.world_size;
        output_time_avg   /= (double)mpi_config.world_size;

        // cout << "Setup Time:            " << setw(precision) << setup_time_avg   << "s  (min " << setw(precision) << setup_time_min   << "s) " << "(max " << setw(precision) << setup_time_max    << "s)\n";
        // cout << "Program Time:          " << setw(precision) << program_time_avg << "s  (min " << setw(precision) << program_time_min << "s) " << "(max " << setw(precision) << program_time_max  << "s)\n";
        // cout << "Output Time:           " << setw(precision) << output_time_avg  << "s  (min " << setw(precision) << output_time_min  << "s) " << "(max " << setw(precision) << output_time_max   << "s)\n";

        fprintf(fp, "Setup Time:            %.*fs  (min %.*fs) (max %.*fs)\n", precision, setup_time_avg, precision, setup_time_min, precision, setup_time_max);
        fprintf(fp, "Program Time:            %.*fs  (min %.*fs) (max %.*fs)\n", precision, program_time_avg, precision, program_time_min, precision, program_time_max);
        fprintf(fp, "Output Time:            %.*fs  (min %.*fs) (max %.*fs)\n", precision, output_time_avg, precision, output_time_min, precision, output_time_max);
    }
	MPI_Barrier(mpi_config.world);
	if(mpi_config.particle_flow_rank == 0)
	{
		if(mpi_config.solver_type == PARTICLE)
		{
			comp_time_avg /= (double)mpi_config.particle_flow_world_size;
            MPI_Send(&comp_time_avg, 1, MPI_DOUBLE, 0, 10, mpi_config.world);
            MPI_Send(&comp_time_min, 1, MPI_DOUBLE, 0, 11, mpi_config.world);
            MPI_Send(&comp_time_max, 1, MPI_DOUBLE, 0, 12, mpi_config.world);

            // cout << "Particle Compute Time: " << setw(precision) << comp_time_avg 
			// 		<< "s  (min " << setw(precision) << comp_time_min << "s) (max "
			// 		<< setw(precision) << comp_time_max << "s)\n";
		}
		else
		{
            double part_comp_time_avg = 0., part_comp_time_min = 0., part_comp_time_max = 0.;
            MPI_Recv(&part_comp_time_avg, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 10, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&part_comp_time_min, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 11, mpi_config.world, MPI_STATUS_IGNORE);
            MPI_Recv(&part_comp_time_max, 1, MPI_DOUBLE, mpi_config.particle_flow_world_size, 12, mpi_config.world, MPI_STATUS_IGNORE);
			comp_time_avg /= (double)mpi_config.particle_flow_world_size;

            // cout << "Flow Compute Time:     " << setw(precision) << comp_time_avg
			// 		<< "s  (min " << setw(precision) << comp_time_min << "s) (max "
			// 		<< setw(precision) << comp_time_max << "s)\n";
            fprintf(fp, "Flow Compute Time:     %.*fs  (min %.*fs) (max %.*fs)\n", precision, comp_time_avg, precision, comp_time_min, precision, comp_time_max);
            fprintf(fp, "Particle Compute Time: %.*fs  (min %.*fs) (max %.*fs)\n", precision, part_comp_time_avg, precision, part_comp_time_min, precision, part_comp_time_max);
		}
	}

    MPI_Barrier(mpi_config.world);
	
	if (mpi_config.solver_type != PARTICLE)
	{
		#ifdef have_gpu
			flow_solver->AMGX_free();
			AMGX_SAFE_CALL(AMGX_finalize());
		#else
			PetscFinalize();
		#endif
	}

    MPI_Win_free(&mpi_config.win_cell_centers);
    MPI_Win_free(&mpi_config.win_cells);
    MPI_Win_free(&mpi_config.win_cell_neighbours);
    MPI_Win_free(&mpi_config.win_points);
    MPI_Win_free(&mpi_config.win_cells_per_point);

    return 0;
}
