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


using namespace minicombust::flow;
using namespace minicombust::particles;
using namespace minicombust::visit;


int main (int argc, char ** argv)
{
	//TODO:reduce mega long lines
	//TODO:sort out vtk_output.
	//TODO:ideal flow tanks seem to cause a segfault

    //Extract runtime parameters with env var
    char *buf = getenv("MINICOMBUST_FRANKS");
    const int flow_ranks = atoi(buf); 

    buf = getenv("MINICOMBUST_PRANKS");
    const int particle_ranks = atoi(buf); 

    buf = getenv("MINICOMBUST_ITERS");
    const uint64_t ntimesteps = atoi(buf); 

    buf = getenv("MINICOMBUST_OUTPUT_ITER");
    const int64_t output_iteration = atoi(buf); 

    buf = getenv("MINICOMBUST_PARTICLES");
    const uint64_t particles_per_timestep = atoi(buf); 

    buf = getenv("MINICOMBUST_CELLS");
    const uint64_t modifier = atoi(buf); 

    #ifdef have_gpu
        //Extract mpi_rank with env var
        buf = getenv("MINICOMBUST_RANK_ID");
        const int rank_id = atoi(buf); 
        
        cudaFree(0);
        if (rank_id < flow_ranks) 
        {
            int cuda_dev = 0;
            gpuErrchk( cudaSetDevice(cuda_dev));
            // printf("Process %d selecting device %d of %d\n", rank_id, cuda_dev, flow_ranks);
        }
    #endif

    // MPI Initialisation 
    MPI_Init(&argc, &argv);
    MPI_Config mpi_config;
    mpi_config.world = MPI_COMM_WORLD;

    // Create overall world
    MPI_Comm_rank(mpi_config.world,  &mpi_config.rank);
    MPI_Comm_size(mpi_config.world,  &mpi_config.world_size);

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
    vec<double>    box_dim                 = {0.05*modifier*2, 0.05*modifier, 0.05*modifier};//{1, 0.5, 0.5};
    vec<uint64_t>  elements_per_dim        = {modifier*2,   modifier*1,  modifier*1};

    if (mpi_config.rank == 0)  
    {
        printf("Starting miniCOMBUST..\n");
        printf("MPI Configuration:\n\tFlow Ranks: %d\n\tParticle Ranks: %d\n", flow_ranks, particle_ranks);
    }

    // Performance
    double mesh_time = 0., setup_time = 0., program_time = 0., output_time = 0.;

    // Perform setup and benchmark cases
    MPI_Barrier(mpi_config.world); setup_time  -= MPI_Wtime(); mesh_time  -= MPI_Wtime(); 
    Mesh<double> *mesh                          = load_mesh(&mpi_config, box_dim, elements_per_dim, flow_ranks, stdout);
    MPI_Barrier(mpi_config.world); mesh_time   += MPI_Wtime();

	if (mpi_config.rank == 0)  printf("Mesh built in %6.2fs!\n\n", mesh_time);

    mpi_config.one_flow_rank             = (int *)     malloc(flow_ranks * sizeof(int));
    mpi_config.every_one_flow_rank       = (int *)     malloc(flow_ranks * sizeof(int));
    mpi_config.one_flow_world_size       = (int *)     malloc(flow_ranks * sizeof(int));
    mpi_config.every_one_flow_world_size = (int *)     malloc(flow_ranks * sizeof(int));
    mpi_config.one_flow_world            = (MPI_Comm *)malloc(flow_ranks * sizeof(MPI_Comm));
    mpi_config.every_one_flow_world      = (MPI_Comm *)malloc(flow_ranks * sizeof(MPI_Comm));
    mpi_config.alias_rank                = (int *)     malloc(flow_ranks * sizeof(int));

    MPI_Barrier(mpi_config.world);

    //Setup solvers
    ParticleSolver<double> *particle_solver = nullptr;
    FlowSolver<double>     *flow_solver     = nullptr;
    if (mpi_config.solver_type == PARTICLE)
    {
        uint64_t       local_particles_per_timestep   = particles_per_timestep / mpi_config.particle_flow_world_size;
        int            remainder_particles            = particles_per_timestep % mpi_config.particle_flow_world_size;

        const uint64_t reserve_particles_size         = 2 * (local_particles_per_timestep + 1) * ntimesteps;

        ParticleDistribution<double> *particle_dist = load_injector_particle_distribution(particles_per_timestep, local_particles_per_timestep, remainder_particles, &mpi_config, box_dim, mesh);
        // ParticleDistribution<double> *particle_dist = load_particle_distribution(particles_per_timestep, local_particles_per_timestep, remainder_particles, &mpi_config, mesh);
        particle_solver = new ParticleSolver<double>(&mpi_config, ntimesteps, delta, particle_dist, mesh, reserve_particles_size, stdout); 
    }
    else
    {
		#ifdef have_gpu
			AMGX_SAFE_CALL(AMGX_initialize());
		#else
			PETSC_COMM_WORLD = mpi_config.particle_flow_world;
			PetscInitialize(&argc, &argv, nullptr, nullptr);
		#endif
        flow_solver     = new FlowSolver<double>(&mpi_config, mesh, delta, stdout);
    }

	if (mpi_config.rank == 0)   cout << endl;
    setup_time += MPI_Wtime(); MPI_Barrier(mpi_config.world); 

    if (mpi_config.rank == 0)  printf("Mesh built in %6.2fs!\n\n", mesh_time);


    // Output mesh 
    MPI_Barrier(mpi_config.world); output_time -= MPI_Wtime(); 
    if ( output_iteration != -1 )
    {
        VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh, &mpi_config);
        vtk_writer->write_mesh("out/minicombust");
    }
    output_time += MPI_Wtime(); MPI_Barrier(mpi_config.world); 

    // Main loop
    if (mpi_config.rank == 0)  printf("Starting simulation..\n");
    MPI_Barrier(mpi_config.world);
    program_time -= MPI_Wtime();

    auto start = std::chrono::system_clock::now();
 
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
 
    if (mpi_config.rank == 0) std::cout << "STARTING CLOCK 1410 " << std::ctime(&start_time) << std::endl;

	for(uint64_t t = 0; t < ntimesteps; t++)
    {



        if (mpi_config.solver_type == PARTICLE)
        {
            particle_solver->timestep();
            
            if (((int64_t)(t % output_iteration) == output_iteration - 1))  
            {
                output_time -= MPI_Wtime();
                particle_solver->output_data(t+1);
                output_time += MPI_Wtime();
            }
        }
		else
		{
            flow_solver->timestep();
			
			if (((int64_t)(t % output_iteration) == output_iteration - 1))
            {
                output_time -= MPI_Wtime();
				flow_solver->output_data(t+1);
                output_time += MPI_Wtime();
            }	
		}
    }

    program_time += MPI_Wtime();
    MPI_Barrier(mpi_config.world);
    auto end = std::chrono::system_clock::now();

    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    if (mpi_config.rank == 0) std::cout << "ENDING CLOCK 1410 " << std::ctime(&end_time) << std::endl;
    

    if (mpi_config.rank == 0) printf("Done!\n\n");

    //Print logger stats and write performance counters
    if (LOGGER) 
    {
        if (mpi_config.solver_type == PARTICLE)
            particle_solver->print_logger_stats(ntimesteps, program_time);
        else
            flow_solver->print_logger_stats(ntimesteps, program_time);
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

	double precision = 5+log10(max({setup_time_max, program_time_max, 
								output_time_max, comp_time_max}));
    cout.precision(2);
    cout.setf(ios::fixed);

    if (mpi_config.rank == 0)
    {
        setup_time_avg    /= (double)mpi_config.world_size;
        program_time_avg  /= (double)mpi_config.world_size;
        output_time_avg   /= (double)mpi_config.world_size;

        cout << "Setup Time:            " << setw(precision) << setup_time_avg   << "s  (min " << setw(precision) << setup_time_min   << "s) " << "(max " << setw(precision) << setup_time_max    << "s)\n";
        cout << "Program Time:          " << setw(precision) << program_time_avg << "s  (min " << setw(precision) << program_time_min << "s) " << "(max " << setw(precision) << program_time_max  << "s)\n";
        cout << "Output Time:           " << setw(precision) << output_time_avg  << "s  (min " << setw(precision) << output_time_min  << "s) " << "(max " << setw(precision) << output_time_max   << "s)\n";
    }
	MPI_Barrier(mpi_config.world);
	if(mpi_config.particle_flow_rank == 0)
	{
		if(mpi_config.solver_type == PARTICLE)
		{
			comp_time_avg /= (double)mpi_config.particle_flow_world_size;
			cout << "Particle Compute Time: " << setw(precision) << comp_time_avg 
					<< "s  (min " << setw(precision) << comp_time_min << "s) (max "
					<< setw(precision) << comp_time_max << "s)\n"; 
		}
		else
		{
			comp_time_avg /= (double)mpi_config.particle_flow_world_size;
			cout << "Flow Compute Time:     " << setw(precision) << comp_time_avg
					<< "s  (min " << setw(precision) << comp_time_min << "s) (max "
					<< setw(precision) << comp_time_max << "s)\n";
		}
	}
	
	if (mpi_config.solver_type != PARTICLE)
	{
		#ifdef have_gpu
			flow_solver->AMGX_free();
			AMGX_SAFE_CALL(AMGX_finalize());
		#else
			PetscFinalize();
		#endif
	}
    
	//TODO: calling MPI_Finalize on "big" runs throws a 
	//PMPI_Finalize error for Device or resource busy 
	//need to look into this and work out what is going on.
	//can test with 0 7 241 -1 1 with 482 threads

    MPI_Win_free(&mpi_config.win_cell_centers);
    MPI_Win_free(&mpi_config.win_cells);
    MPI_Win_free(&mpi_config.win_cell_neighbours);
    MPI_Win_free(&mpi_config.win_points);
    MPI_Win_free(&mpi_config.win_cells_per_point);

	MPI_Finalize();
    return 0;
}
