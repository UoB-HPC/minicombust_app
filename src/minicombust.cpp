#include <stdio.h>
#include <ctime>
#include <inttypes.h>

#include "examples/mesh_examples.hpp"
#include "examples/particle_examples.hpp"
#include "geometry/Mesh.hpp"
#include "utils/utils.hpp"

#include "particles/ParticleSolver.inl"
#include "flow/FlowSolver.inl"


using namespace minicombust::flow;
using namespace minicombust::particles;
using namespace minicombust::visit;

int main (int argc, char ** argv)
{
	//TODO:reduce mega long lines
	//TODO:sort out vtk_output.
	//TODO:ideal flow tanks seem to cause a segfault
	
    // MPI Initialisation 
    MPI_Init(NULL, NULL);
    MPI_Config mpi_config;

    mpi_config.world = MPI_COMM_WORLD;

    // Create overall world
    MPI_Comm_rank(mpi_config.world,  &mpi_config.rank);
    MPI_Comm_size(mpi_config.world,  &mpi_config.world_size);

    int flow_ranks = atoi(argv[1]);
    int particle_ranks     = mpi_config.world_size - flow_ranks;

    // If rank < given number of particle ranks.
    mpi_config.solver_type = (mpi_config.rank >= flow_ranks); // 1 for particles, 0 for flow
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
    const uint64_t ntimesteps                   = (argc > 5) ? atoi(argv[5]) : 1500;
    const double   delta                        = 1.0e-5;
    const int64_t output_iteration              = (argc > 4) ? atoi(argv[4]) : 10;
    const uint64_t particles_per_timestep       = (argc > 2) ? atoi(argv[2]) : 10;
   

    // Mesh Configuration
    const uint64_t modifier                = (argc > 3) ? atoi(argv[3]) : 10;
    vec<double>    box_dim                 = {0.05*modifier*2, 0.05*modifier, 0.05*modifier};
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
		PETSC_COMM_WORLD = mpi_config.particle_flow_world;
		PetscInitialize(&argc, &argv, nullptr, nullptr);
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
        vtk_writer->write_mesh("out/mesh/minicombust");
    }
    output_time += MPI_Wtime(); MPI_Barrier(mpi_config.world); 

    // Main loop
    if (mpi_config.rank == 0)  printf("Starting simulation..\n");
    MPI_Barrier(mpi_config.world);
    program_time -= MPI_Wtime();

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
		}
    }
    program_time += MPI_Wtime();
    MPI_Barrier(mpi_config.world);
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
		PetscFinalize();
	}
    
    MPI_Win_free(&mpi_config.win_cell_centers);
    MPI_Win_free(&mpi_config.win_cells);
    MPI_Win_free(&mpi_config.win_cell_neighbours);
    MPI_Win_free(&mpi_config.win_points);
    MPI_Win_free(&mpi_config.win_cells_per_point);

	MPI_Finalize();
    return 0;
}
