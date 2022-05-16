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

template<typename F, typename P>
void timestep(FlowSolver<F> *flow_solver, ParticleSolver<P> *particle_solver)
{
    flow_solver->timestep();
    particle_solver->timestep();
} 

int main (int argc, char ** argv)
{

    MPI_Init(NULL, NULL);
    MPI_Config mpi_config;
    mpi_config.world = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi_config.world, &mpi_config.rank);
    MPI_Comm_size(mpi_config.world, &mpi_config.world_size);

    if (mpi_config.rank == 0)  printf("Starting miniCOMBUST..\n");

    // Run Configuration
    const uint64_t ntimesteps                   = 1500;
    const double   delta                        = 2.5e-6;
    const uint64_t output_iteration             = -1;
    const uint64_t particles_per_timestep       = (argc > 1) ? atoi(argv[1]) : 10;
    uint64_t local_particles_per_timestep       = particles_per_timestep / mpi_config.world_size;
    uint64_t remainder_particles                = particles_per_timestep % mpi_config.world_size;
    if (mpi_config.rank < remainder_particles)  local_particles_per_timestep++;

    // Mesh Configuration
    const double box_dim                        = 0.3;
    const uint64_t elements_per_dim             = 50;
    const uint64_t reserve_particles_size       = 2 * local_particles_per_timestep * ntimesteps;

    // Performance
    float setup_time = 0.f, program_time = 0.f, output_time = 0.f;

    // Perform setup and benchmark cases
    MPI_Barrier(mpi_config.world); setup_time -= MPI_Wtime(); 
    Mesh<double> *mesh                          = load_mesh(&mpi_config, box_dim, elements_per_dim);
    ParticleDistribution<double> *particle_dist = load_particle_distribution(local_particles_per_timestep, mesh);

    FlowSolver<double>     *flow_solver     = new FlowSolver<double>();
    ParticleSolver<double> *particle_solver = new ParticleSolver<double>(&mpi_config, ntimesteps, delta, particle_dist, mesh, ntimesteps, reserve_particles_size); 
    if (mpi_config.rank == 0)   cout << endl;
    MPI_Barrier(mpi_config.world); setup_time += MPI_Wtime(); 

    // Output mesh 
    MPI_Barrier(mpi_config.world); output_time -= MPI_Wtime(); 
    if (mpi_config.rank == 0)
    {
        VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh);
        vtk_writer->write_mesh("minicombust");
         
    }
    MPI_Barrier(mpi_config.world); output_time += MPI_Wtime();

    // Main loop
    if (mpi_config.rank == 0)  printf("Starting simulation..\n");
    MPI_Barrier(mpi_config.world);
    program_time -= MPI_Wtime();
    for(uint64_t t = 0; t < ntimesteps; t++)
    {
        timestep(flow_solver, particle_solver);
        // MPI_Barrier(mpi_config.world);
        
        // if ((t % output_iteration == output_iteration - 1))  
        // {
        //     output_time -= MPI_Wtime();
        //     particle_solver->output_data(t+1);
        //     output_time += MPI_Wtime();
        // }
    }
    MPI_Barrier(mpi_config.world);
    program_time += MPI_Wtime();


    if (mpi_config.rank == 0) printf("Done!\n\n");

    for (int i = 0; i < mpi_config.world_size; i++)
    {
        if (mpi_config.rank == i)
        {
            cout << "\nRANK " << mpi_config.rank << ":" << endl;
            if (LOGGER)  particle_solver->print_logger_stats(ntimesteps, program_time);
            cout << "  Setup Time:              " << setup_time << "s" << endl;
            cout << "  Program Time:            " << program_time << "s" << endl;
            cout << "  Program Time (per iter): " << program_time / ntimesteps << "s" << endl;
            cout << "  Output Time:             " << output_time  << "s" << endl;
        }
        MPI_Barrier(mpi_config.world);

    }


    MPI_Finalize();

    

    return 0;
}
