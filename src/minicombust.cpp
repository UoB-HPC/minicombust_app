#include <stdio.h>
#include <ctime>

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
    printf("Starting miniCOMBUST..\n");
    Mesh<double> *mesh = nullptr;
    ParticleDistribution<double> *particle_dist = nullptr;
    uint64_t ntimesteps = 0;

    switch (argc)
    {
        default:
            printf("No meshes supplied. Running built in example instead.\n\n");
            const double box_dim                  = 100;
            const uint64_t elements_per_dim       = 100;
            const uint64_t particles_per_timestep = 1000;
            ntimesteps                            = 70;
            mesh          = load_mesh(box_dim, elements_per_dim);
            particle_dist = load_particle_distribution(particles_per_timestep, mesh);
    }

    
    FlowSolver<double>     *flow_solver     = new FlowSolver<double>();
    ParticleSolver<double> *particle_solver = new ParticleSolver<double>(ntimesteps, particle_dist, mesh);
    cout << endl;

    float program_ticks = 0.f, output_ticks = 0.f;
    const clock_t output = clock(); 
    VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh);
    vtk_writer->write_mesh("minicombust");
    output_ticks += float(clock() - output);

    uint64_t print_iteration = 10;

    printf("Starting simulation..\n");
    for(int t = 0; t < ntimesteps; t++)
    {
        if (t % 10 == 0)  printf("Timestep %d of %llu..\n", t, ntimesteps);
        const clock_t timestep_time  = clock();
        timestep(flow_solver, particle_solver);
        program_ticks += float(clock() - timestep_time);
        if ((t % print_iteration == print_iteration - 1))  
        {
            const clock_t output = clock(); 
            particle_solver->output_data(t+1);
            output_ticks += float(clock() - output);
        }

        
    }
    printf("Done!\n\n");

    if (LOGGER)
    {
        particle_solver->print_logger_stats(ntimesteps, program_ticks /  CLOCKS_PER_SEC);
    }
    cout << "\nProgram Runtime: " << program_ticks /  CLOCKS_PER_SEC << "s" << endl;
    cout <<   "Output Time:     " << output_ticks /  CLOCKS_PER_SEC  << "s" << endl;



    

    return 0;
}