#include <stdio.h>
#include <ctime>

#include "examples/mesh_examples.hpp"
#include "examples/particle_examples.hpp"
#include "tests/particle_tests.hpp"
#include "geometry/Mesh.hpp"


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
    printf("\n");

} 


int main (int argc, char ** argv)
{
    printf("Starting miniCOMBUST..\n");
    Mesh<double> *mesh = nullptr;
    ParticleDistribution<double> *particle_dist = nullptr;
    uint64_t ntimesteps = 0;

    switch (argc)
    {
        case 2:
            if(strcmp(argv[1], "tests") == 0)
            {
                printf("Running tests\n");
                run_particle_tests();
                exit(0);
            }   
        
        default:
            printf("No meshes supplied. Running built in example instead.\n\n");
            const double box_dim                  = 100;
            const uint64_t elements_per_dim       = 100;
            const uint64_t particles_per_timestep = 1;
            ntimesteps                            = 100;
            mesh          = load_mesh(box_dim, elements_per_dim);
            particle_dist = load_particle_distribution(particles_per_timestep, mesh);
    }

    


    FlowSolver<double>     *flow_solver     = new FlowSolver<double>();
    ParticleSolver<double> *particle_solver = new ParticleSolver<double>(ntimesteps, particle_dist, mesh);
    cout << endl;

    const clock_t begin_time = clock();
    for(int t = 0; t < ntimesteps; t++)
    {
        printf("Timestep %d..\n\n", t);
        timestep(flow_solver, particle_solver);

        
    }
    const clock_t prog_time = clock();

    cout << "\nProgram Runtime: " << float( prog_time - begin_time ) /  CLOCKS_PER_SEC << "s" << endl;
    particle_solver->output_data(ntimesteps-1);
    cout <<   "Output Time:     " << float( clock () - prog_time ) /  CLOCKS_PER_SEC << "s" << endl;



    

    return 0;
}