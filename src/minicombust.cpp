#include <stdio.h>

#include "examples/mesh_examples.hpp"
#include "examples/particle_examples.hpp"
#include "geometry/Mesh.hpp"
#include "visit/VisitWriter.hpp"


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
    Mesh<double> *boundary_mesh;
    Mesh<double> *global_mesh;
    ParticleDistribution<double> *particle_dist;

    switch (argc)
    {
        // case 2:
        //     printf("Mesh input is: %s\n", argv[1]);
        //     break;
        
        default:
            const double box_dim                  = 100;
            const double elements_per_dim         = 10;
            const uint64_t particles_per_timestep = 1000;
            printf("No meshes supplied. Running built in example instead.\n\n");
            boundary_mesh = load_boundary_box_mesh(box_dim);
            global_mesh   = load_global_mesh(box_dim, elements_per_dim);
            particle_dist = load_particle_distribution(particles_per_timestep);
    }

    


    const uint64_t ntimesteps = 5;
    FlowSolver<double>     *flow_solver     = new FlowSolver<double>();
    ParticleSolver<double> *particle_solver = new ParticleSolver<double>(ntimesteps, particle_dist, boundary_mesh, global_mesh);

    cout << endl;
    for(int t = 0; t < ntimesteps; t++)
    {
        printf("Timestep %d..\n\n", t);
        timestep(flow_solver, particle_solver);
    }

    VisitWriter<double> *vtk_writer = new VisitWriter<double>(global_mesh, boundary_mesh);
    vtk_writer->write_file();

    

    return 0;
}