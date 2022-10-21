#pragma once

#include "utils/utils.hpp"

namespace minicombust::flow 
{
    template<class T>
    class FlowSolver 
    {
        private:

            Mesh<T> *mesh;

            uint64_t neighbours_size;            
            uint64_t *neighbour_indexes;
           
            T turbulence_field;
            T combustion_field;
            T flow_field;

            

        public:
            MPI_Config *mpi_config;
            PerformanceLogger<T> performance_logger;

            double time_stats[11] = {0.0};

            FlowSolver(MPI_Config *mpi_config, Mesh<T> *mesh) : mesh(mesh), mpi_config(mpi_config)
            {
                const size_t cell_index_array_size   = mesh->mesh_size * sizeof(uint64_t);

                neighbour_indexes                 = (uint64_t*)malloc(cell_index_array_size);


                neighbours_size = mesh->mesh_size;
                performance_logger.init_papi();
                performance_logger.load_papi_events(mpi_config->rank);
            }

            void update_flow_field(bool receive_particle);  // Synchronize point with flow solver
            
            void solve_combustion_equations();
            void update_combustion_fields();

            void solve_turbulence_equations();
            void update_turbulence_fields();

            void solve_flow_equations();

            void timestep();


    }; // class FlowSolver

}   // namespace minicombust::flow 