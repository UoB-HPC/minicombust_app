#pragma once

#include "utils/utils.hpp"

namespace minicombust::flow 
{
    template<class T>
    class FlowSolver 
    {
        private:
            T turbulence_field;
            T combustion_field;
            T flow_field;

            Mesh<T> *global_mesh;

            // Velocity (cell centered)
            T* velocity_U;
            T* velocity_V;
            T* velocity_W;

            // Velocity gradients
            vec<T>* velocity_dUdX;
            vec<T>* velocity_dVdX;
            vec<T>* velocity_dWdX;

            // Pressure (cell centered)
            T* pressure;
            vec<T>* pressure_dPdX;

        public:

            FlowSolver(Mesh<T> *global_mesh) : global_mesh(global_mesh) {

                // Allocate velocity fields + gradients
                velocity_U = (T*)malloc(sizeof(T) * global_mesh->mesh_size);
                velocity_V = (T*)malloc(sizeof(T) * global_mesh->mesh_size);
                velocity_W = (T*)malloc(sizeof(T) * global_mesh->mesh_size);

                velocity_dUdX = (vec<T>*)malloc(sizeof(vec<T>) * global_mesh->mesh_size);
                velocity_dVdX = (vec<T>*)malloc(sizeof(vec<T>) * global_mesh->mesh_size);
                velocity_dWdX = (vec<T>*)malloc(sizeof(vec<T>) * global_mesh->mesh_size);

                printf("Allocating velocities + gradients, %.2f MB\n", (float)((sizeof(vec<T>) + sizeof(T)) * global_mesh->mesh_size * 3) / 1000000.0);

                // Initalize to default value
                const T initial_velocity = 0.7;
                for (auto i = 0; i < global_mesh->mesh_size; ++i) {
                    velocity_U[i] = initial_velocity;
                    velocity_V[i] = initial_velocity;
                    velocity_W[i] = initial_velocity;

                    velocity_dUdX[i] = {0.0, 0.0, 0.0};
                    velocity_dVdX[i] = {0.0, 0.0, 0.0};
                    velocity_dWdX[i] = {0.0, 0.0, 0.0};
                }

                // Allocate pressure
                pressure = (T*)malloc(sizeof(T) * global_mesh->mesh_size);
                pressure_dPdX = (vec<T>*)malloc(sizeof(vec<T>) * global_mesh->mesh_size);
                printf("Allocating pressure + gradient, %.2f MB\n", (float)((sizeof(vec<T>)+sizeof(T)) * global_mesh->mesh_size) / 1000000.0);

                // Initalize to default value
                const T initial_pressure = 0.35;
                for (auto i = 0; i < global_mesh->mesh_size; ++i) {
                    pressure[i] = initial_pressure;
                    pressure_dPdX[i] = {0.0, 0.0, 0.0};
                }

            }

            void update_flow_field();  // Synchronize point with flow solver
            
            void solve_combustion_equations();
            void update_combustion_fields();

            void solve_turbulence_equations();
            void update_turbulence_fields();

            void solve_flow_equations();

            void timestep();


    }; // class FlowSolver

}   // namespace minicombust::flow 
