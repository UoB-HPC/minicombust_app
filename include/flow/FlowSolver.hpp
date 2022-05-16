#pragma once

namespace minicombust::flow 
{
    template<class T>
    class FlowSolver 
    {
        private:
            T rank;
            T turbulence_field;
            T combustion_field;
            T flow_field;

        public:

            void update_flow_field();  // Synchronize point with flow solver
            
            void solve_combustion_equations();
            void update_combustion_fields();

            void solve_turbulence_equations();
            void update_turbulence_fields();

            void solve_flow_equations();

            void timestep();


    }; // class FlowSolver

}   // namespace minicombust::flow 