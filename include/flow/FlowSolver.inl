#include <stdio.h>

#include "flow/FlowSolver.hpp"


namespace minicombust::flow 
{
    
    template<typename T> void FlowSolver<T>::update_flow_field()
    {
        printf("\tRunning function update_flow_field.");
    } 
    
    template<typename T> void FlowSolver<T>::solve_combustion_equations()
    {
        printf("\tRunning function solve_combustion_equations.");
    }

    template<typename T> void FlowSolver<T>::update_combustion_fields()
    {
        printf("\tRunning function update_combustion_fields.");
    }

    template<typename T> void FlowSolver<T>::solve_turbulence_equations()
    {
        printf("\tRunning function solve_turbulence_equations.");
    }

    template<typename T> void FlowSolver<T>::update_turbulence_fields()
    {
        printf("\tRunning function update_turbulence_fields.");
    }

    template<typename T> void FlowSolver<T>::solve_flow_equations()
    {
        printf("\tRunning function solve_flow_equations.");
    }

    template<typename T> void FlowSolver<T>::timestep()
    {
        printf("Start flow timestep\n");
        // update_flow_field();
        // solve_combustion_equations();
        // update_combustion_fields();
        // solve_turbulence_equations();
        // update_turbulence_fields();
        // solve_flow_equations();
        printf("Stop flow timestep\n");
    }

}   // namespace minicombust::flow 