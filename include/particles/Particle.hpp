#pragma once

#include "utils/utils.hpp"

using namespace minicombust::utils;

namespace minicombust::particles 
{
    using namespace std; 

    
    template<class T>
    class Particle 
    {
        // private:

        public:
            vec<T> x0 = 0.0;        // starting coordinates
            vec<T> x1 = 0.0;        // current coordinates

            vec<T> v0 = 0.0;        // starting velocity
            vec<T> v1 = 0.0;        // current velocity
            
            vec<T> a1 = 0.0;        // current acceleration
            
            // TODO: Perform check for if particle is actually starting in cell 0.
            int cell0 = 0;         // starting cell
            int cell1 = 0;         // current cell
            // int face  = 0;         // current face the particle is on

            
            bool wall  = false;

            T dens0 = -1.0;       // current density (-1 = undefined)
            T diam0 =  1.0;
            T mass0 =  1.0;

            Particle(vec<T> start, vec<T> velocity, vec<T> acceleration) : x0(start), x1(start), v0(velocity), v1(velocity), a1(acceleration)
            { }

            void timestep()
            {
                // TODO: Timestep needs to be definitive amount of time. For now, it is a constant 0.01s.
                x1 = x1 + v1*0.01;
                v1 = v1 + a1*0.01;
            }

            
    }; // class Particle
 
}   // namespace minicombust::particles 