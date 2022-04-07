#include <stdio.h>

#include "particles/ParticleSolver.hpp"
#include "visit/VisitWriter.hpp"



namespace minicombust::particles 
{
    using namespace minicombust::visit;

    template<class T>
    void ParticleSolver<T>::output_data(int timestep)
    {
        mesh->clear_particles_per_point_array();

        // Assign each particles to one of the vertexes of the bounding cell.
        for(int p = 0; p < current_particle; p++)  // Iterate through each particle
        {
            Particle<T> *particle     = particles + p;
            if ( particle->decayed )  continue;

            double closest_dist       = __DBL_MAX__;
            uint64_t closest_vertex   = UINT64_MAX;
            for (int i = 0; i < mesh->cell_size; i++)  // Iterate through the points of the cell that the particle is in
            {
                const uint64_t point_index = mesh->cells[particle->cell*mesh->cell_size + i];
                const double dist = magnitude(particle->x1 - mesh->points[point_index]);
                if ( dist < closest_dist )
                {
                    closest_dist   = dist;
                    closest_vertex = point_index;
                }
            }
            mesh->particles_per_point[closest_vertex] += 1;
        }

        VisitWriter<double> *vtk_writer = new VisitWriter<double>(mesh);
        vtk_writer->write_particles("minicombust", timestep);
    }

    template<class T>
    void ParticleSolver<T>::print_logger_stats(int timesteps, double runtime)
    {
        cout << "Particle Solver Stats:                         " << endl;
        cout << "\tParticles:                                   " << ((double)logger.num_particles)                                                                   << endl;
        cout << "\tParticles (per iter):                        " << particle_dist->particles_per_timestep                                                            << endl;
        cout << endl;
        cout << "\tCell checks:                                 " << ((double)logger.cell_checks)                                                                     << endl;
        cout << "\tCell checks (per iter):                      " << ((double)logger.cell_checks) / timesteps                                                         << endl;
        cout << "\tCell checks (per particle, per iter):        " << ((double)logger.cell_checks) / (((double)logger.num_particles)*timesteps)                        << endl;
        cout << endl;
        cout << "\tEdge adjustments:                            " << ((double)logger.position_adjustments)                                                            << endl;
        cout << "\tEdge adjustments (per iter):                 " << ((double)logger.position_adjustments) / timesteps                                                << endl;
        cout << "\tEdge adjustments (per particle, per iter):   " << ((double)logger.position_adjustments) / (((double)logger.num_particles)*timesteps)               << endl;
        cout << endl;
        cout << "\tBoundary Intersections:                      " << ((double)logger.boundary_intersections)                                                          << endl;
        cout << "\tDecayed Particles:                           " << round(10000.*(((double)logger.decayed_particles) / ((double)logger.num_particles)))/100. << "% " << endl;

        cout << endl;
        cout << "\tGFLOPS:                                      " << ((double)logger.flops)  / 1000000000.0                                                          << endl;
        cout << "\tPerformance (GFLOPS/s):                      " << (((double)logger.flops) / 1000000000.0) / runtime                                               << endl;
        cout << "\tMemory Loads (GB):                           " << ((double)logger.loads)  / 1000000000.0                                                          << endl;    
        cout << "\tMemory Stores (GB):                          " << ((double)logger.stores) / 1000000000.0                                                          << endl;    
        cout << "\tMemory Bandwidth (GB):                       " << (((double)logger.loads + (double)logger.stores) / 1000000000.0) /runtime                        << endl;    
        cout << "\tOperational Intensity:                       " << (double)logger.flops / ((double)logger.loads + (double)logger.stores)                           << endl;    
    }


    template<class T> 
    void ParticleSolver<T>::update_flow_field()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_flow_field.\n");

    }
            
    template<class T> 
    void ParticleSolver<T>::particle_release()
    {
        // TODO: Reuse decaying particle space
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: particle_release.\n");
        particle_dist->emit_particles(particles + current_particle);
        current_particle     += particle_dist->particles_per_timestep;
        logger.num_particles += particle_dist->particles_per_timestep;
    }

    template<class T> 
    void ParticleSolver<T>::solve_spray_equations()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: solve_spray_equations.\n");

        for (int c = 0; c < mesh->mesh_size; c++)
        {
            mesh->evaporated_fuel_mass_rate[c] = 0.0;
            mesh->particle_energy_rate[c]      = 0.0;
            mesh->particle_momentum_rate[c]    = {0.0, 0.0, 0.0};
        }

        if (LOGGER)
        {
            logger.stores += mesh->mesh_size * (sizeof(vec<T>) + 2 * sizeof(T)); 
        }

        // Solve spray equations
        for (int p = 0; p < current_particle; p++)
        {
            vec<T> total_vector_weight   = {0.0, 0.0, 0.0};
            T total_scalar_weight        = 0.0;

            vec<T> interp_gas_acc = {0.0, 0.0, 0.0};
            T interp_gas_pre      = 0.0;
            T interp_gas_tem      = 0.0;
            for (int n = 0; n < mesh->cell_size; n++)
            {
                const uint64_t node           = mesh->cells[particles[p].cell*mesh->cell_size + n]; 
                const vec<T> node_to_particle = particles[p].x1 - mesh->points[mesh->cells[particles[p].cell*mesh->cell_size + n]];

                vec<T> weight           = 1.0 / (node_to_particle * node_to_particle);
                T weight_magnitude = magnitude(weight);
                
                total_vector_weight   += weight;
                total_scalar_weight   += weight_magnitude;

                interp_gas_acc += weight           * nodal_gas_acceleration[node];
                interp_gas_pre += weight_magnitude * nodal_gas_pressure[node];
                interp_gas_tem += weight_magnitude * nodal_gas_temperature[node];
            }

            interp_gas_acc /= total_vector_weight;
            interp_gas_pre /= total_scalar_weight;
            interp_gas_tem /= total_scalar_weight;

            particles[p].solve_spray(mesh, 0.01, &logger, interp_gas_acc, interp_gas_pre, interp_gas_tem);
        }
    }

    template<class T> 
    void ParticleSolver<T>::update_particle_positions()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_particle_positions.\n");
        
        // Update particle positions
        for (int p = 0; p < current_particle; p++)
        {
            particles[p].timestep(mesh, 0.01, &logger);
        }
    }

    template<class T>
    void ParticleSolver<T>::update_spray_source_terms()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: update_spray_source_terms.\n");
    }

    template<class T> 
    void ParticleSolver<T>::map_source_terms_to_grid()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: map_source_terms_to_grid.\n");
    }

    template<class T> 
    void ParticleSolver<T>::interpolate_nodal_data()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("\tRunning fn: interpolate_data.\n");

        memset(nodal_counter, 0, mesh->points_size * sizeof(uint8_t));
        for (int n = 0; n < mesh->points_size; n++)
        {
            nodal_gas_acceleration[n] = {0.0, 0.0, 0.0};
            nodal_gas_pressure[n]     = 0.0;
            nodal_gas_temperature[n]  = 0.0;
        }

        
        for (int c = 0; c < mesh->mesh_size; c++)
        {
            // Can we mark all neighbours that we need for average
            vec<T> acc           = mesh->gas_acceleration[c];
            vec<T> acc_grad      = mesh->gas_acceleration_gradient[c];
            T pressure           = mesh->gas_pressure[c];
            T pressure_grad      = mesh->gas_pressure_gradient[c];
            T temp               = mesh->gas_temperature[c];
            T temp_grad          = mesh->gas_temperature_gradient[c];
            
            for (int n = 0; n < mesh->cell_size; n++)
            {
                nodal_counter[n]++;

                vec<T> direction           = mesh->points[mesh->cells[c*mesh->cell_size + n]] - mesh->cell_centres[c];
                nodal_gas_acceleration[n] += acc      + dot_product(acc_grad, direction);
                nodal_gas_pressure[n]     += pressure + dot_product(pressure_grad, direction);
                nodal_gas_temperature[n]  += temp     + dot_product(temp_grad, direction);
            }
        }

        
        for (int n = 0; n < mesh->points_size; n++)
        {
            const T node_neighbours       = 4; // Cube specific
            const T boundary_neighbours   = node_neighbours - nodal_counter[n]; // If nodal counter is not 4, we are on a boundary

            nodal_gas_acceleration[n] += boundary_neighbours * (vec<T>){0.1, 0.1, 0.1};
            nodal_gas_pressure[n]     += boundary_neighbours * 6.e3;
            nodal_gas_temperature[n]  += boundary_neighbours * 1500.;
            
            nodal_gas_acceleration[n] /= node_neighbours;
            nodal_gas_pressure[n]     /= node_neighbours;
            nodal_gas_temperature[n]  /= node_neighbours;
        }
    }

    

    template<class T> 
    void ParticleSolver<T>::timestep()
    {
        if (PARTICLE_SOLVER_DEBUG)  printf("Start particle timestep\n");
        // update_flow_field();
        interpolate_nodal_data();
        particle_release();
        solve_spray_equations();
        update_particle_positions();
        // update_spray_source_terms();
        // map_source_terms_to_grid();
        if (PARTICLE_SOLVER_DEBUG)  printf("Stop particle timestep\n");
    }

}   // namespace minicombust::particles 