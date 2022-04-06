#pragma once

#include <cmath>
#include <limits>
#include <iomanip>
#include <bitset>
#include <vector>

#include "utils/utils.hpp"
#include "geometry/Mesh.hpp"


#define PARTICLE_DEBUG 0

namespace minicombust::particles 
{
    using namespace std; 
    using namespace minicombust::utils;
    using namespace minicombust::geometry; 

    #define INVALID_FACE 1;
    #define VALID_FACE 0;
    enum INTERSECT_PLANES { POSSIBLE = 0, IMPOSSIBLE = 1, PARALLEL = 3}; 
    
    template<class T>
    class Particle 
    {
        private:

            double tetrahedral_volume(vec<T> *A, vec<T> *B, vec<T> *C, vec<T> *D)
            {
                // TODO: Is determinent version faster?
                // Algorithm Ref: https://math.stackexchange.com/questions/1603651/volume-of-tetrahedron-using-cross-and-dot-product

                const vec<T> a = *B - *C;
                const vec<T> b = *D - *C;
                const vec<T> c = *A - *C;

                return abs(dot_product(cross_product(a, b), c)) / 6.0;

                // logger->flops  += 31;
            }

            bool check_cell(uint64_t current_cell, Mesh<T> *mesh)
            {
                // TODO: Currently is solely a cube partial volume algorithm. Tetra partial volume algorithm;
                uint64_t *current_cell_points = mesh->cells + current_cell*mesh->cell_size;
                vec<T> box_size = mesh->points[current_cell_points[H_VERTEX]] - mesh->points[current_cell_points[A_VERTEX]];
                double total_volume  = abs(box_size.x*box_size.y*box_size.z);

                double partial_volumes = 0.0;
                for (int i=0; i < mesh->cell_size; i++)
                {
                    vec<T> partial_box_size = mesh->points[current_cell_points[i]] - x1;
                    partial_volumes += abs(partial_box_size.x*partial_box_size.y*partial_box_size.z);
                }

                // if (PARTICLE_DEBUG)  cout << "\t\tpartial_volumes - total  " << partial_volumes-total_volume << endl;
                if (abs(partial_volumes-total_volume) < 5.0e-10)  return true;
                return false;
                
                // logger->flops  += 48;
            }

            void update_cell(Mesh<T> *mesh, particle_logger *logger)
            {

                if ( check_cell(cell, mesh) )
                {
                    if (PARTICLE_DEBUG)  cout << "\t\tParticle is still in cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                    x0 = x1;
                    v0 = v1;

                    if (LOGGER)
                    {
                        logger->flops  += 48;
                        logger->loads  += 9 * sizeof(vec<T>) + 8 * sizeof(uint64_t);  // 8 vertexes + 8 indexes + position
                        logger->stores += 2 * sizeof(vec<T>);     // 2 vectors(position, velocity)
                        logger->cell_checks++;
                    }
                } 
                else
                {
                    bool found_cell     = false;
                    vec<T> artificial_A = mesh->cell_centres[cell];
                    vec<T> *A = &artificial_A;
                    // vec<T> *A = &x0;
                    vec<T> *B = &x1;
                    
                    uint64_t face_mask = POSSIBLE; // Prevents double interceptions     

                    vec<T> z_vec = mesh->points[mesh->cells[cell*mesh->cell_size + E_VERTEX]] - mesh->points[mesh->cells[cell*mesh->cell_size + A_VERTEX]];
                    vec<T> x_vec = mesh->points[mesh->cells[cell*mesh->cell_size + B_VERTEX]] - mesh->points[mesh->cells[cell*mesh->cell_size + A_VERTEX]];
                    vec<T> y_vec = mesh->points[mesh->cells[cell*mesh->cell_size + C_VERTEX]] - mesh->points[mesh->cells[cell*mesh->cell_size + A_VERTEX]];

                    vec<T> x_delta   = x1 - x0;
                    // Detects which cells you could possibly hit. Should at least halve the number of face checks.
                    T dot = dot_product(x_delta, z_vec); 
                    face_mask = face_mask | (((dot == 0) ? PARALLEL : IMPOSSIBLE) << ((dot >= 0) ? 0 : 1));
                    dot = dot_product(x_delta, x_vec); 
                    face_mask = face_mask | (((dot == 0) ? PARALLEL : IMPOSSIBLE) << ((dot >= 0) ? 2 : 3)); 
                    dot = dot_product(x_delta, y_vec); 
                    face_mask = face_mask | (((dot == 0) ? PARALLEL : IMPOSSIBLE) << ((dot >= 0) ? 4 : 5)); 

                    while (!found_cell)
                    {
                        uint64_t *current_cell_points = mesh->cells + cell*mesh->cell_size;       
                        
                        uint64_t intercepted_face_id = 0;
                        uint64_t intercepted_faces   = 0;

                        

                        for (int face = 0; face < 6; face++)
                        {
                            if ((1 << face) & face_mask)  continue;

                            if (PARTICLE_DEBUG) cout << "\t\t\tTrying face " << mesh->get_face_string(face) << " " << bitset<6>(face_mask) << endl; 

                            // Check whether particle - moving from A to B - intercepts face CDEF. If LHS == RHS, particle intercepts this face.
                            vec<T> *C = &mesh->points[current_cell_points[CUBE_FACE_VERTEX_MAP[face][0]]];
                            vec<T> *D = &mesh->points[current_cell_points[CUBE_FACE_VERTEX_MAP[face][1]]];
                            vec<T> *E = &mesh->points[current_cell_points[CUBE_FACE_VERTEX_MAP[face][2]]];
                            vec<T> *F = &mesh->points[current_cell_points[CUBE_FACE_VERTEX_MAP[face][3]]];

                            const double LHS = tetrahedral_volume(A, C, D, B)
                                             + tetrahedral_volume(A, D, F, B)
                                             + tetrahedral_volume(A, F, E, B)
                                             + tetrahedral_volume(A, E, C, B);

                            const double RHS = tetrahedral_volume(A, C, D, F)
                                             + tetrahedral_volume(A, E, C, F)
                                             + tetrahedral_volume(B, C, D, F)
                                             + tetrahedral_volume(B, E, C, F);

                            if (abs(LHS - RHS) < 5e-12 && LHS != 0)  
                            {
                                intercepted_face_id = face;
                                intercepted_faces++;
                                
                                if (PARTICLE_DEBUG)  cout << "\t\t\tIntercepted Face " << mesh->get_face_string(face) << " num faces =  " << intercepted_faces << endl;
                                if (PARTICLE_DEBUG)  printf("\t\t\t%.20f == %.20f\n", LHS, RHS);
                            }
                        }

                        if (LOGGER)
                        {
                            logger->flops  += 840;
                            logger->loads  += 10 * sizeof(vec<T>) + 9 * sizeof(uint64_t);  // 8 vertexes + 8 indexes + cell_centre + cell_id 
                            logger->stores += 2 * sizeof(vec<T>) + 1 * sizeof(uint64_t);   // 2 vectors(position, velocity) + cell index
                            logger->cell_checks++;
                        }

                        
                        if (intercepted_faces > 1)
                        {
                            vec<T> r = vec<T> { static_cast<double>(rand())/(RAND_MAX), static_cast<double>(rand())/(RAND_MAX), static_cast<double>(rand())/(RAND_MAX) } ;
                            artificial_A = mesh->cell_centres[cell] + 0.1 * r*mesh->cell_size_vector;
                            if (PARTICLE_DEBUG)  cout << "\t\t\tNo faces, artificial position " <<  print_vec(artificial_A) << endl;
                            if (LOGGER)
                            {
                                logger->position_adjustments++;
                            }
                            continue;
                        }

                        cell = mesh->cell_neighbours[cell*mesh->faces_per_cell + intercepted_face_id];
                        if (PARTICLE_DEBUG)  cout << "\t\tMoving to cell " << cell << " " << mesh->get_face_string(intercepted_face_id) << " direction" << endl;  

                        // If intercepted with boundary of mesh, decay particle. TODO: Rebound them?
                        if ( cell == MESH_BOUNDARY ) 
                        {
                            if (PARTICLE_DEBUG)  cout << "\t\tParticle decayed after leaving the grid, x1: " << print_vec(x1) << " " << cell <<  endl ;
                            decayed = true;
                            if (LOGGER)
                            {   
                                logger->boundary_intersections++;
                                logger->decayed_particles++;
                            }
                            return;
                        }

                        // Is particle in this new cell?
                        if (check_cell(cell, mesh)) 
                        {
                            found_cell = true;
                            x0 = x1;
                            v0 = v1;
                            if (PARTICLE_DEBUG)  cout << "\t\tParticle has moved to cell " << cell << " " << ", x1: " << print_vec(x1) << endl ;
                        }
                        else { // Particle isn't in immediate neighbour
                            face_mask    = face_mask | (IMPOSSIBLE << (intercepted_face_id ^ 1));  // E.G If we detect interception with front face, don't trigger back face interception next time. // TOD0: Needed now?
                            if (PARTICLE_DEBUG)  cout  << "\t\tParticle isn't in neighbour cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                        }
                    }

                }
            }



        public:
            vec<T> x0 = 0.0;             // coordinates at timestep beginning
            vec<T> x1 = 0.0;             // coordinates at next timestep

            vec<T> v0 = 0.0;             // velocity at timestep beginning
            vec<T> v1 = 0.0;             // velocity at next timestep
            
            vec<T> a1 = 0.0;             // acceleration at next timestep
            
                        
            uint64_t cell = -1;          // cell at timestep beginning

            bool decayed = false;

            

            
            bool wall  = false;

            T mass        = 0.1;           // DUMMY_VAL Current mass (kg)
            T temp        = 373.0;         // DUMMY_VAL Current surface temperature (Kelvin)
            T density     = 1.0;           // DUMMY_VAL Current density (kg/m^3)
            T diameter    = 0.01;          // DUMMY_VAL Relationship between mass and diameter? Droplet is assumed to be spherical.



            Particle(Mesh<T> *mesh, vec<T> start, vec<T> velocity, vec<T> acceleration) : x0(start), x1(start), v0(velocity), v1(velocity), a1(acceleration)
            { 
                for (int c = 0; c < mesh->mesh_size; c++)
                {
                    if (check_cell(c, mesh))  
                    {
                        cell = c;
                        break;
                    }
                }
                if (cell == -1)
                {
                    cout << "Particle is not in mesh!!!!" << endl;
                    exit(0);
                }
                if (PARTICLE_DEBUG)  cout  << "\t\tParticle is starting in " << cell << ", x0: " << print_vec(x0) << " v0: " << print_vec(v0) <<  endl ;
            }

            uint64_t timestep(Mesh<T> *mesh, double delta, particle_logger *logger)  // Update position
            {
                if (decayed)  return cell;

                // if (PARTICLE_DEBUG)  cout << "Beginning of timestep: x0: " << print_vec(x0) << " v0 " << print_vec(v0) << endl; 
                x1 = x0 + v0 * delta;
                v1 = v0 + a1 * delta;

                if (LOGGER)
                {
                    logger->flops  += 12;
                    logger->loads  += 4 * sizeof(vec<T>);  // 3 vectors(position, velocity, acceleration)
                    logger->stores += 2 * sizeof(vec<T>);  // 2 vectors(position, velocity)
                }
                // if (PARTICLE_DEBUG)  cout << "End of timestep: x1: " << print_vec(x1) << " v1 " << print_vec(v1) << endl; 
               
                // Check if particle is in the current cell. Tetras = Volume/Area comparison method. https://www.peertechzpublications.com/articles/TCSIT-6-132.php.
                update_cell(mesh, logger);

                return cell; 
            } 

            inline void solve_spray(Mesh<T> *mesh, double delta, particle_logger *logger)
            {
                // Inputs from flow: relative_acc, relative_gas_liq_vel, kinematic viscoscity?, air_temp, air_pressure
                // Scenario constants: omega?, latent_heat, droplet_pressure?, evaporation_constant
                // Calculated outputs: acceleration, droplet surface temperature, droplet mass, droplet diameter
                // Calculated outputs for flow: evaporated mass?
                // if (decayed) return;

                // TODO: Remove DUMMY_VALs
                // SOLVE SPRAY/DRAG MODEL  https://www.sciencedirect.com/science/article/pii/S0021999121000826?via%3Dihub
                const vec<T> relative_drop_acc           = {0.01, 0.01, 0.01};            // DUMMY_VAL Relative acceleration between droplet and the fluid
                const vec<T> relative_drop_vel           = relative_drop_acc * delta;     // DUMMY_VAL Relative velocity between droplet and the fluid
                const T relative_gas_liq_vel             = 0.1;                           // DUMMY_VAL Relative acceleration between the gas and liquid phase.

                const T omega               = 0.1;                                        // DUMMY_VAL What is this?
                const T kinematic_viscosity = 1.48e-5;                                    // DUMMY_VAL 
                const T reynolds            = density * relative_gas_liq_vel * diameter / kinematic_viscosity;

                const T droplet_frontal_area  = M_PI * (diameter / 2.) * (diameter / 2.);

                // Drag coefficient
                const T drag_coefficient = ( reynolds <= 1000 ) ? 24 * (1 + 0.15 * pow(reynolds, 0.687)) : 0.424;

                // const vec<T> body_force    = Should we account for this?
                const vec<T> virtual_force = (-0.5 * density * omega) * relative_drop_acc;
                const vec<T> drag_force    = (drag_coefficient * reynolds  * 0.5 * density * relative_gas_liq_vel *  droplet_frontal_area) * relative_drop_vel;
                
                a1 = (virtual_force + drag_force) / mass; 
                
                if (LOGGER)
                {
                    logger->flops  += 29;
                    logger->loads  += 3 * sizeof(vec<T>) + 3 * sizeof(T);  // 3 vectors(drop, rel drop, gas_liq velocities), 3 fields(diameter, density, mass)
                    logger->stores += 1 * sizeof(vec<T>);                  // Acceleration
                }



                // SOLVE EVAPORATION MODEL https://arc.aiaa.org/doi/pdf/10.2514/3.8264 
                // Amount of spray evaporation is used in the modified transport equation of mixture fraction (each timestep).
                const T air_pressure           = 6.e3;
                const T fuel_vapour_pressure   = exp((14.2-2777.) / (temp - 43));                     // DUMMY_VAL fuel vapor at drop surface (kP)
                const T pressure_relation      = air_pressure / fuel_vapour_pressure;                 // DUMMY_VAL Clausius-Clapeyron relation. air pressure / fuel vapour pressure.
                const T fuel_pressure          = 29. / 100.;                                          // DUMMY_VAL molecular weight air / molecular weight fuel
                const T mass_fraction          = 1 / (1 + (pressure_relation - 1) * fuel_pressure);   // Mass fraction of fuel vapour at the droplet surface  

                const T thermal_conductivity = 0.4;                                                                                         // DUMMY_VAL mean thermal conductivity. Calc each iteration?
                const T specific_heat        = (0.363 + 0.000467 * temp) * (5-0.001 * density);                                             // DUMMY_VAL specific heat of the gas
                const T mass_transfer        = mass_fraction / (1 - mass_fraction);
                const T mass_delta           = 2 * M_PI * diameter * (thermal_conductivity / specific_heat) * log(1 + mass_transfer);       // Rate of fuel evaporation

                const T latent_heat       = 346.0 * pow((548. - temp) / (548. - 333.), 0.38);                                               // DUMMY_VAL Latent heat of fuel vaporization (kJ/kg)
                const T air_temp          = 1500.;                                                                                          // DUMMY_VAL Gas temperature?
                const T air_heat_transfer = 2 * M_PI * fuel_vapour_pressure * (air_temp - temp) * log(1 + mass_transfer) / mass_transfer;   // The heat transferred from air to fuel
                const T evaporation_heat  = mass_delta * latent_heat;                                                                       // The heat absorbed through evaporation
                const T temp_delta        = (air_heat_transfer - evaporation_heat) / (specific_heat * mass);                                // Temperature change of the droplet's surface

                const T evaporation_constant = 1e-6;  // Evaporation constant

                temp     = temp + temp_delta * delta; // Double check + or -
                mass     = mass + mass_delta * delta;
                diameter = sqrt(diameter * diameter  - evaporation_constant * delta);

                if (LOGGER)
                {
                    logger->flops  += 36;
                    logger->loads  += 3 * sizeof(T);     // 3 fields (air_temp, pressure, temp)
                    logger->stores += 3 * sizeof(T);     // 3 fields (temp, mass, diameter)
                }
                // SOLVE SPRAY BREAKUP MODEL 

            }

            
    }; // class Particle
 
}   // namespace minicombust::particles 