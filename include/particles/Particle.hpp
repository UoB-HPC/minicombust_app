#pragma once

#include <cmath>
#include <limits>
#include <iomanip>
#include <bitset>
#include <vector>

#include "utils/utils.hpp"
#include "geometry/Mesh.hpp"


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

            inline T get_uniform_random(T lower, T upper) 
            {
                T r = ((T) rand()) / RAND_MAX;
                return lower + (r * (upper - lower));
            }

            inline double tetrahedral_volume(vec<T> *A, vec<T> *B, vec<T> *C, vec<T> *D)
            {
                // TODO: Is determinent version faster?
                // Algorithm Ref: https://math.stackexchange.com/questions/1603651/volume-of-tetrahedron-using-cross-and-dot-product

                const vec<T> a = *B - *C;
                const vec<T> b = *D - *C;
                const vec<T> c = *A - *C;

                return abs(dot_product(cross_product(a, b), c)) / 6.0;

            }

            inline bool check_cell(uint64_t current_cell, Mesh<T> *mesh)
            {
                // TODO: Currently is solely a cube partial volume algorithm. Tetra partial volume algorithm;
                uint64_t *current_cell_points = mesh->cells + current_cell*mesh->cell_size;
                vec<T> box_size = mesh->points[current_cell_points[H_VERTEX]] - mesh->points[current_cell_points[A_VERTEX]];
                double total_volume  = abs(box_size.x*box_size.y*box_size.z);

                double partial_volumes = 0.0;
                for (uint64_t i=0; i < mesh->cell_size; i++)
                {
                    vec<T> partial_box_size = mesh->points[current_cell_points[i]] - x1;
                    partial_volumes += abs(partial_box_size.x*partial_box_size.y*partial_box_size.z);
                }

                // if (PARTICLE_DEBUG)  cout << "\t\tpartial_volumes - total  " << partial_volumes-total_volume << endl;
                if (abs(partial_volumes-total_volume) < 5.0e-10)  return true;
                return false;
                
            }

            



        public:
            vec<T> x0 = 0.0;             // coordinates at timestep beginning
            vec<T> x1 = 0.0;             // coordinates at next timestep
            vec<T> v1 = 0.0;             // velocity at next timestep
            vec<T> a1 = 0.0;             // acceleration at next timestep

            bool decayed = false;

            T mass        = 0.1;           // DUMMY_VAL Current mass (kg)
            T temp;                        // DUMMY_VAL Current surface temperature (Kelvin)
            T diameter;                    // DUMMY_VAL Relationship between mass and diameter? Droplet is assumed to be spherical.


            T age = 0.0;

            uint64_t cell = MESH_BOUNDARY;          // cell at timestep beginning





            Particle(Mesh<T> *mesh, vec<T> start, vec<T> velocity, vec<T> acceleration, T temp) : x0(start), x1(start), v1(velocity), a1(acceleration), temp(temp)
            { 
                for (uint64_t c = 0; c < mesh->mesh_size; c++)
                {
                    if (check_cell(c, mesh))  
                    {
                        cell = c;
                        break;
                    }
                }
                if (cell == MESH_BOUNDARY)
                {
                    decayed = true;
                }

                diameter = 2 * pow(0.75 * mass / ( M_PI * 724.), 1./3.);

                if (PARTICLE_DEBUG)  cout  << "\t\tParticle is starting in " << cell << ", x0: " << print_vec(x0) << " v1: " << print_vec(v1) <<  endl ;
            }

            Particle(Mesh<T> *mesh, vec<T> start, vec<T> finish, vec<T> velocity, vec<T> acceleration, T mass, T temp, T diameter, uint64_t cell) : 
                     x0(start), x1(finish), v1(velocity), a1(acceleration),
                     mass(mass), temp(temp), diameter(diameter), cell(cell)
            { }

            inline uint64_t update_cell(Mesh<T> *mesh, particle_logger *logger)
            {

                if ( check_cell(cell, mesh) )
                {
                    if (PARTICLE_DEBUG)  cout << "\t\tParticle is still in cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                    x0 = x1;
                    return cell;

                    if (LOGGER)
                    {
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

                        

                        for (uint64_t face = 0; face < 6; face++)
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
                            return cell;
                        }

                        // Is particle in this new cell?
                        if (check_cell(cell, mesh)) 
                        {
                            found_cell = true;
                            x0 = x1;
                            if (PARTICLE_DEBUG)  cout << "\t\tParticle has moved to cell " << cell << " " << ", x1: " << print_vec(x1) << endl ;
                            return cell;
                        }
                        else { // Particle isn't in immediate neighbour
                            face_mask    = face_mask | (IMPOSSIBLE << (intercepted_face_id ^ 1));  // E.G If we detect interception with front face, don't trigger back face interception next time. // TOD0: Needed now?
                            if (PARTICLE_DEBUG)  cout  << "\t\tParticle isn't in neighbour cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                        }
                    }
                }
                return cell;
            }

            inline void solve_spray(Mesh<T> *mesh, double delta, particle_logger *logger, vec<T> gas_vel, T gas_pressure, T gas_temperature, uint64_t *current_particle, Particle<T> *particles)
            {
                // Inputs from flow: relative_acc, kinematic viscoscity?, air_temp, air_pressure
                // Scenario constants: omega?, latent_heat, droplet_pressure?, evaporation_constant
                // Calculated outputs: acceleration, droplet surface temperature, droplet mass, droplet diameter
                // Calculated outputs for flow: evaporated mass?
                // if (decayed) return;

                // TODO Add better flop estimates for pow and ln. Also, can we get a fast approximation. Taylor series?

                // TODO: Remove DUMMY_VALs
                // SOLVE SPRAY/DRAG MODEL  https://www.sciencedirect.com/science/article/pii/S0021999121000826?via%3Dihub
                const vec<T> relative_drop_vel           = gas_vel - v1;                                         // DUMMY_VAL Relative velocity between droplet and the fluid
                const T relative_drop_vel_mag            = magnitude(relative_drop_vel);                         // DUMMY_VAL Relative acceleration between the gas and liquid phase.
                const vec<T> relative_drop_acc           = a1 ;                                                  // DUMMY_VAL Relative acceleration between droplet and the fluid


                const T gas_density  = 0.59;                                               // DUMMY VAL
                const T fuel_density =  724. * (1. - 1.8 * 0.000645 * (temp - 288.6) - 0.090 * pow(temp - 288.6, 2.) / pow(548. - 288.6, 2.));


                const T omega               = 1.;                                                                  // DUMMY_VAL What is this?
                const T kinematic_viscosity = 1.48e-5 * pow(gas_temperature, 1.5) / (gas_temperature + 110.4);    // DUMMY_VAL 
                const T reynolds            = gas_density * relative_drop_vel_mag * diameter / kinematic_viscosity;

                const T droplet_frontal_area  = M_PI * (diameter / 2.) * (diameter / 2.);

                // Drag coefficient
                const T drag_coefficient = ( reynolds <= 1000. ) ? 24 * (1. + 0.15 * pow(reynolds, 0.687))/reynolds : 0.424;

                // const vec<T> body_force    = Should we account for this?
                const vec<T> virtual_force = (-0.5 * gas_density * omega) * relative_drop_acc;
                const vec<T> drag_force    = (drag_coefficient * reynolds  * 0.5 * gas_density * relative_drop_vel_mag *  droplet_frontal_area) * relative_drop_vel;
                
                

                // cout << "vf " << print_vec(virtual_force) << " df " << print_vec(drag_force) << " m " << mass << endl;
                a1 = ((virtual_force + drag_force) / mass) * delta;
                v1 = v1 + a1 * delta;
                x1 = x0 + v1 * delta;


                // SOLVE EVAPORATION MODEL https://arc.aiaa.org/doi/pdf/10.2514/3.8264 
                // Amount of spray evaporation is used in the modified transport equation of mixture fraction (each timestep).
                const T air_pressure           = gas_pressure;
                const T boiling_temp           = 333.;
                const T critical_temp          = 548.;
                const T a_constant             = (temp < boiling_temp) ? 13.7600 : 14.1964;
                const T b_constant             = (temp < boiling_temp) ? 2651.13 : 2777.65;
                const T fuel_vapour_pressure   = exp(a_constant - b_constant / (temp - 43));                         // DUMMY_VAL fuel vapor at drop surface (kP)
                const T pressure_relation      = (air_pressure + fuel_vapour_pressure) / fuel_vapour_pressure;       // DUMMY_VAL Clausius-Clapeyron relation. air pressure / fuel vapour pressure.
                const T molecular_ratio        = 29. / 108.;                                                         // DUMMY_VAL molecular weight air / molecular weight fuel
                const T mass_fraction_fuel     = 1 / (1 + (pressure_relation - 1) * molecular_ratio);                // Mass fraction of fuel vapour at the droplet surface  
                const T mass_fraction_fuel_ref = (2./3.) * mass_fraction_fuel;                                       // Mass fraction of fuel vapour ref at the droplet surface  
                const T mass_fraction_air_ref  = 1 - mass_fraction_fuel_ref;                                         // Mass fraction of air vapour  ref at the droplet surface  

                const T thermal_conduct_air      = 0.04418;                                                                                                                        // DUMMY_VAL mean thermal conduct. Calc each iteration?
                const T thermal_conduct_fuel     = 1.e-6*(13.2 - 0.0313 * (boiling_temp - 273)) * pow(temp / 273, 2 - 0.0372 * ((temp * temp) / (boiling_temp * boiling_temp)));   // DUMMY_VAL mean thermal conductivity. Calc each iteration?
                const T thermal_conductivity     = mass_fraction_air_ref * thermal_conduct_air + mass_fraction_fuel_ref * thermal_conduct_fuel;                                    // DUMMY_VAL specific heat of the gas


                const T specific_heat_fuel       = (0.363 + 0.000467 * temp) * (5 - 0.001 * fuel_density);                                      // DUMMY_VAL specific heat of the gas
                const T specific_heat_air        = 1044;                                                                                        // DUMMY_VAL specific heat of the gas
                const T specific_heat            = mass_fraction_air_ref * specific_heat_air + mass_fraction_fuel_ref * specific_heat_fuel;     // DUMMY_VAL specific heat of the gas


                const T mass_transfer        = mass_fraction_fuel / (1 - mass_fraction_fuel);
                const T mass_delta           = 2 * M_PI * diameter * (thermal_conductivity / specific_heat_fuel) * log(1 + mass_transfer);       // Rate of fuel evaporation

                
                const T latent_heat       = 346.0 * pow((critical_temp - temp) / (critical_temp - boiling_temp), 0.38);                     // DUMMY_VAL Latent heat of fuel vaporization (kJ/kg)
                const T air_temp          = gas_temperature;                                                                                // DUMMY_VAL Gas temperature?
                const T air_heat_transfer = 2 * M_PI * fuel_vapour_pressure * (air_temp - temp) * log(1 + mass_transfer) / mass_transfer;   // The heat transferred from air to fuel
                const T evaporation_heat  = mass_delta * latent_heat;                                                                       // The heat absorbed through evaporation
                const T temp_delta        = (air_heat_transfer - evaporation_heat) / (specific_heat * mass);                                // Temperature change of the droplet's surface

                const T evaporation_constant = 8 * log(1 + mass_transfer) * thermal_conductivity / (fuel_density * specific_heat_fuel);     // Evaporation constant

                temp     = temp + temp_delta * delta;
                mass     = mass - mass_delta * delta;
                diameter = sqrt(diameter * diameter  - evaporation_constant * delta);


                mesh->evaporated_fuel_mass_rate[cell] += mass_delta * delta;                             // Do we need to worry about the if the particle is vaporized
                mesh->particle_energy_rate[cell]      += (air_heat_transfer - evaporation_heat)*delta;
                mesh->particle_momentum_rate[cell]    += mass * v1 * delta;

                if (mass < 0 || temp > critical_temp) 
                {
                    decayed = true;
                    if (LOGGER)
                    {   
                        logger->decayed_particles++;
                        logger->burnt_particles++;
                    }
                    return;
                }

                // cout << endl << "Particle Drag Effects" << endl;
                // cout << "\ta                     "     << print_vec(a1)     << endl;
                // cout << "\tvel1                  "     << print_vec(v1)     << endl;
                // cout << "\tcell                  "     << cell     << endl;
                // cout << "\tgas_vel               "     << print_vec(gas_vel)     << endl;
                // cout << "\trelative_drop_acc     "     << print_vec(relative_drop_acc)     << endl;
                // cout << "\trelative_drop_vel     "     << print_vec(relative_drop_vel)     << endl;
                // cout << "\trelative_drop_vel_mag "  << relative_drop_vel_mag << endl;
                // cout << "\tdroplet_frontal_area  "  << droplet_frontal_area << endl;
                // cout << "\treynolds              "  << reynolds << endl;
                // cout << "\tdrag_coefficient      "  << drag_coefficient << endl;
                // cout << "\tkinematic_viscosity   "  << kinematic_viscosity << endl;
                // cout << "\tmass                  "  << mass << endl;
                // cout << "\tdiameter              "  << diameter << endl;
                // cout << "\tvirtual_force         "  << print_vec(virtual_force) << endl;
                // cout << "\tdrag_force            "  << print_vec(drag_force) << endl;
                // cout << "\tnet force             "  << print_vec(drag_force + virtual_force) << endl;
                // cout << "\tacc_delta             "  << print_vec(((virtual_force + drag_force) / mass) * delta) << endl;
                // cout << "\tvel_delta             "  << print_vec(a1 * delta) << endl;
                // cout << "\tpos_delta             "  << print_vec(a1 * delta * delta) << endl;
                // cout << "\tposition              "  << print_vec(x1) << endl << endl ;


                // cout << "\ttemp                  "  << temp << endl;
                // cout << "\tgas_temperature       "  << gas_temperature << endl;
                // cout << "\tfuel_vapour_pressure  "  << fuel_vapour_pressure << endl;
                // cout << "\tair_pressure          "  << air_pressure << endl;
                // cout << "\tpressure_relation     "  << pressure_relation << endl;
                // cout << "\tmass_fraction_fuel    "  << mass_fraction_fuel << endl;
                // cout << "\tmass_fraction_air_ref "  << mass_fraction_air_ref << endl;
                // cout << "\tmass_fraction_fuel_ref"  << mass_fraction_fuel_ref << endl;
                // cout << "\tthermal_conduct_fuel  "  << thermal_conduct_fuel << endl;
                // cout << "\tthermal_conductivity  "  << thermal_conductivity << endl;
                // cout << "\tfuel_density          "  << fuel_density << endl;
                // cout << "\tspecific_heat_fuel    "  << specific_heat_fuel << endl;
                // cout << "\tspecific_heat_air     "  << specific_heat_air << endl;
                // cout << "\tspecific_heat         "  << specific_heat << endl;
                // cout << "\tmass_transfer         "  << mass_transfer << endl;
                // cout << "\tlatent_heat           "  << latent_heat << endl;
                // cout << "\tair_heat_transfer     "  << air_heat_transfer << endl;
                // cout << "\tevaporation_heat      "  << evaporation_heat << endl;
                // cout << "\tevaporation_constant  "  << evaporation_constant << endl;
                // cout << "\ttemp_delta            "  << temp_delta << endl;
                // cout << "\tmass_delta            "  << mass_delta << endl << endl;
                // cout << "\tdecayed               "  << decayed << endl << endl;

                


                if (*current_particle < (mesh->max_cell_particles * mesh->mesh_size))
                {

                    // SOLVE SPRAY BREAKUP MODEL
                    age += delta;

                    const T breakup_age   = sqrt(fuel_density / (3*gas_density)) * (diameter / (2.0*relative_drop_vel_mag));

                    const T surface_tension  = fuel_vapour_pressure * diameter / 4;
                    const T weber_droplet    = fuel_density * (relative_drop_vel_mag * relative_drop_vel_mag) * diameter / surface_tension;
                    const T weber_critical   = 0.5;

                    if (age > breakup_age && weber_droplet > weber_critical)  // Ternary?
                    {
                        // const T first_moment   = 0.6 * log(weber_critical / weber_droplet); 
                        // const T second_moment  = - first_moment * weber_droplet; 
                        
                        // TODO: How do you get a random number from distribution 0.5 * (1 + erf((x - diameter - first_moment) / sqrt(2*second_moment))); 
                        const T rand_prop = get_uniform_random(0.3, 0.7);
                        const T diameter1 = rand_prop * diameter;
                        const T diameter2 = diameter - diameter1;

                        const T droplet1_ratio = diameter1 / diameter;

                        const T mass1 = droplet1_ratio * mass;
                        const T mass2 = mass - mass1;

                        // Product droplet velocity is computed by adding a factor to the parent velocity
                        const T magnitude = diameter / (2.0 * breakup_age);
                        
                        // Direction is a random unit vector normal to the relative velocity
                        // Randomly seed x component of velocities, with knowledge that direction magnitude = 1 
                        vec<T> velocity1, velocity2;
                        velocity1.x = get_uniform_random(-1.0, 1.0);
                        velocity2.x = get_uniform_random(-1.0, 1.0);

                        // Random seed y component of velocities, with knowledge that direction magnitude = 1
                        const T remaining_mag1 = sqrt(1 - (velocity1.x * velocity1.x));
                        const T remaining_mag2 = sqrt(1 - (velocity2.x * velocity2.x));
                        velocity1.y = get_uniform_random(-remaining_mag1, remaining_mag1);
                        velocity2.y = get_uniform_random(-remaining_mag2, remaining_mag2);

                        // Dot product = 0 for perpendicular vectors, solve for direction z components
                        velocity1.z = - (relative_drop_vel.x * velocity1.x + relative_drop_vel.y * velocity1.y) / relative_drop_vel.z;
                        velocity2.z = - (relative_drop_vel.x * velocity2.x + relative_drop_vel.y * velocity2.y) / relative_drop_vel.z;
 
                        
                        particles[*current_particle] = Particle<T>(mesh, x0, x1, velocity2 * magnitude + v1, a1, mass2, temp, diameter2, cell);
                        (*current_particle)++;

                        // Update parent to droplet1;
                        v1   += velocity1 * magnitude;
                        mass = mass1;
                        age  = 0.0;

                        if (LOGGER)
                        {   
                            logger->breakups++;
                            logger->num_particles++;     
                        }
                    }

                }
                else if (LOGGER)
                {
                    logger->unsplit_particles++;
                }
            }

            
    }; // class Particle
 
}   // namespace minicombust::particles 
