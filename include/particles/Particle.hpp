#pragma once

#include <cmath>
#include <limits>
#include <iomanip>
#include <bitset>

#include "utils/utils.hpp"
#include "geometry/Mesh.hpp"




namespace minicombust::particles 
{
    using namespace std; 
    using namespace minicombust::utils;
    using namespace minicombust::geometry; 

    #define INVALID_FACE 1;
    #define VALID_FACE 0;

    const double EPSILON = 5.0e-17;

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
                vec<T> box_size = mesh->points[mesh->cells[(current_cell - mesh->shmem_cell_disp)*mesh->cell_size + H_VERTEX] - mesh->shmem_point_disp] - mesh->points[mesh->cells[(current_cell - mesh->shmem_cell_disp)*mesh->cell_size + A_VERTEX] - mesh->shmem_point_disp];
                double total_volume  = abs(box_size.x * box_size.y * box_size.z);

                double partial_volumes = 0.0;
                #pragma ivdep
                for (uint64_t i=0; i < mesh->cell_size; i++)
                {
                    vec<T> partial_box_size = mesh->points[mesh->cells[(current_cell - mesh->shmem_cell_disp)*mesh->cell_size + i] - mesh->shmem_point_disp] - x1;
                    partial_volumes += abs(partial_box_size.x * partial_box_size.y * partial_box_size.z);
                }

                // if (PARTICLE_DEBUG)  cout << "\t\tpartial_volumes - total  " << partial_volumes-total_volume << endl;
                if (abs(partial_volumes-total_volume) < EPSILON)  return true;
                return false;
                
            }


        public:
            vec<T> x1 = 0.0;             // coordinates at next timestep
            vec<T> v1 = 0.0;             // velocity at next timestep
            vec<T> a1 = 0.0;             // acceleration at next timestep

            bool decayed = false;

            T mass        = 0.02;          // DUMMY_VAL Current mass (kg)
            T temp;                        // DUMMY_VAL Current surface temperature (Kelvin)
            T diameter;                    // DUMMY_VAL Relationship between mass and diameter? Droplet is assumed to be spherical.

            flow_aos<T> local_flow_value = {{0.0, 0.0, 0.0}, 0.0, 0.0};

            particle_aos<T> particle_cell_fields;


            T age = 0.0;

            uint64_t cell;          // cell at timestep beginning


            Particle(Mesh<T> *mesh, vec<T> start, vec<T> velocity, vec<T> acceleration, T temp, uint64_t cell, Particle_Logger *logger) : x1(start), v1(velocity), a1(acceleration), temp(temp), cell(cell)
            { 
                update_cell(mesh, logger);

                diameter = 2 * pow(0.75 * mass / ( M_PI * 724.), 1./3.);

                if (PARTICLE_DEBUG)  cout  << "\t\tParticle is starting in " << cell << ", x1: " << print_vec(x1) << " v1: " << print_vec(v1) <<  endl ;
            }

            Particle(vec<T> position, vec<T> velocity, vec<T> acceleration, T mass, T temp, T diameter, uint64_t cell) : 
                     x1(position), v1(velocity), a1(acceleration),
                     mass(mass), temp(temp), diameter(diameter), cell(cell)
            { }

            inline uint64_t update_cell(Mesh<T> *mesh, Particle_Logger *logger)
            {


                if ( check_cell(cell, mesh) )
                {
                    if (PARTICLE_DEBUG)  cout << "\t\tParticle is still in cell " << cell << ", x1: " << print_vec(x1) <<  endl ;

                    if (LOGGER)  logger->cell_checks++;

                    return cell;
                } 
                else
                {
                    bool found_cell     = false;
                    vec<T> artificial_A = mesh->cell_centers[cell - mesh->shmem_cell_disp];
                    vec<T> *A = &artificial_A;
                    vec<T> *B = &x1;

                    uint64_t num_tries = 0;
                    
                    uint64_t face_mask = POSSIBLE; // Prevents double interceptions     

                    vec<T> z_vec = mesh->points[mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + E_VERTEX] - mesh->shmem_point_disp] - mesh->points[mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + A_VERTEX] - mesh->shmem_point_disp];
                    vec<T> x_vec = mesh->points[mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + B_VERTEX] - mesh->shmem_point_disp] - mesh->points[mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + A_VERTEX] - mesh->shmem_point_disp];
                    vec<T> y_vec = mesh->points[mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + C_VERTEX] - mesh->shmem_point_disp] - mesh->points[mesh->cells[(cell - mesh->shmem_cell_disp)*mesh->cell_size + A_VERTEX] - mesh->shmem_point_disp];

                    vec<T> x_delta   = x1 - artificial_A;
                    
                    // Detects which cells you could possibly hit. Should at least halve the number of face checks.
                    T dot = dot_product(x_delta, z_vec); 
                    face_mask = face_mask | (((dot == 0) ? PARALLEL : IMPOSSIBLE) << ((dot >= 0) ? 0 : 1));
                    dot = dot_product(x_delta, x_vec); 
                    face_mask = face_mask | (((dot == 0) ? PARALLEL : IMPOSSIBLE) << ((dot >= 0) ? 2 : 3)); 
                    dot = dot_product(x_delta, y_vec); 
                    face_mask = face_mask | (((dot == 0) ? PARALLEL : IMPOSSIBLE) << ((dot >= 0) ? 4 : 5)); 

                    while (!found_cell)
                    {
                        uint64_t intercepted_face_id = -1;
                        uint64_t intercepted_faces   = 0;
 
                        for (uint64_t face = 0; face < 6; face++)
                        {
                            if ((1 << face) & face_mask)  continue;

                            if (PARTICLE_DEBUG) cout << "\t\t\tTrying face " << mesh->get_face_string(face) << " " << bitset<6>(face_mask) << endl; 
                            if (PARTICLE_DEBUG) cout << "\t\t\tMoving from cell " << cell << " with pos " << print_vec(*A) << " to " << print_vec (*B) << endl; 

                            // Check whether particle - moving from A to B - intercepts face CDEF. If LHS == RHS, particle intercepts this face.
                            vec<T> *C = &mesh->points[mesh->cells[ (cell - mesh->shmem_cell_disp) * mesh->cell_size + CUBE_FACE_VERTEX_MAP[face][0]] - mesh->shmem_point_disp];
                            vec<T> *D = &mesh->points[mesh->cells[ (cell - mesh->shmem_cell_disp) * mesh->cell_size + CUBE_FACE_VERTEX_MAP[face][1]] - mesh->shmem_point_disp];
                            vec<T> *E = &mesh->points[mesh->cells[ (cell - mesh->shmem_cell_disp) * mesh->cell_size + CUBE_FACE_VERTEX_MAP[face][2]] - mesh->shmem_point_disp];
                            vec<T> *F = &mesh->points[mesh->cells[ (cell - mesh->shmem_cell_disp) * mesh->cell_size + CUBE_FACE_VERTEX_MAP[face][3]] - mesh->shmem_point_disp];

                            const double LHS = tetrahedral_volume(A, C, D, B)
                                             + tetrahedral_volume(A, D, F, B)
                                             + tetrahedral_volume(A, F, E, B)
                                             + tetrahedral_volume(A, E, C, B);

                            const double RHS = tetrahedral_volume(A, C, D, F)
                                             + tetrahedral_volume(A, E, C, F)
                                             + tetrahedral_volume(B, C, D, F)
                                             + tetrahedral_volume(B, E, C, F);

                            if (PARTICLE_DEBUG)  printf("\t\t\t%.20f == %.20f\n", LHS, RHS);
                            if ( (abs(LHS - RHS) < EPSILON) && (LHS != 0.) )  
                            {
                                intercepted_face_id = face;
                                intercepted_faces++;
                                
                                if (PARTICLE_DEBUG)  cout << "\t\t\tIntercepted Face " << mesh->get_face_string(face) << " num faces =  " << intercepted_faces << endl;
                            }
                        }

                        if (LOGGER)
                        {
                            logger->cell_checks++;
                        }

                        // Check if multiple faces have been intercepted
                        if (intercepted_faces != 1)
                        {
                            // Get random point within unit sphere
                            vec<T> r = vec<T> { static_cast<double>(rand())/(RAND_MAX), static_cast<double>(rand())/(RAND_MAX), static_cast<double>(rand())/(RAND_MAX) } ;
                            r        = - 1. + (r * 2.);
                            artificial_A = mesh->cell_centers[cell - mesh->shmem_cell_disp] + 0.4 * r * mesh->cell_size_vector / 2.;

                            if (PARTICLE_DEBUG)  cout << "\t\t\tMultiple faces, artificial position " <<  print_vec(artificial_A) << endl;
                            if (LOGGER)
                            {
                                logger->position_adjustments++;
                            }

                            if (num_tries++ > 4)
                            {
                                decayed = true;
                                cell    = MESH_BOUNDARY;
                                if (PARTICLE_DEBUG)  cout << "\t\t\tLost Particle " << endl;
                                if (LOGGER) logger->lost_particles++;
                                if (LOGGER) logger->decayed_particles++;
                                return cell;
                            }
                            continue;
                        }

                        num_tries = 0;

                        // Test neighbour cell
                        cell = mesh->cell_neighbours[(cell - mesh->shmem_cell_disp)*mesh->faces_per_cell + intercepted_face_id];
                        artificial_A = mesh->cell_centers[cell - mesh->shmem_cell_disp];

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

            inline void solve_spray(double delta, Particle_Logger *logger, vector<Particle<T>>& particles)
            {
                // Inputs from flow: relative_acc, kinematic viscoscity?, air_temp, air_pressure
                // Scenario constants: omega?, latent_heat, droplet_pressure?, evaporation_constant
                // Calculated outputs: acceleration, droplet surface temperature, droplet mass, droplet diameter
                // Calculated outputs for flow: evaporated mass?
                // if (decayed) return;

                // TODO Add better flop estimates for pow and ln. Also, can we get a fast approximation. Taylor series?

                // TODO: Remove DUMMY_VALs
                // SOLVE SPRAY/DRAG MODEL  https://www.sciencedirect.com/science/article/pii/S0021999121000826?via%3Dihub7
                const vec<T> relative_drop_vel           = 0.65 * (local_flow_value.vel - v1);                                         // DUMMY_VAL Relative velocity between droplet and the fluid 
                const T relative_drop_vel_mag            = magnitude(relative_drop_vel);                         // DUMMY_VAL Relative acceleration between the gas and liquid phase.
                const vec<T> relative_drop_acc           = a1 * delta ;                                                  // DUMMY_VAL Relative acceleration between droplet and the fluid CURRENTLY assumes no change for gas temp

                // cout << "local_flow_value.vel " << print_vec(local_flow_value.vel) << "v1 " << print_vec(v1) << " relative_drop_vel_mag " << relative_drop_vel_mag << " m " << mass << endl;

                const T gas_density  = 6.9;                                               // DUMMY VAL
                const T fuel_density = 724. * (1. - 1.8 * 0.000645 * (temp - 288.6) - 0.090 * ((temp - 288.6) * (temp - 288.6)) / 67288.36);


                const T omega               = 1.;                                                                  // DUMMY_VAL What is this?
                const T kinematic_viscosity = 1.48e-5 * pow(local_flow_value.temp, 1.5) / (local_flow_value.temp + 110.4);    // DUMMY_VAL 
                const T reynolds            = gas_density * relative_drop_vel_mag * diameter / kinematic_viscosity;

                const T droplet_frontal_area  = M_PI * (diameter / 2.) * (diameter / 2.);

                // Drag coefficient
                const T drag_coefficient = ( reynolds <= 1000. ) ? 24 * (1. + 0.15 * pow(reynolds, 0.687))/reynolds : 0.424;

                // const vec<T> body_force    = Should we account for this?
                const vec<T> virtual_force = (-0.5 * gas_density * omega) * relative_drop_acc;
                const vec<T> drag_force    = (drag_coefficient * reynolds  * 0.5 * gas_density * relative_drop_vel_mag *  droplet_frontal_area) * relative_drop_vel;
                
                

                // cout << "vf " << print_vec(virtual_force) << " df " << print_vec(drag_force) << " m " << mass << endl;
                a1 = ((virtual_force + drag_force) / mass);
                v1 = v1 + a1 * delta;
                


                // SOLVE EVAPORATION MODEL https://arc.aiaa.org/doi/pdf/10.2514/3.8264 
                // Amount of spray evaporation is used in the modified transport equation of mixture fraction (each timestep).
                const T air_pressure           = local_flow_value.pressure;
                const T boiling_temp           = 333.;
                const T critical_temp          = 548.;
                const T a_constant             = (temp < boiling_temp) ? 13.7600 : 14.1964;
                const T b_constant             = (temp < boiling_temp) ? 2651.13 : 2777.65;
                const T fuel_vapour_pressure   = exp(a_constant - b_constant / (temp - 43.));                         // DUMMY_VAL fuel vapor at drop surface (kP)
                const T pressure_relation      = (air_pressure + fuel_vapour_pressure) / fuel_vapour_pressure;       // DUMMY_VAL Clausius-Clapeyron relation. air pressure / fuel vapour pressure.
                const T molecular_ratio        = 29. / 108.;                                                         // DUMMY_VAL molecular weight air / molecular weight fuel
                const T mass_fraction_fuel     = 1. / (1. + (pressure_relation - 1.) * molecular_ratio);                // Mass fraction of fuel vapour at the droplet surface  
                const T mass_fraction_fuel_ref = (2./3.) * mass_fraction_fuel;                                       // Mass fraction of fuel vapour ref at the droplet surface  
                const T mass_fraction_air_ref  = 1. - mass_fraction_fuel_ref;                                         // Mass fraction of air vapour  ref at the droplet surface  

                const T thermal_conduct_air      = 0.04418;                                                                                                                        // DUMMY_VAL mean thermal conduct. Calc each iteration?
                const T thermal_conduct_fuel     = 1.e-6*(13.2 - 0.0313 * (boiling_temp - 273.)) * pow(temp / 273., 2. - 0.0372 * ((temp * temp) / (boiling_temp * boiling_temp)));   // DUMMY_VAL mean thermal conductivity. Calc each iteration?
                const T thermal_conductivity     = mass_fraction_air_ref * thermal_conduct_air + mass_fraction_fuel_ref * thermal_conduct_fuel;                                    // DUMMY_VAL specific heat of the gas


                const T specific_heat_fuel       = (0.363 + 0.000467 * temp) * (5. - 0.001 * fuel_density);                                      // DUMMY_VAL specific heat of the gas
                const T specific_heat_air        = 1044.;                                                                                        // DUMMY_VAL specific heat of the gas
                const T specific_heat            = mass_fraction_air_ref * specific_heat_air + mass_fraction_fuel_ref * specific_heat_fuel;     // DUMMY_VAL specific heat of the gas


                const T mass_transfer        = mass_fraction_fuel / (1. - mass_fraction_fuel);
                const T log_mass_transfer    = log(1. + mass_transfer);
                const T mass_delta           = 2. * M_PI * diameter * (thermal_conductivity / specific_heat_fuel) * log_mass_transfer;       // Rate of fuel evaporation

                
                const T latent_heat       = 346.0 * pow((critical_temp - temp) / (critical_temp - boiling_temp), 0.38);                     // DUMMY_VAL Latent heat of fuel vaporization (kJ/kg)
                const T air_heat_transfer = 2. * M_PI * fuel_vapour_pressure * (local_flow_value.temp - temp) * log_mass_transfer / mass_transfer;   // The heat transferred from air to fuel
                const T evaporation_heat  = mass_delta * latent_heat;                                                                       // The heat absorbed through evaporation
                const T temp_delta        = (air_heat_transfer - evaporation_heat) / (specific_heat * mass);                                // Temperature change of the droplet's surface

                const T evaporation_constant = 8. * log_mass_transfer * thermal_conductivity / (fuel_density * specific_heat_fuel);     // Evaporation constant

                temp     = temp + temp_delta * delta;
                mass     = mass - mass_delta * delta;
                diameter = sqrt(diameter * diameter  - evaporation_constant * delta);

                // Store particle fields
                particle_cell_fields = {mass * v1 * delta, (air_heat_transfer - evaporation_heat) * delta, mass_delta * delta};



                decayed = (mass < 0 || temp > critical_temp);


                // cout << endl << "Particle Drag Effects" << endl;
                // cout << "\ta                     "     << print_vec(a1)     << endl;
                // cout << "\tvel1                  "     << print_vec(v1)     << endl;
                // cout << "\tcell                  "     << cell     << endl;
                // cout << "\tlocal_flow_value.vel               "     << print_vec(local_flow_value.vel)     << endl;
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
                // cout << "\tlocal_flow_value.temp       "  << local_flow_value.temp << endl;
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
                // cout << "\tbreakup_age            "  << breakup_age << endl << endl;
                
                // cout << "\tdecayed               "  << decayed << endl << endl;

                


                if (!decayed)
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
                        const T rand_prop = get_uniform_random(0.1, 0.9);
                        const T diameter1 = rand_prop * diameter;
                        const T diameter2 = diameter - diameter1;

                        const T droplet1_ratio = diameter1 / diameter;

                        const T mass1 = droplet1_ratio * mass;
                        const T mass2 = mass - mass1;

                        // Product droplet velocity is computed by adding a factor to the parent velocity
                        const T length = diameter / (2.0 * breakup_age);

                        vec<T> velocity1 =  { static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX } ;
                        vec<T> velocity2 =  { static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX, static_cast<double>(rand())/RAND_MAX } ;

                        velocity1 = -1. + (velocity1 * 2.);
                        velocity2 = -1. + (velocity2 * 2.);

                        vec<T> unit_rel_velocity = relative_drop_vel / magnitude(relative_drop_vel);

                        velocity1 = velocity1 - dot_product(velocity1, unit_rel_velocity) * unit_rel_velocity; 
                        velocity2 = velocity2 - dot_product(velocity2, unit_rel_velocity) * unit_rel_velocity; 

                        
                        particles.push_back(Particle<T>(x1 + (velocity2 * length + v1 * delta), velocity2 * length + v1, a1, mass2, temp, diameter2, cell));

                        // Update parent to droplet1;
                        v1  += velocity1 * length;
                        mass = mass1;
                        age  = 0.0;

                        if (LOGGER)
                        {   
                            logger->breakups++;
                            logger->num_particles++;
                            logger->breakup_age = breakup_age;
                        }
                    }

                    if (LOGGER)
                    {   
                        logger->breakup_age = breakup_age;
                    }
                } 
                else if (LOGGER)
                {
                    logger->decayed_particles++;
                    logger->burnt_particles++;
                }


                x1 = x1 + v1 * delta;
            }

    }; // class Particle
 
}   // namespace minicombust::particles 