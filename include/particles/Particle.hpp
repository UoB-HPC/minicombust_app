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

            }

            void update_cell(Mesh<T> *mesh, particle_logger *logger)
            {

                if ( check_cell(cell, mesh) )
                {
                    if (PARTICLE_DEBUG)  cout << "\t\tParticle is still in cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                    x0 = x1;
                    v0 = v1;
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

                        logger->cell_checks++;

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
                        
                        if (intercepted_faces > 1)
                        {
                            vec<T> r = vec<T> { static_cast<double>(rand())/(RAND_MAX), static_cast<double>(rand())/(RAND_MAX), static_cast<double>(rand())/(RAND_MAX) } ;
                            artificial_A = mesh->cell_centres[cell] + 0.1 * r*mesh->cell_size_vector;
                            if (PARTICLE_DEBUG)  cout << "\t\t\tNo faces, artificial position " <<  print_vec(artificial_A) << endl;
                            logger->position_adjustments++;
                            continue;
                        }

                        cell = mesh->cell_neighbours[cell*mesh->faces_per_cell + intercepted_face_id];
                        if (PARTICLE_DEBUG)  cout << "\t\tMoving to cell " << cell << " " << mesh->get_face_string(intercepted_face_id) << " direction" << endl;  

                        // If intercepted with boundary of mesh, decay particle. TODO: Rebound them?
                        if ( cell == MESH_BOUNDARY ) 
                        {
                            if (PARTICLE_DEBUG)  cout << "\t\tParticle decayed after leaving the grid, x1: " << print_vec(x1) << " " << cell <<  endl ;
                            decayed = true;
                            logger->boundary_intersections++;
                            logger->decayed_particles++;
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

            T dens0 = -1.0;       
            T diam0 =  1.0;
            T mass0 =  1.0;

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
                x1 = x0 + v1*delta;
                v1 = v0 + a1*delta;
                // if (PARTICLE_DEBUG)  cout << "End of timestep: x1: " << print_vec(x1) << " v1 " << print_vec(v1) << endl; 
               
                // Check if particle is in the current cell. Tetras = Volume/Area comparison method. https://www.peertechzpublications.com/articles/TCSIT-6-132.php.
                update_cell(mesh, logger);

                return cell; 
            } 

            
    }; // class Particle
 
}   // namespace minicombust::particles 