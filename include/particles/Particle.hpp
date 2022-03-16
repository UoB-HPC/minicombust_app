#pragma once

#include <cmath>
#include <limits>
#include <iomanip>

// #include "utils/utils.hpp"
#include "geometry/Mesh.hpp"


#define PARTICLE_DEBUG 1

namespace minicombust::particles 
{
    using namespace std; 
    using namespace minicombust::utils;
    using namespace minicombust::geometry; 

    
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
                
                if (abs(partial_volumes-total_volume) < 5.0e-10)  return true;
                return false;

            }

            void update_cell(Mesh<T> *mesh)
            {
                if ( check_cell(cell, mesh) )
                {
                    if (PARTICLE_DEBUG)  cout << "\t\tParticle is still in cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                } 
                else
                {
                    bool found_cell = false;
                    vec<T> *A = &x0;
                    vec<T> *B = &x1;
                    
                    uint64_t face_mask = 0; // Prevents double interceptions     

                    // TODO: Fix bug where parallelepiped is formed and breaks the intersection logic
                    int parallelepiped = 0;
                    vec<T> x_delta   = x1 - x0;
                    vec<T> cell_size = mesh->points[mesh->cells[cell*mesh->cell_size + H_VERTEX]] - mesh->points[mesh->cells[cell*mesh->cell_size + A_VERTEX]];
                    parallelepiped += (abs(x_delta.x) == abs(cell_size.x)) ? 1 : 0;
                    parallelepiped += (abs(x_delta.y) == abs(cell_size.y)) ? 1 : 0;
                    parallelepiped += (abs(x_delta.z) == abs(cell_size.z)) ? 1 : 0;
                    if (parallelepiped > 1)  {
                        x1 += 1e-8*x_delta;
                        // x1 += numeric_limits<T>::min()*x_delta;
                        if (PARTICLE_DEBUG)  cout << "\t\tParticle parallelepiped detected." << endl;
                    }
                    while (!found_cell)
                    {
                        uint64_t *current_cell_points = mesh->cells + cell*mesh->cell_size;       
                        

                        int intercepted_face = -1;
                        // TODO Check if particle goes through vertex.
                        int edge_vertex = 0;
                        double ACDB, ADFB, AFEB, AECB; // LHS partial volumes. Also used for edge/vertex detection.

                        double LHS; // LHS =  ACDB + ADFB + AFEB + AECB 
                        double RHS; // RHS =  ACDF + BCDF + AECF + BECF   
                        for (int face = 0; face < 6; face++)
                        {
                            if ((1 << face) & face_mask)  continue;

                            // cout << "\t\t\tTrying face " << mesh->get_face_string(face) << " " << face_mask << endl; 

                            // Check whether particle - moving from A to B - intercepts face CDEF. If LHS == RHS, particle intercepts this face.
                            LHS = 0.0;
                            RHS = 0.0;

                            vec<T> *C = &mesh->points[current_cell_points[CUBE_FACE_VERTEX_MAP[face][0]]];
                            vec<T> *D = &mesh->points[current_cell_points[CUBE_FACE_VERTEX_MAP[face][1]]];
                            vec<T> *E = &mesh->points[current_cell_points[CUBE_FACE_VERTEX_MAP[face][2]]];
                            vec<T> *F = &mesh->points[current_cell_points[CUBE_FACE_VERTEX_MAP[face][3]]];

                            // cout << print_vec(*C) << " | " << print_vec(*D) << " | " << print_vec(*E) << " | " << print_vec(*F) << endl;
                            // cout << print_vec(*A) << " | " << print_vec(*B) << " | " << endl;

                            ACDB = tetrahedral_volume(A, C, D, B);
                            LHS += ACDB;
                            ADFB = tetrahedral_volume(A, D, F, B);
                            LHS += ADFB;
                            AFEB = tetrahedral_volume(A, F, E, B);
                            LHS += AFEB;
                            AECB = tetrahedral_volume(A, E, C, B);
                            LHS += AECB;
                            // cout << ACDB << " " << ADFB << " " << AFEB << " " << AECB << endl;
                            // LHS += tetrahedral_volume(A, C, D, B);
                            // LHS += tetrahedral_volume(A, D, F, B);
                            // LHS += tetrahedral_volume(A, F, E, B);
                            // LHS += tetrahedral_volume(A, E, C, B);

                            RHS += tetrahedral_volume(A, C, D, F);
                            RHS += tetrahedral_volume(A, E, C, F);
                            RHS += tetrahedral_volume(B, C, D, F);
                            RHS += tetrahedral_volume(B, E, C, F);

                                // printf("\t%.17f %.17f\n", LHS, RHS);

                            if (abs(LHS - RHS) < 5e-12)  
                            {
                                intercepted_face = face;
                                if (PARTICLE_DEBUG)  cout << "\t\t\tIntercepted Face " << mesh->get_face_string(intercepted_face) << endl;
                                // printf("\t\t\t%.17f == %.17f\n", LHS, RHS);
                                // cout << print_vec(*C) << "  |  " << print_vec(*D) << "  |  "<< print_vec(*E) << "  |  " << print_vec(*F) << "  |  " << endl;
                                edge_vertex = 0;
                                edge_vertex += (ACDB == 0.) ? 1 : 0;
                                edge_vertex += (ADFB == 0.) ? 1 : 0;
                                edge_vertex += (AFEB == 0.) ? 1 : 0;
                                edge_vertex += (AECB == 0.) ? 1 : 0;


                                break;
                            }
                        }

                        // Code for detecting edges/vertexes. Don't think this will be faster. Probably required for tetra implementation
                        // edge_vertex += (ACDB == 0.) ? 1 : 0;
                        // edge_vertex += (ADFB == 0.) ? 1 : 0;
                        // edge_vertex += (AFEB == 0.) ? 1 : 0;
                        // edge_vertex += (AECB == 0.) ? 1 : 0;

                        // if (edge_vertex > 0) // Edge/Vertex detected. Edge = 1, Vertex = 2
                        // {
                        //     if (edge_vertex == 1)
                        //     {

                        //     }
                        // }
                        
                        cell = mesh->cell_neighbours[cell*mesh->faces_per_cell + intercepted_face];


                        // If intercepted with boundary of mesh, decay particle. TODO: Rebound them?
                        if (cell == MESH_BOUNDARY) 
                        {
                            if (PARTICLE_DEBUG)  cout << "\t\tParticle decayed after leaving the grid, x1: " << print_vec(x1) << " " << cell <<  endl ;
                            decayed = true;
                            return;
                        }



                        if (check_cell(cell, mesh)) // Particle is in immediately neighbouring cell
                        {
                            found_cell = true;
                            x0 = x1;
                            v0 = v1;
                            if (PARTICLE_DEBUG)  cout << "\t\tParticle has moved to cell " << cell << " " << ", x1: " << print_vec(x1) << endl ;
                        }
                        else { // Particle isn't in immediate neighbour
                            face_mask    = face_mask | (1 << (intercepted_face ^ 1));  // E.G If we detect interception with front face, don't trigger back face interception next time.
                            if (PARTICLE_DEBUG)  cout  << "\t\tParticle isn't in neighbour cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                        }
                    }
                    if (parallelepiped > 1)  x1 -= 1e-8*x_delta;;
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

            uint64_t timestep(Mesh<T> *mesh)  // Update position
            {
                if (decayed)  return MESH_BOUNDARY;

                // TODO: Timestep needs to be definitive amount of time. For now, it is a constant 0.01s.
                x1 = x0 + v1*1.0;
                v1 = v0 + a1*1.0;

               


                // Check if particle is in the current cell. Tetras = Volume/Area comparison method. https://www.peertechzpublications.com/articles/TCSIT-6-132.php.
                update_cell(mesh);

                return cell; 
            } 

            
    }; // class Particle
 
}   // namespace minicombust::particles 