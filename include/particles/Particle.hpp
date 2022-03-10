#pragma once

#include <cmath>
#include <limits>
#include <iomanip>

#include "utils/utils.hpp"
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
                
                // const double result = abs(dot_product(cross_product(a, b), c)) / 6.0;
                // cout << print_vec(*A) << " " << print_vec(*B) << " " << print_vec(*C) << " " << print_vec(*D) << " " << result << endl;

                return abs(dot_product(cross_product(a, b), c)) / 6.0;
            }

            bool check_cell(vec<T> **current_cell)
            {
                // TODO: Currently is solely a cube partial volume algorithm. Tetra partial volume algorithm;
                vec<T> box_size = *current_cell[H_VERTEX] - *current_cell[A_VERTEX];
                double total_volume  = abs(box_size.x*box_size.y*box_size.z);

                double partial_volumes = 0.0;
                for (int i=0; i < 8; i++)
                {
                    vec<T> partial_box_size = *current_cell[i] - x1;
                    partial_volumes += abs(partial_box_size.x*partial_box_size.y*partial_box_size.z);
                }
                
                if (abs(partial_volumes-total_volume) < 5.0e-10)  return true;
                return false;

            }

            void check_current_cell(Mesh<T> *mesh)
            {
                vec<T> **current_cell = mesh->cells[cell];
                if ( check_cell(current_cell) )
                {
                    if (PARTICLE_DEBUG)  cout << "\t\tParticle is still in cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                } 
                else
                {
                    bool found_cell = false;
                    vec<T> *A = &x0;
                    vec<T> *B = &x1;
                    
                    uint64_t face_mask = 0; // Prevents double interceptions                     

                    while (!found_cell)
                    {
                        int intercepted_face = -1;
                        // TODO Check if particle goes through vertex.
                        // int edge_vertex = 0;

                        double LHS; // LHS =  ACDB + ADFB + AFEB + AECB 
                        double RHS; // RHS =  ACDF + BCDF + AECF + BECF   
                        double ACDB, ADFB, AFEB, AECB; // LHS partial volumes. Also used for edge/vertex detection.
                        for (int face = 0; face < 6; face++)
                        {
                            if ((1 << face) & face_mask)  continue;
                            // TODO: Can we predict the order better? 
                            // Check whether particle - moving from A to B - intercepts face CDEF. If LHS == RHS, particle intercepts this face.
                            LHS = 0.0;
                            RHS = 0.0;

                            vec<T> *C = current_cell[CUBE_FACE_VERTEX_MAP[face][0]];
                            vec<T> *D = current_cell[CUBE_FACE_VERTEX_MAP[face][1]];
                            vec<T> *E = current_cell[CUBE_FACE_VERTEX_MAP[face][2]];
                            vec<T> *F = current_cell[CUBE_FACE_VERTEX_MAP[face][3]];

                            ACDB = tetrahedral_volume(A, C, D, B);
                            LHS += ACDB;
                            ADFB = tetrahedral_volume(A, D, F, B);
                            LHS += ADFB;
                            AFEB = tetrahedral_volume(A, F, E, B);
                            LHS += AFEB;
                            AECB = tetrahedral_volume(A, E, C, B);
                            LHS += AECB;

                            RHS += tetrahedral_volume(A, C, D, F);
                            RHS += tetrahedral_volume(A, E, C, F);
                            RHS += tetrahedral_volume(B, C, D, F);
                            RHS += tetrahedral_volume(B, E, C, F);

                            if (abs(LHS - RHS) < 5e-12)  
                            {
                                intercepted_face = face;
                                if (PARTICLE_DEBUG)  cout << "\t\tIntercepted Face " << intercepted_face << endl;
                                // printf("\t%.17f %.17f\n", LHS, RHS);
                                break;
                            }
                        }


                        // If intercepted with boundary of mesh, decay particle. TODO: Rebound them?
                        if (mesh->mesh_faces[cell][intercepted_face]->cell0 == nullptr) 
                        {
                            if (PARTICLE_DEBUG)  cout << "\t\tParticle decayed after leaving the grid, x1: " << print_vec(x1) <<  endl ;
                            decayed = true;
                            return;
                        }

                        vec<T> **neighbour_cell;


                        // Code for detecting edges/vertexes. Don't think this will be faster. Probably required for tetra implementation
                        // edge_vertex += (ACDB == 0.) ? 1 : 0;
                        // edge_vertex += (ADFB == 0.) ? 1 : 0;
                        // edge_vertex += (AFEB == 0.) ? 1 : 0;
                        // edge_vertex += (AECB == 0.) ? 1 : 0;

                        // if (edge_vertex > 0) // Edge/Vertex detected. Edge = 1, Vertex = 2
                        // {
                            
                        // }

                        // TODO: Is there a way of finding neighbouring cell, without checking if neighbour is our own cell?
                        if ( mesh->mesh_faces[cell][intercepted_face]->cell0 == current_cell)  neighbour_cell = mesh->mesh_faces[cell][intercepted_face]->cell1;
                        else                                                                   neighbour_cell = mesh->mesh_faces[cell][intercepted_face]->cell0;

                        if (check_cell(neighbour_cell)) // Particle is in immediate neighbour?
                        {
                            found_cell = true;
                            cell = (neighbour_cell - mesh->cells[0])/mesh->cell_size;
                            x0 = x1;
                            v0 = v1;
                            if (PARTICLE_DEBUG)  cout << "\t\tParticle has moved to cell " << cell << " " << ", x1: " << print_vec(x1) <<  endl ;
                        }
                        else { // Particle isn't in immediate neighbour
                            current_cell = neighbour_cell;
                            cell         = (neighbour_cell - mesh->cells[0])/mesh->cell_size;
                            face_mask    = face_mask | (1 << (intercepted_face ^ 1));  // E.G If we detect interception with front face, don't trigger back face interception next time.
                            if (PARTICLE_DEBUG)  cout  << "\t\tParticle isn't in neighbour cell " << cell << ", x1: " << print_vec(x1) <<  endl ;
                        }

                    }
                }
            }

        public:
            vec<T> x0 = 0.0;        // coordinates at timestep beginning
            vec<T> x1 = 0.0;        // coordinates at next timestep

            vec<T> v0 = 0.0;        // velocity at timestep beginning
            vec<T> v1 = 0.0;        // velocity at next timestep
            
            vec<T> a1 = 0.0;        // acceleration at next timestep
                        
            uint64_t cell = -1;          // cell at timestep beginning

            bool decayed = false;

            
            bool wall  = false;

            T dens0 = -1.0;       // current density (-1 = undefined)
            T diam0 =  1.0;
            T mass0 =  1.0;

            Particle(Mesh<T> *mesh,vec<T> start, vec<T> velocity, vec<T> acceleration) : x0(start), x1(start), v0(velocity), v1(velocity), a1(acceleration)
            { 
                for (int c = 0; c < mesh->mesh_size; c++)
                {
                    if (check_cell(mesh->cells[c]))  
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
                if (PARTICLE_DEBUG)  cout  << "\t\tParticle is starting in " << cell << ", x1: " << print_vec(x1) <<  endl ;
            }

            void timestep(Mesh<T> *mesh)
            {
                if (decayed)  return;

                // TODO: Timestep needs to be definitive amount of time. For now, it is a constant 0.01s.
                x1 = x0 + v1*1.0;
                v1 = v0 + a1*1.0;

                // Check if particle is in the current cell. Tetras = Volume/Area comparison method. https://www.peertechzpublications.com/articles/TCSIT-6-132.php.


                // Cube algorithm, assumes A is closer to origin. TODO.
                check_current_cell(mesh);

                // HOw to give wole class access to pointer, without stoing n times 
            } 

            
    }; // class Particle
 
}   // namespace minicombust::particles 