#pragma once

#include <map>

#include "utils/utils.hpp"

using namespace minicombust::utils; 



namespace minicombust::geometry 
{   
    template<class T>
    class Tetrahedral 
    {
        public:
            vec<T> vertex_A;
            vec<T> vertex_B;
            vec<T> vertex_C;
            vec<T> vertex_D;

            Tetrahedral (vec<T> A, vec<T> B, vec<T> C, vec<T> D) : 
                                        vertex_A(A), vertex_B(B), vertex_C(C), vertex_D(D)
            { }

    }; // class Tetrahedral

    template<class T>
    class Mesh 
    {
        private:
 
        public:
            const uint32_t obj_vertex_count = 3; // Generic, 3 = triangle etc

            uint64_t mesh_points_size;  // Number of points in the mesh
            uint64_t mesh_obj_size;     // Number of polygons in the mesh

            vec<T> *mesh_points;          // Mesh Points
            vec<T> ***mesh_obj_vertexes;   // Array of [obj_vertex_count*mesh_point pointers]

            Mesh(uint64_t points_size, uint64_t objects_size) : mesh_points_size(points_size), mesh_obj_size(objects_size)
            {
                mesh_points = (vec<T> *)malloc(mesh_points_size * (sizeof(vec<T>)));

                mesh_obj_vertexes = (vec<T> **)malloc(obj_vertex_count * mesh_obj_size * (sizeof(vec<T> *)));
            }

            Mesh(uint64_t points_size, uint64_t objects_size, vec<T> *points, vec<T> ***objects) : mesh_points_size(points_size), mesh_obj_size(objects_size),
                                                                                                  mesh_points(points), mesh_obj_vertexes(objects)
            { }

    }; // class Mesh

}   // namespace minicombust::particles 


// namespace minicombust::geometry 
// {
    



//     template<class T>
//     class Tetrahedral 
//     {
//         public:
//             vec<T> vertex_A;
//             vec<T> vertex_B;
//             vec<T> vertex_C;
//             vec<T> vertex_D;

//             Tetrahedral (vec<T> A, vec<T> B, vec<T> C, vec<T> D) : 
//                                         vertex_A(A), vertex_B(B), vertex_C(C), vertex_D(D)
//             { }

//             void add_tetrahedral(vec<T> A, vec<T> B, vec<T> C, vec<T> D);
//             void add_tetrahedral(Tetrahedral *tetrahedral);
//             void add_cuboid(vec<T> A, vec<T> B, vec<T> C, vec<T> D, vec<T> W, vec<T> X, vec<T> Y, vec<T> Z);

//     }; // class Tetrahedral

//     template<class T>
//     class Mesh 
//     {
//         private:
//             vec<T> mesh_points

//         public:
//             uint64_t mesh_size;
//             Tetrahedral<T> **mesh;
            

//             Mesh(uint64_t len) : mesh_size(len) 
//             {
//                 mesh = (Tetrahedral<T> **)malloc(mesh_size * (sizeof(Tetrahedral<T> *)));
//             }

//             void add_tetrahedral(vec<T> A, vec<T> B, vec<T> C, vec<T> D)
//             {
//                 if (current_size >= mesh_size)
//                 {
//                     printf("mesh_full\n");
//                     return;
//                 }

//                 Tetrahedral<T> *new_tetrahedral = new Tetrahedral<T>(A, B, C, D);
//                 mesh[current_size++] = new_tetrahedral;
//             }

//             void add_tetrahedral(Tetrahedral<T> *new_tetrahedral)
//             {
//                 if (current_size >= mesh_size)
//                 {
//                     printf("mesh_full\n");
//                     return;
//                 }

//                 mesh[current_size++] = new_tetrahedral;
//             }

//             void add_cuboid(vec<T> A, vec<T> B, vec<T> C, vec<T> D, vec<T> W, vec<T> X, vec<T> Y, vec<T> Z)
//             {
//                 if (current_size + 5 > mesh_size)
//                 {
//                     printf("mesh_full: %d\n", current_size);
//                     return;
//                 }

//                 Tetrahedral<T> *central_tetrahedral  = new Tetrahedral<T>(A, C, X, Z);
//                 Tetrahedral<T> *W_corner_tetrahedral = new Tetrahedral<T>(A, W, X, Z);
//                 Tetrahedral<T> *B_corner_tetrahedral = new Tetrahedral<T>(A, B, C, X);
//                 Tetrahedral<T> *D_corner_tetrahedral = new Tetrahedral<T>(A, C, D, Z);
//                 Tetrahedral<T> *Y_corner_tetrahedral = new Tetrahedral<T>(C, X, Y, Z);

//                 mesh[current_size++] = central_tetrahedral;
//                 mesh[current_size++] = W_corner_tetrahedral;
//                 mesh[current_size++] = B_corner_tetrahedral;
//                 mesh[current_size++] = D_corner_tetrahedral;
//                 mesh[current_size++] = Y_corner_tetrahedral;

//             }

//     }; // class Mesh

// }   // namespace minicombust::particles 