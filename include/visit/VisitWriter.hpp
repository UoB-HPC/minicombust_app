#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "utils/utils.hpp"
#include "visit/vtkCellType.h"



namespace minicombust::visit 
{   
    using namespace std;
    using namespace minicombust::utils; 

    template<class T>
    class VisitWriter 
    {

        public:
            Mesh<T> *mesh;

            VisitWriter(Mesh<T> *mesh) : mesh(mesh)
            {  }

            void write_mesh(string filename)
            {
                                // Print VTK Header
                ofstream vtk_file;
                vtk_file.open ("out/" + filename + "_mesh.vtk");
                vtk_file << "# vtk DataFile Version 3.0 " << endl;
                vtk_file << "MiniCOMBUST " << endl;
                vtk_file << "ASCII " << endl;
                vtk_file << "DATASET UNSTRUCTURED_GRID " << endl;

                
                // TODO: Allow different datatypes
                // Print point data
                vtk_file << endl << "POINTS " << mesh->points_size << " double" << endl;
                for(uint64_t p = 0; p < mesh->points_size; p++)
                {
                    const int data_per_line = 10;
                    if (p % data_per_line == 0)  vtk_file << endl;
                    else             vtk_file << "  ";
                    vtk_file << print_vec(mesh->points[p]);
                }
                vtk_file << endl;

                // Print cell data
                vtk_file << endl << "CELLS " << mesh->mesh_size << " " << mesh->mesh_size*mesh->cell_size + mesh->mesh_size << endl;
                for(uint64_t c = 0; c < mesh->mesh_size; c++)
                {
                    vtk_file << mesh->cell_size << " ";
                    for (uint64_t v = 0; v < mesh->cell_size; v++)  vtk_file << mesh->cells[c*mesh->cell_size + v] << " ";
                    vtk_file << endl;
                }

                // Print cell types
                vtk_file << endl << "CELL_TYPES " << mesh->mesh_size;
                for(uint64_t c = 0; c < mesh->mesh_size; c++)
                {
                    const int data_per_line = 30;
                    if (c % data_per_line == 0)  vtk_file << endl;
                    else             vtk_file << " ";
                    vtk_file << VTK_VOXEL;
                }
                vtk_file << endl;
            }


            void write_particles(string filename, int id, vector<Particle<T>>& particles)
            {
                // Print VTK Header
                ofstream vtk_file;

                vtk_file.open ("out/" + filename + "_particle_timestep" + to_string(id) + ".vtk");
                vtk_file << "# vtk DataFile Version 3.0 " << endl;
                vtk_file << "MiniCOMBUST " << endl;
                vtk_file << "ASCII " << endl;
                vtk_file << "DATASET POLYDATA " << endl;

                
                // TODO: Allow different datatypes
                // Print point data
                // vtk_file << endl << "POINTS " << mesh->points_size << " float"  << endl;
                // for(int p = 0; p < mesh->points_size; p++)
                // {
                //     // const int data_per_line = 10;
                //     // if (p % data_per_line == 0)  vtk_file << endl;
                //     // else             vtk_file << "  ";
                //     vtk_file << print_vec(mesh->points[p]) << "\t";
                // }
                // vtk_file << endl << endl;




            
                // // Print particle values for points
                // int non_zero_points = 0;
                // for(int p = 0; p < mesh->points_size; p++)  non_zero_points += (mesh->particles_per_point[p] > 0) ? 1 : 0;
                // vtk_file << "VERTICES " << non_zero_points << " " << non_zero_points * 2  << endl;
                // for(int p = 0; p < mesh->points_size; p++)
                // {
                //     if (mesh->particles_per_point[p] > 0) vtk_file << "1 " << p << "\t";
                // }
                // vtk_file << mesh->points_size - non_zero_points << " ";
                // for(int p = 0; p < mesh->points_size; p++)
                // {
                //     if (mesh->particles_per_point[p] == 0) vtk_file << p << " ";
                // }
                // vtk_file << endl << endl;



                // vtk_file << "POINT_DATA " << non_zero_points << endl;
                // vtk_file << "SCALARS num_particles float" << endl;
                // vtk_file << "LOOKUP_TABLE default" << endl;
                // for(int p = 0; p < mesh->points_size; p++)
                // {
                //     if (mesh->particles_per_point[p] > 0)  vtk_file << mesh->particles_per_point[p] << "\t";
                //     if (mesh->particles_per_point[p] > 0)  cout << mesh->particles_per_point[p] << " particles are present at " << p << endl;
                // }
                // vtk_file << 0 << "\t";
                // vtk_file << endl;




                // // Print particle values for points
                // int non_zero_points = 0;
                // for(int p = 0; p < mesh->points_size; p++)  non_zero_points += (mesh->particles_per_point[p] != 0) ? 1 : 0;
                // vtk_file << "VERTICES " << mesh->points_size << " " << mesh->points_size * 2  << endl;
                // for(int p = 0; p < mesh->points_size; p++)
                // {
                //     vtk_file << "1 " << p << "\t";
                // }
                // vtk_file << endl;



                // vtk_file << "POINT_DATA " << mesh->points_size << endl;
                // vtk_file << "SCALARS num_particles float" << endl;
                // vtk_file << "LOOKUP_TABLE default" << endl;
                // for(int p = 0; p < mesh->points_size; p++)
                // {
                //     vtk_file << mesh->particles_per_point[p] << "\t";
                //     if (mesh->particles_per_point[p] > 0)  cout << mesh->particles_per_point[p] << " particles are present at " << p << endl;
                // } 
                // vtk_file << endl;


                
                vtk_file << endl << "POINTS " << particles.size() << " float"  << endl;
                for (uint64_t p = 0; p < particles.size(); p++)
                {
                    vtk_file << print_vec(particles[p].x1) << endl;
                }
                vtk_file << endl;


                // Print particle values for points
                vtk_file << endl << "VERTICES " << particles.size() << " " << particles.size() * 2  << endl;
                uint64_t count = 0;
                for(uint64_t p = 0; p < particles.size(); p++)
                {
                    vtk_file << "1 " << count++ << "\t";
                }
                vtk_file << endl;

                vtk_file << endl << "POINT_DATA " << particles.size() << endl;
                vtk_file << "SCALARS mass float" << endl;
                vtk_file << "LOOKUP_TABLE default" << endl;
                for(uint64_t p = 0; p < particles.size(); p++)
                {
                    vtk_file << particles[p].mass << "\t";
                } 
                vtk_file << endl;

                vtk_file << "SCALARS temp float" << endl;
                vtk_file << "LOOKUP_TABLE default" << endl;
                for(uint64_t p = 0; p < particles.size(); p++)
                {
                    vtk_file << particles[p].temp << "\t";
                } 
                vtk_file << endl;

                vtk_file << "VECTORS velocity float" << endl;
                for(uint64_t p = 0; p < particles.size(); p++)
                {
                    vtk_file << print_vec(particles[p].v1) << "\t";
                } 
                vtk_file << endl;

                vtk_file.close();
            }

    }; // class VisitWriter

}   // namespace minicombust::visit 
