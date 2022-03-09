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
            Mesh<T> *global_mesh;

            VisitWriter(Mesh<T> *global_mesh) : global_mesh(global_mesh)
            { }

            
            void write_file(string filename, int id)
            {
                // Print VTK Header
                ofstream vtk_file;
                vtk_file.open ("out/"+filename+"_timestep"+to_string(id)+".vtk");
                vtk_file << "# vtk DataFile Version 3.2 " << endl;
                vtk_file << "MiniCOMBUST " << endl;
                vtk_file << "ASCII " << endl;
                vtk_file << "DATASET UNSTRUCTURED_GRID " << endl;

                
                // TODO: Allow different datatypes
                // Print point data
                vtk_file << endl << "POINTS " << global_mesh->mesh_points_size << " double" ;
                for(int p = 0; p < global_mesh->mesh_points_size; p++)
                {
                    const int data_per_line = 10;
                    if (p % data_per_line == 0)  vtk_file << endl;
                    else             vtk_file << "  ";
                    vtk_file << print_vec(global_mesh->mesh_points[p]);
                }
                vtk_file << endl;

                // Print cell data
                vtk_file << endl << "CELLS " << global_mesh->mesh_size << " " << global_mesh->mesh_size*global_mesh->cell_size + global_mesh->mesh_size << endl;
                for(int c = 0; c < global_mesh->mesh_size; c++)
                {
                    vtk_file << global_mesh->cell_size << " ";
                    vec<T> *base_address = global_mesh->mesh_points;
                    for (int v = 0; v < global_mesh->cell_size; v++)
                    {
                        const uint64_t point_index = global_mesh->cells[c][v] - base_address;
                        vtk_file << point_index << " ";
                    }
                    vtk_file << endl;
                }

                // Print cell types
                vtk_file << endl << "CELL_TYPES " << global_mesh->mesh_size;
                for(int c = 0; c < global_mesh->mesh_size; c++)
                {
                    const int data_per_line = 30;
                    if (c % data_per_line == 0)  vtk_file << endl;
                    else             vtk_file << " ";
                    vtk_file << VTK_VOXEL;
                }
                vtk_file << endl;

                // // Print temperature values for points
                // vtk_file << endl << "POINT_DATA " << global_mesh->mesh_points_size << endl;
                // vtk_file << "SCALARS Temperature float 1" << endl;
                // vtk_file << "LOOKUP_TABLE default" << endl;
                // for(int p = 0; p < global_mesh->mesh_points_size; p++)
                // {
                //     // TODO: Instead of index, set values to be based of temperature values
                //     const int data_per_line = 20;
                //     if (p % data_per_line == 0)  vtk_file << endl;
                //     else             vtk_file << " ";
                //     vtk_file << p << " ";
                // } 
                // vtk_file << endl;

                // Print particle values for points
                vtk_file << endl << "POINT_DATA " << global_mesh->mesh_points_size << endl;
                vtk_file << "SCALARS Particles unsigned_int 1" << endl;
                vtk_file << "LOOKUP_TABLE default" << endl;
                for(int p = 0; p < global_mesh->mesh_points_size; p++)
                {
                    // TODO: Instead of index, set values to be based of temperature values
                    const int data_per_line = 20;
                    if (p % data_per_line == 0)  vtk_file << endl;
                    else             vtk_file << " ";
                    vtk_file << global_mesh->particles_per_point[p] << " ";
                } 
                vtk_file << endl;

                // TODO: Allow different cell types

                vtk_file.close();
            }

    }; // class VisitWriter

}   // namespace minicombust::visit 