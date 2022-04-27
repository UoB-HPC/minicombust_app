#pragma once

#include <map>

#include <memory.h>
#include "utils/utils.hpp"

using namespace minicombust::utils; 

namespace minicombust::geometry 
{   
    static const uint64_t MESH_BOUNDARY = UINT64_MAX;

    enum FACE_DIRECTIONS { FRONT_FACE = 0, BACK_FACE = 1, LEFT_FACE = 2, RIGHT_FACE = 3, DOWN_FACE = 4, UP_FACE = 5};
    enum CUBE_VERTEXES { A_VERTEX = 0, B_VERTEX = 1, C_VERTEX = 2, D_VERTEX = 3, E_VERTEX = 4, F_VERTEX = 5, G_VERTEX = 6, H_VERTEX = 7};

    static const uint64_t CUBE_FACE_VERTEX_MAP[6][4] = 
    {
        {A_VERTEX, B_VERTEX, C_VERTEX, D_VERTEX}, // FRONT FACE
        {E_VERTEX, F_VERTEX, G_VERTEX, H_VERTEX}, // BACK FACE
        {A_VERTEX, C_VERTEX, E_VERTEX, G_VERTEX}, // LEFT FACE
        {B_VERTEX, D_VERTEX, F_VERTEX, H_VERTEX}, // RIGHT FACE
        {A_VERTEX, B_VERTEX, E_VERTEX, F_VERTEX}, // DOWN FACE
        {C_VERTEX, D_VERTEX, G_VERTEX, H_VERTEX}, // UP FACE
    };


    template<class T>
    class Face
    {
        private:


        public:
            uint64_t cell0;
            uint64_t cell1;

            Face(uint64_t cell0, uint64_t cell1) : cell0(cell0), cell1(cell1)
            { }
            
    }; // class Face

    template<class T>
    class Mesh 
    {
        private:

          // Calculate the centre point in each cell
          // Computed as the average of all vertex positions
          void calculate_cell_centres(void) {
              for (uint64_t c = 0; c < mesh_size; ++c) {
                  cell_centres[c] = vec<T>{0.0, 0.0, 0.0};
                  for (uint32_t i = 0; i < cells_vertex_count; ++i) {
                      cell_centres[c] += points[cells[c*cell_size + i]];
                  }
                  cell_centres[c] /= static_cast<T>(cells_vertex_count);
              }
          }
 
        public:
            const uint32_t cells_vertex_count = 3; // Generic, 3 = triangle etc

            uint64_t points_size;         // Number of points in the mesh
            uint64_t mesh_size;           // Number of polygons in the mesh
            uint64_t cell_size;           // Number of points in the cell
            uint64_t faces_size;          // Number of unique faces in the mesh
            uint64_t faces_per_cell;      // Number of faces in a cell

            
            
            vec<T> cell_size_vector;      // Cell size
            vec<T> *points;               // Mesh points    = {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, ...}:
            uint64_t *cells;              // Cells          = {{0, 1, 2, 300, 40, 36, 7, 2}, {1, 2, 4, 300}, ...};
            Face<T> *faces;               // Faces          = {{0, BOUNDARY}, {0, BOUNDARY}, {0, BOUNDARY}, {0, 1}, ...}; 
            vec<T> *cell_centres;         // Cell centres   = {{0.5, 3.0, 4.0}, {2.5, 3.0, 4.0}, ...};
            uint64_t *cell_neighbours;    // Cell faces     = {{0, 1, 2, 3, 4, 5}, {6, 1, 7, 3, 8, 5}}
 
            uint64_t max_cell_particles;

            uint64_t *particles_per_point; // Number of particles in each cell

            // Particle source terms
            T      *evaporated_fuel_mass_rate;       // particle_rate_of_mass
            T      *particle_energy_rate;            // particle_energy
            vec<T> *particle_momentum_rate;          // particle_momentum_rate

            // Flow source terms
            // Inputs from flow: relative_acc, relative_gas_liq_vel, kinematic viscoscity?, air_temp, air_pressure
            vec<T> *gas_velocity;
            vec<T> *gas_velocity_gradient;
            T      *gas_pressure;
            T      *gas_pressure_gradient;
            T      *gas_temperature;
            T      *gas_temperature_gradient;

            Mesh(uint64_t points_size, uint64_t mesh_size, uint64_t cell_size, uint64_t faces_size, uint64_t faces_per_cell, vec<T> *points, uint64_t *cells, Face<T> *faces, uint64_t *cell_neighbours, uint64_t max_cell_particles) 
            : points_size(points_size), mesh_size(mesh_size), cell_size(cell_size), faces_size(faces_size), faces_per_cell(faces_per_cell), points(points), cells(cells), faces(faces), cell_neighbours(cell_neighbours), max_cell_particles(max_cell_particles)
            {
                // Allocate space for and calculate cell centre co-ordinates
                const size_t mesh_cell_centre_size = mesh_size * sizeof(vec<T>);
                const size_t particles_per_point_size = points_size * sizeof(uint64_t);
                const size_t points_array_size           = points_size*sizeof(vec<double>);
                const size_t cells_array_size            = mesh_size*cell_size*sizeof(uint64_t);
                const size_t faces_array_size            = faces_size*sizeof(Face<T>);
                const size_t cell_neighbours_array_size  = mesh_size*faces_per_cell*sizeof(uint64_t);
                const size_t evaporated_fuel_mass_rate_size  = mesh_size*sizeof(T);
                const size_t particle_energy_size            = mesh_size*sizeof(T);
                const size_t particle_momentum_rate_size     = mesh_size*sizeof(vec<T>);
                const size_t gas_velocity_size               = mesh_size*sizeof(vec<T>);
                const size_t gas_velocity_gradient_size      = mesh_size*sizeof(vec<T>);
                const size_t gas_pressure_size               = mesh_size*sizeof(T);
                const size_t gas_pressure_gradient_size      = mesh_size*sizeof(T);
                const size_t gas_temperature_size            = mesh_size*sizeof(T);
                const size_t gas_temperature_gradient_size   = mesh_size*sizeof(T);
                cell_centres = (vec<T> *)malloc(mesh_cell_centre_size);
                particles_per_point = (uint64_t *)malloc(particles_per_point_size);
                evaporated_fuel_mass_rate = (T *)     malloc(evaporated_fuel_mass_rate_size);
                particle_energy_rate      = (T *)     malloc(particle_energy_size);
                particle_momentum_rate    = (vec<T> *)malloc(particle_momentum_rate_size);
                gas_velocity                                 = (vec<T> *)malloc(gas_velocity_size);
                gas_velocity_gradient                        = (vec<T> *)malloc(gas_velocity_size);
                gas_pressure                                 = (T *)     malloc(gas_pressure_size);
                gas_pressure_gradient                        = (T *)     malloc(gas_pressure_size);
                gas_temperature                              = (T *)     malloc(gas_temperature_size);
                gas_temperature_gradient                     = (T *)     malloc(gas_temperature_size);
                const size_t total_size = mesh_cell_centre_size + particles_per_point_size + points_array_size + cells_array_size + 
                                          faces_array_size + cell_neighbours_array_size + 
                                          evaporated_fuel_mass_rate_size + particle_energy_size + particle_momentum_rate_size + 
                                          gas_velocity_size + gas_pressure_size + gas_temperature_size +
                                          gas_velocity_gradient_size + gas_pressure_gradient_size + gas_temperature_gradient_size;

                
                printf("\nMesh storage requirements:\n");
                printf("\tAllocating mesh cell centres                                (%.2f MB)\n",                 (float)(mesh_cell_centre_size)/1000000.0);
                printf("\tAllocating array of particles per cell                      (%.2f MB)\n",                 (float)(particles_per_point_size)/1000000.0);
                printf("\tAllocating vertexes                                         (%.2f MB) (%llu vertexes)\n", (float)(points_array_size)/1000000.0, points_size);
                printf("\tAllocating cells                                            (%.2f MB) (%llu cells)\n",    (float)(cells_array_size)/1000000.0, mesh_size);
                printf("\tAllocating faces                                            (%.2f MB) (%llu faces)\n",    (float)(faces_array_size)/1000000.0, faces_size);
                printf("\tAllocating cell neighbour indexes                           (%.2f MB)\n",                 (float)(cell_neighbours_array_size)/1000000.0);
                printf("\tAllocating evaporated fuel mass source term                 (%.2f MB)\n",                 (float)(evaporated_fuel_mass_rate_size)/1000000.0);
                printf("\tAllocating particle energy source term                      (%.2f MB)\n",                 (float)(particle_energy_size)/1000000.0);
                printf("\tAllocating particle_momentum source term                    (%.2f MB)\n",                 (float)(particle_momentum_rate_size)/1000000.0);
                printf("\tAllocating gas velocity source term                         (%.2f MB)\n",                 (float)(gas_velocity_size)/1000000.0);
                printf("\tAllocating gas velocity gradient source term                (%.2f MB)\n",                 (float)(gas_velocity_size)/1000000.0);
                printf("\tAllocating gas pressure source term                         (%.2f MB)\n",                 (float)(gas_pressure_size)/1000000.0);
                printf("\tAllocating gas pressure gradient source term                (%.2f MB)\n",                 (float)(gas_pressure_size)/1000000.0);
                printf("\tAllocating gas temperature source term                      (%.2f MB)\n",                 (float)(gas_temperature_size)/1000000.0);
                printf("\tAllocating gas temperature gradient source term             (%.2f MB)\n",                 (float)(gas_temperature_size)/1000000.0);
                printf("\tAllocated mesh. Total size                                  (%.2f MB)\n\n",               (float)total_size/1000000.0);

                calculate_cell_centres();
                cell_size_vector = points[cells[H_VERTEX]] - points[cells[A_VERTEX]];

                // DUMMY VALUES 
                for (int c = 0; c < mesh_size; c++)
                {
                    gas_velocity[c]     = {100., 100., 100.};
                    gas_pressure[c]     = 960.;
                    gas_temperature[c]  = 550.;
                }
            }

            void clear_particles_per_point_array(void)
            {
                memset(particles_per_point, 0, points_size * sizeof(uint64_t));
            }


            string get_face_string(uint64_t face)
            {
                switch(face)  
                {
                    case FRONT_FACE:
                        return "FRONT";
                    case BACK_FACE:
                        return "BACK";
                    case LEFT_FACE:
                        return "LEFT";
                    case RIGHT_FACE:
                        return "RIGHT";
                    case DOWN_FACE:
                        return "DOWN";
                    case UP_FACE:
                        return "UP";
                default:
                        return "INVALID FACE"; 
                }
            }
            string get_vertex_string(uint64_t vertex)
            {
                switch(vertex)  
                {
                    case A_VERTEX:
                        return "A";
                    case B_VERTEX:
                        return "B";
                    case C_VERTEX:
                        return "C";
                    case D_VERTEX:
                        return "D";
                    case E_VERTEX:
                        return "E";
                    case F_VERTEX:
                        return "F";
                    case G_VERTEX:
                        return "G";
                    case H_VERTEX:
                        return "H";
                default:
                        return "INVALID VERTEX"; 
                }
            }

    }; // class Mesh

}   // namespace minicombust::particles 
