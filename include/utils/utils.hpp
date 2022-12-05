#pragma once


#include <string>  
#include <iostream> 
#include <functional> 
#include <sstream>   
#include <math.h>
#include <inttypes.h>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <algorithm>


#include <mpi.h>

#define PARTICLE_DEBUG 0
#define LOGGER 1
#define PARTICLE_SOLVER_DEBUG 0

#define FLOW 0
#define PARTICLE 1


typedef long long int int128_t;
typedef unsigned long long int uint128_t;

namespace minicombust::utils 
{
    using namespace std;

    template <typename T> 
    struct vec {
        T x;
        T y;
        T z;

        inline vec<T>& operator+=(const vec<T>& rhs)
        {
            x += rhs.x;
            y += rhs.y;
            z += rhs.z;
	        return *this;
        }

        inline vec<T>& operator-=(const vec<T>& rhs)
        {
            x -= rhs.x;
            y -= rhs.y;
            z -= rhs.z;
	        return *this;
        }

        inline vec<T>& operator/=(const T rhs)
        {
            x /= rhs;
            y /= rhs;
            z /= rhs;
	        return *this;
        }

        inline vec<T>& operator/=(const vec<T> rhs)
        {
            x /= rhs.x;
            y /= rhs.y;
            z /= rhs.z;
	        return *this;
        }
    };

    template <typename T> 
    struct vec_soa {
        T *x;
        T *y;
        T *z;
    };

    template <typename T> 
    struct flow_aos 
    {
        vec<T> vel;
        T pressure;
        T temp;
    };

    template <typename T> 
    struct particle_aos 
    {
        vec<T> momentum = {0.0, 0.0, 0.0};
        T energy        = 0.0;
        T fuel          = 0.0;
    };

    template<typename T>
    inline bool vec_nequal(const vec<T> lhs, const vec<T> rhs)
    {
        return (lhs.x != rhs.x) || (lhs.y != rhs.y) || (lhs.z != rhs.z);
    }

    template<typename T>
    inline bool vec_equal(const vec<T> lhs, const vec<T> rhs)
    {
        return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
    }

    template<typename T>
    inline T sum(vec<T> a) 
    {
        return a.x + a.y + a.z;
    }

    template<typename T>
    inline vec<T> operator+(vec<T> a, vec<T> b) 
    {
        vec<T> sum = {a.x + b.x, a.y + b.y, a.z + b.z};
        return sum;
    }
    template<typename T>
    inline vec<T> operator+(vec<T> a, T b) 
    {
        vec<T> sum = {a.x + b, a.y + b, a.z + b};
        return sum;
    }
    template<typename T>
    inline vec<T> operator+(T a, vec<T> b) 
    {
        vec<T> sum = {a + b.x, a + b.y, a + b.z};
        return sum;
    }


    template<typename T>
    inline vec<T> operator-(vec<T> a, vec<T> b) 
    {
        vec<T> sum = {a.x - b.x, a.y - b.y, a.z - b.z};
        return sum;
    }

    template<typename T>
    inline vec<T> operator-(const vec<T> b) 
    {
        vec<T> negative = {b.x, b.y, b.z};
        return negative;
    }


    template<typename T>
    inline vec<T> operator/(vec<T> a, T b) 
    {
        vec<T> sum = {a.x / b, a.y / b, a.z / b};
        return sum;
    }

    template<typename T>
    inline vec<T> operator/(T a, vec<T> b) 
    {
        vec<T> sum = {a / b.x, a / b.y, a / b.z};
        return sum;
    }

    template<typename T>
    inline vec<T> operator/(vec<T> a, vec<T> b) 
    {
        vec<T> sum = {a.x / b.x, a.y / b.y, a.z / b.z};
        return sum;
    }

    template<typename T>
    inline vec<T> operator*(vec<T> a, T b) 
    {
        vec<T> sum = {a.x * b, a.y * b, a.z * b};
        return sum;
    }

    template<typename T>
    inline vec<T> operator*(T b, vec<T> a) 
    {
        vec<T> sum = {a.x * b, a.y * b, a.z * b};
        return sum;
    }

    template<typename T>
    inline vec<T> operator*(vec<T> b, vec<T> a) 
    {
        vec<T> sum = {a.x * b.x, a.y * b.y, a.z * b.z};
        return sum;
    }


    template<typename T>
    inline bool operator<(vec<T> a, vec<T> b) 
    {
        return a.x < b.x && a.y < b.y && a.z < b.z;
    }

    template<typename T>
    inline bool operator>(vec<T> a, vec<T> b) 
    {
        return a.x > b.x && a.y > b.y && a.z > b.z;
    }

    template<typename T>
    inline bool operator<=(vec<T> a, vec<T> b) 
    {
        return a.x <= b.x && a.y <= b.y && a.z <= b.z;
    }

    template<typename T>
    inline bool operator>=(vec<T> a, vec<T> b) 
    {
        return a.x >= b.x && a.y >= b.y && a.z >= b.z;
    }

    template<typename T> 
    inline T magnitude(vec<T> v)
    {
        return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    }

    template<typename T> 
    inline T magnitude(T v)
    {
        return abs(v);
    }

    template<typename T> 
    inline vec<T> abs_vec(vec<T> v)
    {
        return vec<T> {abs(v.x), abs(v.y), abs(v.z)};
    }

    template<typename T> 
    inline vec<T> cross_product(vec<T> a, vec<T> b)
    {
        vec<T> cross_product = {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
        return cross_product;
    }

    template<typename T> 
    inline T dot_product(vec<T> a, vec<T> b)
    {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    template<typename T> 
    inline T dot_product(T a, vec<T> b)
    {
        return a*(b.x + b.y + b.z);
    }

    template<typename T> 
    inline string print_vec(vec<T> v)
    {
        stringstream buffer;
        buffer << v.x << " " << v.y << " " << v.z;
        return buffer.str();
    }

    template<typename T> 
    inline string print_cube_cell(uint64_t *cube, vec<T> *points)
    {
        stringstream buffer;
        buffer << endl;
        for(int y=0; y < 2; y++)
        {
            for(int x=0; x < 4; x++)
            {
                buffer << "{";
                buffer << print_vec(points[cube[y*4 + x]]);
                buffer << "}";
                if (x!=3)  buffer << ", ";
            }
            buffer << endl;
        }
        return buffer.str();
    }

    template<typename T> 
    inline vec_soa<T> allocate_vec_soa(uint64_t n)
    {
        vec_soa<T> v;
        v.x = (T*)malloc(n*sizeof(T));
        v.y = (T*)malloc(n*sizeof(T));
        v.z = (T*)malloc(n*sizeof(T));
        return v;
    }

    struct Particle_Logger {
        uint64_t num_particles;
        uint64_t emitted_particles;
        uint64_t cell_checks;
        uint64_t position_adjustments;
        uint64_t lost_particles;
        uint64_t boundary_intersections;
        uint64_t decayed_particles;
        uint64_t unsplit_particles;
        uint64_t breakups;
        uint64_t burnt_particles;
        double avg_particles;
        double breakup_age;
        double interpolated_cells;
    };

    struct MPI_Config {
        int rank;
        int particle_flow_rank;
        int world_size;
        int particle_flow_world_size;
        MPI_Comm world;
        MPI_Comm particle_flow_world;
        int solver_type;
        MPI_Datatype MPI_FLOW_STRUCTURE;
        MPI_Datatype MPI_PARTICLE_STRUCTURE;
        MPI_Op MPI_PARTICLE_OPERATION;
    };

    inline void MPI_GatherSet (MPI_Config *mpi_config, unordered_set<uint64_t>& indexes_set, uint64_t *indexes, function<void(uint64_t, uint64_t **)> resize_fn)
    { 
        const uint64_t rank = mpi_config->rank;

        uint64_t *curr_indexes = indexes; 

        bool have_data = true;
        for ( int level = 2; level <= mpi_config->world_size ; level *= 2)
        {
            MPI_Barrier(mpi_config->world);
            if (have_data)
            {
                bool reciever = ((rank+1) % level) ? false : true;

                if ( reciever )
                {
                    uint64_t send_rank = rank - (level / 2);
                    int send_count;
                    // printf("LEVEL %d: Rank %d recv from %d\n", level, rank, send_rank);

                    MPI_Recv (&send_count,  1, MPI_UINT64_T, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);
                    resize_fn(indexes_set.size() + send_count, &curr_indexes);

                    uint64_t *recv_indexes = curr_indexes + indexes_set.size(); // Make sure this is done after resizing to ensure memory is contiguous.
                    MPI_Recv (recv_indexes, send_count, MPI_UINT64_T, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);
                    
                    for (int i = 0; i < send_count; i++)
                    {
                        if ( !indexes_set.contains(recv_indexes[i]) )
                        {
                            curr_indexes[indexes_set.size()] = recv_indexes[i];
                            indexes_set.insert(recv_indexes[i]);
                        }
                    }
                }
                else
                {
                    uint64_t recv_rank  = rank + (level / 2);
                    int send_count = indexes_set.size();
                    // printf("LEVEL %d: Rank %d send to %d\n", level, rank, recv_rank);

                    MPI_Send (&send_count,           1, MPI_UINT64_T, recv_rank, level, mpi_config->world);
                    MPI_Send (curr_indexes, send_count, MPI_UINT64_T, recv_rank, level, mpi_config->world);
                    
                    have_data = false;
                }

                // if (!have_data) printf("LEVEL %d: Rank %d NO DATA \n", level, rank);
            }
        }
    }


    template<typename T>
    inline void MPI_GatherMap (MPI_Config *mpi_config, unordered_map<uint64_t, particle_aos<T>>& cell_particle_map, uint64_t *indexes, particle_aos<T> *indexed_fields, function<void(uint64_t, uint64_t **, particle_aos<T> **)> resize_fn)
    { // NEED TO FIX ARRAY RESIZING
        const uint64_t rank = mpi_config->rank;

        uint64_t *curr_indexes               = indexes; 
        particle_aos<T> *curr_indexed_fields = indexed_fields; 

        bool have_data = true;
        for ( int level = 2; level <= mpi_config->world_size ; level *= 2)
        {
            if (have_data)
            {
                bool reciever = ((rank+1) % level) ? false : true;

                if ( reciever )
                {
                    uint64_t send_rank = rank - (level / 2);
                    int send_count;
                    // printf("LEVEL %d: Rank %d recv from %d\n", level, rank, send_rank);

                    MPI_Recv (&send_count, 1, MPI_UINT64_T, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);
                    resize_fn(cell_particle_map.size() + send_count, &curr_indexes, &curr_indexed_fields);
                    uint64_t        *recv_indexes        = curr_indexes        + cell_particle_map.size();
                    particle_aos<T> *recv_indexed_fields = curr_indexed_fields + cell_particle_map.size();


                    MPI_Recv (recv_indexes,        send_count, MPI_UINT64_T, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);
                    MPI_Recv (recv_indexed_fields, send_count, mpi_config->MPI_PARTICLE_STRUCTURE, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);

                    for (int i = 0; i < send_count; i++)
                    {
                        if ( cell_particle_map.contains(recv_indexes[i]) )
                        {
                            cell_particle_map[recv_indexes[i]].momentum  += recv_indexed_fields[i].momentum;
                            cell_particle_map[recv_indexes[i]].energy    += recv_indexed_fields[i].energy;
                            cell_particle_map[recv_indexes[i]].fuel      += recv_indexed_fields[i].fuel;
                        }
                        else
                        {
                            curr_indexes[cell_particle_map.size()]        = recv_indexes[i];
                            curr_indexed_fields[cell_particle_map.size()] = recv_indexed_fields[i];
                            cell_particle_map[recv_indexes[i]]       = recv_indexed_fields[i];
                        }
                    }
                }
                else
                {
                    uint64_t recv_rank  = rank + (level / 2);
                    int send_count = cell_particle_map.size();
                    // printf("LEVEL %d: Rank %d send to %d\n", level, rank, recv_rank);

                    MPI_Ssend (&send_count,                  1, MPI_UINT64_T,                       recv_rank, level, mpi_config->world);
                    MPI_Ssend (curr_indexes,        send_count, MPI_UINT64_T,                       recv_rank, level, mpi_config->world);
                    MPI_Ssend (curr_indexed_fields, send_count, mpi_config->MPI_PARTICLE_STRUCTURE, recv_rank, level, mpi_config->world);
                    
                    have_data = false;
                }
                // if (!have_data) printf("LEVEL %d: Rank %d NO DATA \n", level, rank);
            }
        }
    }

    template<typename T>
    void sum_particle_aos(void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype* datatype)
    {
        (void)(datatype); //Squashes unused warning. Parameter required for MPI op.

        particle_aos<T>* input  = (particle_aos<T>*)inputBuffer;
        particle_aos<T>* output = (particle_aos<T>*)outputBuffer;
    
        for(int i = 0; i < *len; i++)
        {
            output[i].momentum += input[i].momentum;
            output[i].energy   += input[i].energy;
            output[i].fuel     += input[i].fuel;
        }
    }
}