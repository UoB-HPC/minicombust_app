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

    inline int get_prime_factors ( int n, int *prime_factors )
    {
        const int max_factors = ceil(log2(n));
        cout << n << " prime factors: ";
        prime_factors = (int *)malloc(max_factors * sizeof(int));


        int nfactors = 0;

        // Print the number of 2s that divide n
        while (n % 2 == 0)
        {
            prime_factors[nfactors++] = 2;
            n = n/2;
        }
    
        // n must be odd at this point. So we can skip
        // one element (Note i = i+2)
        for (int i = 3; i <= sqrt(n); i += 2)
        {
            // While i divides n, print i and divide n
            while (n % i == 0)
            {
                prime_factors[nfactors++] = i;
                n = n/i;
            }
        }
    
        // This condition is to handle the case when n
        // is a prime number greater than 2
        if (n > 2 || nfactors == 0)
            prime_factors[nfactors++] = n;

        for (int x = 0;  x < nfactors; x++)
            cout << prime_factors[x] << " ";
        cout << endl;
    
        return nfactors;
    }

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
        const int world_size = mpi_config->world_size;
        int rank             = mpi_config->rank;
        int alias_rank       = (rank + 1) % world_size;
        
        int rounded_world_size = (int)pow(2., ceil(log((double)world_size)/log(2.)));

        uint64_t *curr_indexes = indexes; 

        bool have_data = true;
        for ( int level = 2; level <= rounded_world_size ; level *= 2)
        {
            if (have_data)
            {
                bool reciever        = !(alias_rank % level);
                int alias_send_rank  = alias_rank + (level / 2);
                int alias_recv_rank  = alias_rank - (level / 2);

                if ( reciever )
                {
                    if (alias_send_rank >= world_size)  continue;

                    uint64_t send_count;
                    int send_rank = (alias_send_rank + world_size - 1) % world_size;
                    // printf("LEVEL %d: Rank %d (AR %d) recv from %d\n", level, rank, alias_rank, send_rank);
                
                    MPI_Recv (&send_count, 1, MPI_UINT64_T, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);
                    resize_fn(indexes_set.size() + send_count, &curr_indexes);
                    uint64_t *recv_indexes = curr_indexes + indexes_set.size();  // Make sure this is done after resizing to ensure memory is contiguous.
                    MPI_Recv (recv_indexes, send_count, MPI_UINT64_T, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);
                    for (uint64_t i = 0; i < send_count; i++)
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
                    uint64_t send_count = indexes_set.size();
                    int recv_rank = (alias_recv_rank + world_size - 1) % world_size;
                    // printf("LEVEL %d: Rank %d (AR %d) send to %d\n", level, rank, alias_rank, recv_rank);

                    MPI_Ssend (&send_count,           1, MPI_UINT64_T, recv_rank, level, mpi_config->world);
                    MPI_Ssend (curr_indexes, send_count, MPI_UINT64_T, recv_rank, level, mpi_config->world);
                    have_data = false;
                }
                // if (!have_data) printf("LEVEL %d: Rank %d HAVE DATA %d \n", level, rank, have_data);
            }
        }
    }

    template<typename T>
    inline void MPI_GatherMap (MPI_Config *mpi_config, unordered_map<uint64_t, particle_aos<T>>& cell_particle_map, uint64_t *indexes, particle_aos<T> *indexed_fields, function<void(uint64_t, uint64_t **, particle_aos<T> **)> resize_fn)
    { 
        const int world_size = mpi_config->world_size;
        int rank             = mpi_config->rank;
        int alias_rank       = (rank + 1) % world_size;
        
        int rounded_world_size = (int)pow(2., ceil(log((double)world_size)/log(2.)));

        uint64_t *curr_indexes               = indexes; 
        particle_aos<T> *curr_indexed_fields = indexed_fields; 

        bool have_data = true;
        for ( int level = 2; level <= rounded_world_size ; level *= 2)
        {
            if (have_data)
            {
                bool reciever        = !(alias_rank % level);
                int alias_send_rank  = alias_rank + (level / 2);
                int alias_recv_rank  = alias_rank - (level / 2);

                if ( reciever )
                {
                    if (alias_send_rank >= world_size)  continue;

                    uint64_t send_count;
                    int send_rank = (alias_send_rank + world_size - 1) % world_size;
                    // printf("LEVEL %d: Rank %d (AR %d) recv from %d\n", level, rank, alias_rank, send_rank);
                    MPI_Recv (&send_count, 1, MPI_UINT64_T, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);
                    resize_fn(cell_particle_map.size() + send_count, &curr_indexes, &curr_indexed_fields);
                    uint64_t        *recv_indexes        = curr_indexes        + cell_particle_map.size();
                    particle_aos<T> *recv_indexed_fields = curr_indexed_fields + cell_particle_map.size();

                    MPI_Recv (recv_indexes,        send_count, MPI_UINT64_T,                       send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);
                    MPI_Recv (recv_indexed_fields, send_count, mpi_config->MPI_PARTICLE_STRUCTURE, send_rank, level, mpi_config->world, MPI_STATUS_IGNORE);

                    for (uint64_t i = 0; i < send_count; i++)
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
                    uint64_t send_count = cell_particle_map.size();
                    int recv_rank = (alias_recv_rank + world_size - 1) % world_size;
                    // printf("LEVEL %d: Rank %d (AR %d) send to %d\n", level, rank, alias_rank, recv_rank);

                    MPI_Ssend (&send_count,                  1, MPI_UINT64_T,                       recv_rank, level, mpi_config->world);
                    MPI_Ssend (curr_indexes,        send_count, MPI_UINT64_T,                       recv_rank, level, mpi_config->world);
                    MPI_Ssend (curr_indexed_fields, send_count, mpi_config->MPI_PARTICLE_STRUCTURE, recv_rank, level, mpi_config->world);

                    // printf("LEVEL %d: Rank %d (AR %d) send to %d\n", level, rank, alias_rank, recv_rank);
                    have_data = false;
                }

                // if (!have_data) printf("LEVEL %d: Rank %d HAVE DATA %d \n", level, rank, have_data);
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