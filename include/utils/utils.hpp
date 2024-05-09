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
#define TIMER_OUTPUT_INTERVAL 100
#define FLOW_SOLVER_DEBUG 0
#define FLOW_SOLVER_FINE_TIME 0
#define FLOW_SOLVER_TIME 0
#define FLOW_SOLVER_LIMIT_GRAD 0
#define PARTICLE_SOLVER_DEBUG 0
#define FLOW 0
#define PARTICLE 1

#define TERBTE 0
#define TERBED 1
#define TEMP 2
#define FUEL 3
#define PROG  4
#define VARFU 5
#define VARPR 6

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

        inline T &operator[](int i) 
        {
            if ( i == 0 )  return this->x;
            if ( i == 1 )  return this->y;
            if ( i == 2 )  return this->z;
            printf("ERROR: Undefined struct index\n");
            exit(1);
        }
    };

    inline int get_prime_factors ( int n, int *prime_factors )
    {
        // cout << n << " prime factors: " ;

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

        // for (int x = 0;  x < nfactors; x++)
        //     cout << prime_factors[x] << " ";
    
        return nfactors;
    }

    inline int get_block_id(vec<uint64_t> pos, vec<uint64_t> dim)
    {
        return pos.z * dim.y * dim.x + pos.y * dim.x + pos.x;
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

    template <typename T>
    struct phi_vector
    {
        T *U;
        T *V;
        T *W;
        T *P;
		T *PP;
		T *TE;
		T *ED;
		T *TP;
		T *TEM;
		T *FUL;
		T *PRO;
		T *VARF;
		T *VARP;
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
    inline vec<T> normalise(vec<T> a)
    {
        return a / magnitude(a);
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
    inline void ptr_swap( T **ptr0, T **ptr1 )
    {
        T *ptr_tmp = *ptr0;
        *ptr0 = *ptr1;
        *ptr1 = ptr_tmp;
    }

    template<typename T> 
    inline T vector_cosangle(vec<T> a, vec<T> b)
    {
        return dot_product(a, b) / (magnitude(a) * magnitude(b));
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



    template <typename T>
    inline void check_flow_field_exit (const char *check_string, const flow_aos<T> *value, const flow_aos<T> *check_value, uint64_t position )
    {
        double epsilon = 1.e-20;
        if (fabs(value->temp - check_value->temp) > epsilon)              
        {
            printf("ERROR %s : Wrong temp value %f should be %f at %lu. DIFF = %.11f \n",          check_string,  value->temp,    check_value->temp,      position, fabs(value->temp - check_value->temp) ); 
            exit(1);
        }

        if (fabs(value->pressure - check_value->pressure) > epsilon)              
        {
            printf("ERROR %s : Wrong pres value %f should be %f at %lu. DIFF = %.11f \n",          check_string,  value->pressure,    check_value->pressure,  position, fabs(value->pressure - check_value->pressure) ); 
            exit(1);
        }

        if (fabs(value->vel.x - check_value->vel.x) > epsilon) 
        {
            printf("ERROR %s : Wrong velo value {%.10f y z} should be %f at %lu. DIFF = %.11f \n", check_string,  value->vel.x,  check_value->vel.x,       position, fabs(value->vel.x - check_value->vel.x) ); 
            exit(1);
        }

        if (fabs(value->vel.y - check_value->vel.y) > epsilon) 
        {
            printf("ERROR %s : Wrong velo value {x %.10f z} should be %f at %lu. DIFF = %.11f \n", check_string,  value->vel.y,  check_value->vel.y,       position, fabs(value->vel.y - check_value->vel.y) ); 
            exit(1);
        }

        if (fabs(value->vel.z - check_value->vel.z) > epsilon) 
        {
            printf("ERROR %s : Wrong velo value {x y %.10f} should be %f at %lu. DIFF = %.11f \n", check_string,  value->vel.z,  check_value->vel.z,       position, fabs(value->vel.z - check_value->vel.z) ); 
            exit(1);
        }
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
        double sent_cells_per_block;
        double sent_cells;
        double nodes_recieved;
        double useful_nodes_proportion;
    };

    struct Flow_Logger {
        double recieved_cells;
        double reduced_recieved_cells;
        double sent_nodes;
    };

    struct MPI_Config {
        int rank;
        int particle_flow_rank;
        int world_size;
        int particle_flow_world_size;

        MPI_Comm world;
        MPI_Comm particle_flow_world;

        int      *alias_rank;
        int      *one_flow_rank;
        int      *every_one_flow_rank;
        int      *one_flow_world_size;
        int      *every_one_flow_world_size;
        MPI_Comm *one_flow_world;
        MPI_Comm *every_one_flow_world;

        int node_rank;
        int node_world_size;
        MPI_Comm node_world;

        MPI_Win win_cells;
        MPI_Win win_cell_centers;
        MPI_Win win_cell_neighbours;
        MPI_Win win_points;
        MPI_Win win_cells_per_point;
        
        int solver_type;
        MPI_Datatype MPI_FLOW_STRUCTURE;
        MPI_Datatype MPI_PARTICLE_STRUCTURE;
        MPI_Datatype MPI_VEC_STRUCTURE;
        MPI_Op MPI_PARTICLE_OPERATION;
    };

    template<typename T> 
    inline void check_array_nan (const char *array_name, T *array, uint64_t length, MPI_Config *mpi_config, uint64_t timestep)
    {
        bool found_nan = false;
        for ( uint64_t i = 0; i < length; i++ )
        {
            if ( isnan(array[i]) )
            {
                found_nan = true;
                printf("T%lu Rank %d: NAN at %lu in %s\n", timestep, mpi_config->rank, i, array_name);
                break;
            }
        }

        MPI_Barrier(mpi_config->particle_flow_world);
        if (found_nan)
            exit(1);
    }

    inline void MPI_GatherSet (MPI_Config *mpi_config, uint64_t num_blocks, vector<unordered_set<uint64_t>>& indexes_sets, uint64_t **indexes, uint64_t *elements, function<void(uint64_t*, uint64_t ***)> resize_fn)
    { 
        const int *world_sizes = mpi_config->one_flow_world_size;
        int       *ranks       = mpi_config->one_flow_rank;
        int        alias_rank[num_blocks];
        for ( uint64_t b = 0; b < num_blocks; b++ )
            alias_rank[b] = (ranks[b] + 1) % world_sizes[b];
        
        int max_rounded_world_size = (mpi_config->solver_type == FLOW) ?  (int)pow(2., ceil(log((double)world_sizes[mpi_config->particle_flow_rank])/log(2.))) : 1;
        MPI_Allreduce( MPI_IN_PLACE, &max_rounded_world_size, 1, MPI_INT, MPI_MAX, mpi_config->world);
        
        if ( (world_sizes[mpi_config->particle_flow_rank] == 1) && (mpi_config->solver_type == FLOW) )  return;


        bool have_data[num_blocks];
        for ( uint64_t b = 0; b < num_blocks; b++ )
        {
            alias_rank[b]          = (ranks[b] + 1) % world_sizes[b];
            have_data[b]           = ((mpi_config->solver_type == PARTICLE && indexes_sets[b].size()) || (mpi_config->solver_type == FLOW && (uint64_t)mpi_config->particle_flow_rank == b &&(mpi_config->solver_type == FLOW)) );
        }
        uint64_t **curr_indexes = indexes; 

        uint64_t send_counts[num_blocks];
        uint64_t *recv_indexes[num_blocks];

        MPI_Request requests[num_blocks];

        bool posted_count[num_blocks];
        bool recieved_indexes[num_blocks];
        bool processed_block[num_blocks];

        for ( int level = 2; level <= max_rounded_world_size ; level *= 2)
        {
            
            uint64_t b = num_blocks - 1;
            bool all_processed = false;
            for (uint64_t bii = 0; bii < num_blocks; bii++)  
            {
                posted_count[bii]     = !have_data[bii];
                recieved_indexes[bii] = !have_data[bii];
                processed_block[bii]  = !have_data[bii];

                // printf("LEVEL %d | Rank %d started have_data[%d] = %d\n", level, mpi_config->rank, bii, have_data[bii]);
            }


            while (!all_processed)
            {
                b = (b + 1) % num_blocks;

                if (have_data[b])
                {
                    bool reciever         = !(alias_rank[b] % level);
                    int  alias_send_rank  = alias_rank[b] + (level / 2);
                    int  alias_recv_rank  = alias_rank[b] - (level / 2);

                    if (reciever)
                    {
                        // printf("LEVEL %d | Block %lu | Rank %d reciever = %d\n", level, b, mpi_config->rank, reciever);
                        if (alias_send_rank >= world_sizes[b])
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d sender>world\n", level, b, mpi_config->rank);
                            processed_block[b] = true;
                            all_processed = true;
                            for (uint64_t bii = 0; bii < num_blocks; bii++)  all_processed &= processed_block[bii];
                            continue;
                        }

                        int send_rank = (alias_send_rank + world_sizes[b] - 1) % world_sizes[b];

                        uint64_t flow_block_index = mpi_config->solver_type == PARTICLE ? b : 0;
                
                        if (!posted_count[b])  
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) recv from %d \n", level, b, mpi_config->rank, alias_rank[b], alias_send_rank);
                            MPI_Irecv (&send_counts[b], 1, MPI_UINT64_T, send_rank, level, mpi_config->one_flow_world[b], &requests[b]);
                            posted_count[b] = true;
                        }

                        int count_done;
                        MPI_Test(&requests[b], &count_done, MPI_STATUS_IGNORE);

                        if (count_done && !recieved_indexes[b])
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) async recv indexes from %d \n", level, b, mpi_config->rank, alias_rank[b], alias_send_rank);
                            elements[flow_block_index] = indexes_sets[flow_block_index].size() + send_counts[b];
                            resize_fn(elements, &curr_indexes);

                            recv_indexes[b] = curr_indexes[flow_block_index] + indexes_sets[flow_block_index].size();  // Make sure this is done after resizing to ensure correct pointer
                            MPI_Irecv (recv_indexes[b], send_counts[b], MPI_UINT64_T, send_rank, level, mpi_config->one_flow_world[b], &requests[b]);
                            recieved_indexes[b] = true;
                        }

                        int index_done = 0;
                        if (recieved_indexes[b])  MPI_Test(&requests[b], &index_done, MPI_STATUS_IGNORE);

                        if (index_done &&  recieved_indexes[b] && !processed_block[b])
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) recv processing %lu elements \n", level, b, mpi_config->rank, alias_rank[b], send_counts[b]);

                            for (uint64_t i = 0; i < send_counts[b]; i++)
                            {
                                if ( !indexes_sets[flow_block_index].count(recv_indexes[b][i]) )
                                {
                                    curr_indexes[flow_block_index][indexes_sets[flow_block_index].size()] = recv_indexes[b][i];
                                    indexes_sets[flow_block_index].insert(recv_indexes[b][i]);
                                }
                            }

                            processed_block[b] = true;
                        }
                        
                    }
                    else 
                    {
                        send_counts[b] = indexes_sets[b].size();
                        int recv_rank = (alias_recv_rank + world_sizes[b] - 1) % world_sizes[b];

                        if (!posted_count[b])  
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) send to %d send_count %lu\n", level, b, mpi_config->rank, alias_rank[b], alias_recv_rank, send_counts[b]);
                            MPI_Isend (&send_counts[b], 1, MPI_UINT64_T, recv_rank, level, mpi_config->one_flow_world[b], &requests[b]);
                            posted_count[b] = true;
                        }
                        
                        int count_done;
                        MPI_Test(&requests[b], &count_done, MPI_STATUS_IGNORE);

                        if (count_done && !recieved_indexes[b])
                        {
                            int recv_rank = (alias_recv_rank + world_sizes[b] - 1) % world_sizes[b];
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) async send indexes to %d send_count %lu\n", level, b, mpi_config->rank, alias_rank[b], alias_recv_rank, send_counts[b]);

                            MPI_Isend (curr_indexes[b], send_counts[b], MPI_UINT64_T, recv_rank, level, mpi_config->one_flow_world[b], &requests[b]);
                            recieved_indexes[b] = true;
                        }

                        int index_done = 0;
                        if (recieved_indexes[b])  MPI_Test(&requests[b], &index_done, MPI_STATUS_IGNORE);

                        if (index_done && recieved_indexes[b] && !processed_block[b])
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) send done \n", level, b, mpi_config->rank, alias_rank[b]);
                            have_data[b] = false;
                            processed_block[b] = true;
                        }
                    }

                }

                all_processed = true;
                for (uint64_t bii = 0; bii < num_blocks; bii++)  all_processed &= processed_block[bii];
            }

        }
    }

    template<typename T>
    inline void MPI_GatherMap (MPI_Config *mpi_config, const uint64_t num_blocks, vector<unordered_map<uint64_t, uint64_t>>& cell_particle_maps, uint64_t **indexes, particle_aos<T> **indexed_fields, uint64_t *elements, bool *async_locks, uint64_t *send_counts, uint64_t **recv_indexes, particle_aos<T> **recv_indexed_fields, MPI_Request *requests, function<void(uint64_t*, uint64_t ***, particle_aos<T> ***)> resize_fn)
    {
        int      *const world_sizes = mpi_config->one_flow_world_size;
        int      *const ranks       = mpi_config->one_flow_rank;
        int      *const alias_rank  = mpi_config->alias_rank;
        MPI_Comm *const worlds      = mpi_config->one_flow_world;

        int max_rounded_world_size = 1;
        if (mpi_config->solver_type == FLOW) // Note causes valgrind invalid read. IGNORE.
            max_rounded_world_size = (int)pow(2., ceil(log((double)world_sizes[mpi_config->particle_flow_rank])/log(2.)));

        MPI_Allreduce( MPI_IN_PLACE, &max_rounded_world_size, 1, MPI_INT, MPI_MAX, mpi_config->world);
        
        if ( (world_sizes[mpi_config->particle_flow_rank] == 1) && (mpi_config->solver_type == FLOW) )  return;

        
        bool *have_data        = async_locks + 0 * num_blocks;
        bool *posted_count     = async_locks + 1 * num_blocks;
        bool *recieved_indexes = async_locks + 2 * num_blocks;
        bool *processed_block  = async_locks + 3 * num_blocks;
        for ( uint64_t b = 0; b < num_blocks; b++ )
        {
            alias_rank[b]          = (ranks[b] + 1) % world_sizes[b];
            have_data[b]           = ((mpi_config->solver_type == PARTICLE && cell_particle_maps[b].size()) || (mpi_config->solver_type == FLOW && (uint64_t)mpi_config->particle_flow_rank == b) );
        }
        uint64_t **curr_indexes               = indexes; 
        particle_aos<T> **curr_indexed_fields = indexed_fields; 

        for ( int level = 2; level <= max_rounded_world_size ; level *= 2)
        {

            uint64_t b = num_blocks - 1;
            bool all_processed = false;
            for (uint64_t bii = 0; bii < num_blocks; bii++)  
            {
                posted_count[bii]     = !have_data[bii];
                recieved_indexes[bii] = !have_data[bii];
                processed_block[bii]  = !have_data[bii];

                // printf("LEVEL %d | Rank %d started have_data[%d] = %d\n", level, mpi_config->rank, bii, have_data[bii]);
            }

            while (!all_processed)
            {
                b = (b + 1) % num_blocks;

                if (have_data[b])
                {
                    bool reciever         = !(alias_rank[b] % level);
                    int  alias_send_rank  = alias_rank[b] + (level / 2);
                    int  alias_recv_rank  = alias_rank[b] - (level / 2);

                    if (reciever)
                    {
                        // printf("LEVEL %d | Block %lu | Rank %d reciever = %d\n", level, b, mpi_config->rank, reciever);
                        if (alias_send_rank >= world_sizes[b])
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d sender>world\n", level, b, mpi_config->rank);
                            processed_block[b] = true;
                            all_processed = true;
                            for (uint64_t bii = 0; bii < num_blocks; bii++)  all_processed &= processed_block[bii];
                            continue;
                        }

                        int send_rank = (alias_send_rank + world_sizes[b] - 1) % world_sizes[b];

                        uint64_t flow_block_index = mpi_config->solver_type == PARTICLE ? b : 0;
                
                        if (!posted_count[b])  
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) recv from %d \n", level, b, mpi_config->rank, alias_rank[b], send_rank);
                            MPI_Irecv (&send_counts[b], 1, MPI_UINT64_T, send_rank, level, worlds[b], &requests[3 * b]);
                            posted_count[b] = true;
                            continue;
                        }

                        int count_done;
                        MPI_Test(&requests[3 * b], &count_done, MPI_STATUSES_IGNORE);

                        if (count_done && !recieved_indexes[b])
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) async recv indexes from %d \n", level, b, mpi_config->rank, alias_rank[b], alias_send_rank);
                            elements[flow_block_index] = cell_particle_maps[flow_block_index].size() + send_counts[b];
                            resize_fn(elements, &curr_indexes, &curr_indexed_fields);
                            
                            recv_indexes[b]        = curr_indexes[flow_block_index]        + cell_particle_maps[flow_block_index].size();
                            recv_indexed_fields[b] = curr_indexed_fields[flow_block_index] + cell_particle_maps[flow_block_index].size();
                            MPI_Irecv (recv_indexes[b],        send_counts[b], MPI_UINT64_T,                       send_rank, level, worlds[b], &requests[3 * b + 1]);
                            MPI_Irecv (recv_indexed_fields[b], send_counts[b], mpi_config->MPI_PARTICLE_STRUCTURE, send_rank, level, worlds[b], &requests[3 * b + 2]);

                            recieved_indexes[b] = true;
                            continue;
                        }

                        int index_done = 0;
                        if (recieved_indexes[b])  MPI_Testall(2, &requests[3 * b + 1], &index_done, MPI_STATUSES_IGNORE);

                        if (index_done &&  recieved_indexes[b] && !processed_block[b])
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) recv processing %lu elements \n", level, b, mpi_config->rank, alias_rank[b], send_counts[b]);
                            for (uint64_t i = 0; i < send_counts[b]; i++)
                            {
                                const uint64_t cell = recv_indexes[b][i];
                                if ( cell_particle_maps[flow_block_index].count(cell) )
                                {
                                    const uint64_t index = cell_particle_maps[flow_block_index][cell];

                                    curr_indexed_fields[flow_block_index][index].momentum  += recv_indexed_fields[b][i].momentum;
                                    curr_indexed_fields[flow_block_index][index].energy    += recv_indexed_fields[b][i].energy;
                                    curr_indexed_fields[flow_block_index][index].fuel      += recv_indexed_fields[b][i].fuel;
                                }
                                else
                                {
                                    const uint64_t index = cell_particle_maps[flow_block_index].size();

                                    curr_indexes[flow_block_index][index]           = cell;
                                    curr_indexed_fields[flow_block_index][index]    = recv_indexed_fields[b][i];
                                    cell_particle_maps[flow_block_index][cell]      = index;
                                }
                            }
                            processed_block[b] = true;
                        }
                    }
                    else 
                    {
                        send_counts[b] = cell_particle_maps[b].size();
                        int recv_rank = (alias_recv_rank + world_sizes[b] - 1) % world_sizes[b];

                        
                        if (!posted_count[b])  
                        {
                            // printf("LEVEL %d | Block %lu | Rank %d (AR %d) send to   %d \n", level, b, mpi_config->rank, alias_rank[b], recv_rank);

                            MPI_Isend (&send_counts[b],                     1, MPI_UINT64_T,                       recv_rank, level, worlds[b], &requests[3 * b + 0]);
                            MPI_Isend (curr_indexes[b],        send_counts[b], MPI_UINT64_T,                       recv_rank, level, worlds[b], &requests[3 * b + 1]);
                            MPI_Isend (curr_indexed_fields[b], send_counts[b], mpi_config->MPI_PARTICLE_STRUCTURE, recv_rank, level, worlds[b], &requests[3 * b + 2]);
                            posted_count[b] = true;
                        }

                        int sends_done;
                        MPI_Testall(3, &requests[3 * b], &sends_done, MPI_STATUSES_IGNORE);

                        if (sends_done)
                        {
                            have_data[b] = false;
                            processed_block[b] = true;
                        }
                    }
                }
                all_processed = true;
                for (uint64_t bii = 0; bii < num_blocks; bii++)  all_processed &= processed_block[bii];
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
