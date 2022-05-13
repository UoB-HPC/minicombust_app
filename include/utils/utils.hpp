#pragma once


#include <string>  
#include <iostream> 
#include <sstream>   
#include <math.h>
#include <inttypes.h>
#include <vector>
#include <set>
#include <unordered_set>

#define PARTICLE_DEBUG 0
#define LOGGER 1
#define PARTICLE_SOLVER_DEBUG 0

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
    struct flow_aos {
        vec<T> vel;
        T pressure;
        T temp;
    };


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

    struct particle_logger {
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
    };
}


    
