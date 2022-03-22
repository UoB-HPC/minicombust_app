#pragma once


#include <string>  
#include <iostream> 
#include <sstream>   
#include <math.h>


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

    };

    template<typename T>
    inline vec<T> operator+(vec<T> a, vec<T> b) 
    {
        vec<T> sum = {a.x + b.x, a.y + b.y, a.z + b.z};
        return sum;
    }


    template<typename T>
    inline vec<T> operator-(vec<T> a, vec<T> b) 
    {
        vec<T> sum = {a.x - b.x, a.y - b.y, a.z - b.z};
        return sum;
    }

    template<typename T>
    inline vec<T> operator/(vec<T> a, T b) 
    {
        vec<T> sum = {a.x / b, a.y / b, a.z / b};
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
}


    
