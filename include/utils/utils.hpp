#pragma once

#include <string>  
#include <iostream> 
#include <sstream>   

namespace minicombust::utils 
{

    template <typename T> 
    struct vec {
        T x;
        T y;
        T z;
    };

    template<typename T>
    vec<T> operator+(vec<T> a, vec<T> b) 
    {
        vec<T> sum = {a.x + b.x, a.y + b.y, a.z + b.z};
        return sum;
    }

    template<typename T>
    vec<T> operator-(vec<T> a, vec<T> b) 
    {
        vec<T> sum = {a.x - b.x, a.y - b.y, a.z - b.z};
        return sum;
    }

    template<typename T>
    vec<T> operator/(vec<T> a, T b) 
    {
        vec<T> sum = {a.x / b, a.y / b, a.z / b};
        return sum;
    }

    template<typename T>
    vec<T> operator*(vec<T> a, T b) 
    {
        vec<T> sum = {a.x * b, a.y * b, a.z * b};
        return sum;
    }

    template<typename T> 
    std::string print_vec(vec<T> v)
    {
        std::stringstream buffer;
        buffer << "{" << v.x << ", " << v.y << ", " << v.z << "}";
        return buffer.str();
    }
}
    