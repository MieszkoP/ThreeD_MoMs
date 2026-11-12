#ifndef CREATOR_H
#define CREATOR_H


#include <iostream>
#include <complex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuComplex.h>
#include <math.h>
//#include <string>
//#define cmplx make_cuDoubleComplex
//#using complex = cuDoubleComplex;
#define SHIFT_POINT_DISTANCE_IF_NaN 0.001

namespace py = pybind11;

using complex = cuDoubleComplex;
using std_complex = std::complex<double>;

typedef py::array_t<double> numpy_array_real;
typedef py::array_t<int> numpy_array_int;
typedef py::array_t<std_complex> numpy_array;
typedef unsigned int uint;

constexpr double c_v = 299792458;  // [m/s]
constexpr double PI = 3.14159265358979323846;
constexpr double epsilon = 8.8541878e-12; // [F/m]
constexpr double mu = 1.2566371e-6; // [H/m]
constexpr double euler = M_E;

#define SEVEN_POINTS_Q_INIT { \
    {0.333333, 0.333333,0.333333,0.225}, \
    {0.05971587, 0.47014206,0.47014206,0.13239415},  \
    {0.47014206, 0.05971587,0.47014206,0.13239415},\
    {0.47014206, 0.47014206,0.05971587,0.13239415}, \
    {0.79742698, 0.10128650,0.10128650,0.12593918},\
    {0.10128650, 0.79742698,0.10128650,0.12593918},\
    {0.10128650, 0.10128650,0.79742698,0.12593918}\
};

constexpr int N_points_quad = 7;

template <typename T>
struct point_type
{
    T x;
    T y;
    T z;
    point_type() = default;
    point_type& operator=(const point_type& other) = default;

    __host__ __device__
    point_type(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }

    template <
    typename V = T,
    std::enable_if_t<!std::is_same_v<V, cuDoubleComplex>, int> = 0
>
    __host__ __device__ point_type(const point_type<double>& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
    }
    template <
    typename V = T,
    std::enable_if_t<std::is_same_v<V, cuDoubleComplex>, int> = 0
>
    __host__ __device__ point_type(const point_type<double>& other)
    : x(make_cuDoubleComplex(other.x, 0.0)),
          y(make_cuDoubleComplex(other.y, 0.0)),
          z(make_cuDoubleComplex(other.z, 0.0))
    {}

    template <typename U, typename V = T,  typename = std::enable_if_t<std::is_same_v<U, double> && std::is_same_v<V, cuDoubleComplex>>>
    __host__ __device__ point_type(U _x, U _y, U _z)
        : x(make_cuDoubleComplex(_x, 0.0)),
          y(make_cuDoubleComplex(_y, 0.0)),
          z(make_cuDoubleComplex(_z, 0.0))
    {}
};

typedef point_type<double> point;
typedef point_type<complex> sc_point;

template <typename T>
__host__ __device__ const T& index(const T* matrix, size_t i,size_t j, size_t J)
{
    return matrix[i*J+j];
}

template <typename T>
__host__ __device__ T& index(T* matrix, size_t i,size_t j, size_t J)
{
    return matrix[i*J+j];
}

typedef enum
{
    Far,
    Near,
    Self
} Proximity;

typedef enum
{
    L,
    nK
} equationType;

typedef struct 
{
    point LI;
    double LII;
    point KI;
    point KII;
}NearD_Type;

typedef struct 
{
    double LI;
    double LII;
}SelfD_Type;


typedef enum
{
    Paz,
    Pez,
    Pay,
    Pey,
    Pax,
    Pex,
} polType;

__host__ __device__ void WritePriximity(Proximity prox);
__host__ __device__ void mean_point(unsigned int v1, unsigned int v2, unsigned int v3, point& mean, const double* _verticies);
__host__ __device__ double distance(double* v1, double* v2);

template<typename T>
__host__ __device__ double distance(point_type<T> v1, point_type<T> v2)
{
    return sqrt(pow((v1.x-v2.x),2)+pow((v1.y-v2.y),2)+pow((v1.z-v2.z),2));
}

__host__ __device__ double distance(uint v1, uint v2, const double* _verticies);
__host__ __device__ Proximity checkProximity(uint fm_pm, uint fn_pm, const double* _verticies, const double* _faces, const uint* _face_curr, double freq);
__host__ __device__ double GetLengthOfEdge(uint fn_pn, const uint* _face_curr, const double* _faces, const double* _verticies);
__host__ __device__ void vector_sum(double* A, double* B, point &sum);

template<typename T>
__host__ __device__ point_type<T> vector_sum(point_type<T> A, point_type<T> B)
{
    point_type<T> sum;
    sum.x = A.x+B.x;
    sum.y = A.y+B.y;
    sum.z = A.z+B.z;
    return sum;
}

__host__ __device__ double vec_sc_m(double* A, double* B);

template<typename T>
__host__ __device__ T vec_sc_m(point_type<T> A, point_type<T> B)
{
    return A.x*B.x + A.y*B.y + A.z*B.z;
}

__host__ __device__ void vec_vec_m(double* A, double* B, double* result);

template<typename T>
__host__ __device__ point_type<T> vec_vec_m(point_type<T> A, point_type<T> B)
{
    point_type<T>  result;
    result.x = A.y*B.z - A.z*B.y;
    result.y = A.z*B.x - A.x*B.z;
    result.z = A.x*B.y - A.y*B.x;
    return result;
}

template<typename T>
__host__ __device__ point_type<T> vec_times_sc(point_type<T> A, T B)
{
    point_type<T>  result;
    result.x = A.x*B;
    result.y = A.y*B;
    result.z = A.z*B;
    return result;
}

template<typename T>
__host__ __device__ double vec_len(point_type<T> A)
{
    return sqrt(pow(A.x,2) + pow(A.y,2) + pow(A.z,2));
}

template<typename T>
__host__ __device__ void Rho(uint p, bool is_plus, uint v_n, point_type<T>& result, const double* _verticies, double* points_x, double* points_y, double* points_z)
{
    if(is_plus)
    {
        point v_n_arr = {index(_verticies, v_n, 0,3), index(_verticies, v_n, 1,3), index(_verticies, v_n, 2,3)};
        point r = {-points_x[p], -points_y[p], -points_z[p]};
        result = vector_sum(v_n_arr, r);
    }
    else
    {
        point v_n_arr = {-index(_verticies, v_n, 0,3), -index(_verticies, v_n, 1,3), -index(_verticies, v_n, 2,3)};
        point r = {points_x[p], points_y[p], points_z[p]};
        result = vector_sum(r, v_n_arr);
    }
}

__host__ __device__ double omega(double freq);

__host__ __device__ double k_f(double freq);


typedef struct
{
    uint triangle;
    uint v_n;
    uint isplus;
    uint v1;
    uint v2;
    uint v3;
    double L;
    double A;
    double points_x[7];
    double points_y[7];
    double points_z[7];
    point normal;
} fn_pm_params;

__host__ __device__ fn_pm_params getParamsFromFn_pm(uint fn_pm, const double* _verticies, const double* _faces, const uint* _face_curr);

template <typename T>
__host__ __device__ T sq(T x) {
    return x * x;
}

template <typename T>
__host__ __device__ T cube(T x) {
    return x * x * x;
}

#endif