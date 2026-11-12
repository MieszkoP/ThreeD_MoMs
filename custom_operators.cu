#include "creator.cuh"
#include "custom_operators.cuh"

__host__ __device__ complex operator*(double a, complex b)
{
    return make_cuDoubleComplex(
        a * cuCreal(b),
        a * cuCimag(b)
    );
}

__host__ __device__ complex operator+(double a, complex b)
{
    return make_cuDoubleComplex(
        a + cuCreal(b),
        cuCimag(b)
    );
}

__host__ __device__ complex operator+(complex a, double b)
{
    return make_cuDoubleComplex(
        b + cuCreal(a),
        cuCimag(a)
    );
}

__host__ __device__ complex operator/(complex a, double b)
{
    return make_cuDoubleComplex(
        cuCreal(a) /b,
        cuCimag(a) /b
    );
}

__host__ __device__ complex operator-(complex a, double b)
{
    return make_cuDoubleComplex(
        cuCreal(a) - b,
        cuCimag(a)
    );
}

__host__ __device__ complex operator-(complex a, complex b)
{
    return make_cuDoubleComplex(
        cuCreal(a) - cuCreal(b),
        cuCimag(a) - cuCimag(b)
    );
}

__host__ __device__ complex operator+(complex a, complex b)
{
    return make_cuDoubleComplex(
        cuCreal(a) + cuCreal(b),
        cuCimag(a) + cuCimag(b)
    );
}

__host__ __device__ complex operator*(complex a, double b)
{
    return make_cuDoubleComplex(
        cuCreal(a) * b,
        cuCimag(a) * b
    );
}

__host__ __device__ complex operator*(complex a, complex b)
{
    return cuCmul(a,b);
}

__host__ __device__ complex& operator+=(complex& a, complex b)
{
    a = a+b;
    return a;
}

__host__ __device__ inline complex complex_exp(complex z)
{
    double e = exp(cuCreal(z));

    return {
        e * cos(cuCimag(z)),
        e * sin(cuCimag(z))
    };
}

__host__ __device__ complex exp(complex a)
{
    return complex_exp(a);
}
