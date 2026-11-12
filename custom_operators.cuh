#pragma once
#include "creator.cuh"

__host__ __device__ complex operator*(double a, complex b);

__host__ __device__ complex operator+(double a, complex b);

__host__ __device__ complex operator+(complex a, double b);

__host__ __device__ complex operator/(complex a, double b);

__host__ __device__ complex operator-(complex a, double b);

__host__ __device__ complex operator-(complex a, complex b);

__host__ __device__ complex operator+(complex a, complex b);

__host__ __device__ complex operator*(complex a, double b);

__host__ __device__ complex operator*(complex a, complex b);

__host__ __device__ complex& operator+=(complex& a, complex b);

__host__ __device__ inline complex complex_exp(complex z);

__host__ __device__ complex exp(complex a);