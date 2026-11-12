#include "creator.cuh"

void allocateMemory(void** pointer, size_t size)
{
    cudaError_t err = cudaMalloc(pointer, size);
    if (err != cudaSuccess) {
        py::print("cudaMalloc:", cudaGetErrorString(err), py::arg("flush") = true);
        return;
    }
}

void copyMemoryToGPU(void *dst, const void *src, size_t count)
{
    cudaError_t err = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        py::print("cudaMemcpy H2D:", cudaGetErrorString(err), py::arg("flush") = true);
        return;
    }
}

void copyMemoryFromGPU(void *dst, const void *src, size_t count)
{
    cudaError_t err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        py::print("cudaMemcpy D2H:", cudaGetErrorString(err), py::arg("flush") = true);
        return;
    }
}

void Synchronize()
{
    cudaError_t err = cudaDeviceSynchronize();
     if (err != cudaSuccess) {
        py::print("Kernel execution:", cudaGetErrorString(err), py::arg("flush") = true);
        return;
    }
}