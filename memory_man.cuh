#pragma once
#include "creator.cuh"

void allocateMemory(void** pointer, size_t size);

void copyMemoryToGPU(void *dst, const void *src, size_t count);

void copyMemoryFromGPU(void *dst, const void *src, size_t count);

void Synchronize();