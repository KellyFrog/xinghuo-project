#include "sum.cuh"
#include <cassert>
#include <cstdio>
#include <vector>

const int THREADS = 1024;

__global__ void KernalReduceSum(unsigned* a, std::size_t n, std::size_t dim, unsigned* b) {
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if(j < dim) {
		unsigned sum = 0;
#pragma unroll
		for(int i = 0; i < n; ++i) sum += a[i * dim + j];
		b[j] = sum;
	}
}

void ReduceSum(unsigned* a, std::size_t n, std::size_t dim, unsigned* b) {
	KernalReduceSum<<<(dim + THREADS - 1) / THREADS, THREADS>>>(a, n, dim, b);
}
